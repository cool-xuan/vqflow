import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, MVTecFeatureDataset, VisADataset, MultiClassDataset
from models.extractors import build_extractor
from models.flow_models import build_vqflow_model
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc

from models.quantize import Quantize

def extract_features(c, extractor, inputs, mode='test'):
    if c.pre_extract and mode == 'train':
        h_list = inputs
    else:
        h_list = extractor(*inputs)
        
    return h_list

def model_forward(c, extractor, models, inputs, mode='test'):
    h_list = extract_features(c, extractor, inputs, mode)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    parallel_flows = models['parallel_flows']
    fusion_flow = models['fusion_flow']
    semantic_encoder = getattr(models, 'semantic_encoder', None)
    semantic_mlp = getattr(models, 'semantic_mlp', None)
    upsamplers = getattr(models, 'upsamplers', None)
    feature_mlps = getattr(models, 'feature_mlps', None)
    cspc_quantizers = getattr(models, 'cspc_quantizers', None)
    cpc_quantizer = getattr(models, 'cpc_quantizer', None)
    sigma_mu_generators = getattr(models, 'sigma_mu_generators', None)

    reduced_semantic_embedding = semantic_encoder(h_list[-1]) # B, C, H, W 
    h_top = h_list[-1]
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    semantic_vector = semantic_mlp(avg_pool(h_list[-1]).squeeze(-1).squeeze(-1)) # B, C
    if c.quantize_enable:
        conceptual_prototype, diff_dynamic, emb_ind = cpc_quantizer(semantic_vector.unsqueeze(-1).unsqueeze(-1))
        conceptual_prototype = conceptual_prototype.squeeze(-1).squeeze(-1)
    else:
        diff_dynamic = 0.

    z_list = []
    parallel_jac_list = []
    diff_cond = 0.
    prototype_quantize_distribution = torch.zeros(c.k_cpc).to(c.device)
    if c.quantize_enable:
        prototype_quantize_distribution += torch.bincount(emb_ind.reshape(-1), minlength=c.k_cpc)
    pattern_quantize_distributions = {}
    mus = []
    sigmas = []
    for idx, (h, parallel_flow, c_pos_cond) in enumerate(zip(h_list[:-1], parallel_flows, c.c_pos_conds)):
        pattern_quantize_distributions[idx] = torch.zeros(cspc_quantizers[idx].n_embed).to(c.device)
        h = pool_layer(h)
        B, _, H, W = h.shape
        pe = positionalencoding2d(c_pos_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        _h = feature_mlps[idx](h)
        _h = upsamplers[idx](reduced_semantic_embedding) + _h
        if c.quantize_enable and cspc_quantizers is not None:
            cond_prototype = conceptual_prototype.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            if c.no_pattern_quantize:
                _h = torch.cat([_h, pe], dim=1) if c.concat_pos else _h
            else:
                if c.quantize_type == 'residual':
                    #! residual quantize
                    pos_prototype = torch.cat([pe, cond_prototype], dim=1) if c.concat_pos else cond_prototype
                    residual = _h - pos_prototype
                    residual, _diff_cond, emb_ind = cspc_quantizers[idx](residual)
                    _h = pos_prototype + residual
                else:
                    #* indenpendent quantize
                    _h = torch.cat([_h, pe], dim=1) if c.concat_pos else _h
                    _h, _diff_cond, emb_ind = cspc_quantizers[idx](_h)
                diff_cond += _diff_cond.mean()
                if c.quantize_enable:
                    pattern_quantize_distributions[idx] += torch.bincount(emb_ind.reshape(-1), minlength=cspc_quantizers[idx].n_embed)
        parallel_cond = torch.cat([_h, cond_prototype], dim=1)           
        z, jac = parallel_flow(h, [parallel_cond, ])
        if c.mixed_gaussian:
            if c.quantize_enable:
                mu, sigma = sigma_mu_generators[idx](conceptual_prototype)
            else:
                mu, sigma = sigma_mu_generators[idx](semantic_vector)
            mus.append(mu)
            sigmas.append(sigma)
            z = (z - mu) / torch.sqrt(sigma**2 + 1e-8)
        z_list.append(z)
        parallel_jac_list.append(jac)

    pattern_quantize_distributions[3] = torch.zeros(cspc_quantizers[-1].n_embed).to(c.device)
    _h_top = feature_mlps[-1](h_top)
    _, _, H, W = _h_top.shape
    pe = positionalencoding2d(c.c_pos_conds[-1], H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
    if c.quantize_enable and cspc_quantizers is not None:
        cond_prototype = conceptual_prototype.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        if c.no_pattern_quantize:
            _h_top = torch.cat([_h_top, pe], dim=1) if c.concat_pos else _h_top
        else:
            if c.quantize_type == 'residual':
                #! residual quantize
                pos_prototype = torch.cat([pe, cond_prototype], dim=1) if c.concat_pos else cond_prototype
                residual = _h_top - pos_prototype
                residual, _diff_cond, emb_ind = cspc_quantizers[-1](residual)
                _h_top = pos_prototype + residual
            else:
                #* indenpendent quantize
                _h_top = torch.cat([_h_top, pe], dim=1) if c.concat_pos else _h_top
                _h_top, _diff_cond, emb_ind = cspc_quantizers[-1](_h_top)
            diff_cond += _diff_cond.mean()
            if c.quantize_enable:
                pattern_quantize_distributions[3] += torch.bincount(emb_ind.reshape(-1), minlength=cspc_quantizers[3].n_embed)
    fusion_cond = torch.cat([_h_top, cond_prototype], dim=1)
    z_list, fuse_jac = fusion_flow(z_list, fusion_cond)
    jac = fuse_jac + sum(parallel_jac_list)
    
    return z_list, jac, diff_dynamic+diff_cond, prototype_quantize_distribution, pattern_quantize_distributions

def train_meta_epoch(c, epoch, loader, extractor, models, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None):
    models = models.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_prototype_quantize_distribution = torch.zeros(c.k_cpc).to(c.device)
        epoch_pattern_quantize_distributions = [torch.zeros(quantize.n_embed).to(c.device) for quantize in models['cspc_quantizers']]
        epoch_loss = 0.
        image_count = 0
        for idx, data in enumerate(loader):
            cls = data[-1]
            mask = data[-2]
            label = data[-3]
            inputs = data[:-3]
            inputs = [x.to(c.device) for x in inputs]
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    z_list, jac, diff = model_forward(c, extractor, models, inputs, mode='train')
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                    if c.quantize_enable:
                        loss += diff * c.quantize_weight
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list, jac, diff, prototype_quantize_distribution, pattern_quantize_distributions \
                    = model_forward(c, extractor, models, inputs)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                loss = loss - jac
                loss = loss.mean()
                if c.quantize_enable:
                    loss += diff * c.quantize_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            image_count += label.shape[0]

            epoch_prototype_quantize_distribution += prototype_quantize_distribution
            for i, pattern_quantize_distribution in pattern_quantize_distributions.items():
                epoch_pattern_quantize_distributions[i] += pattern_quantize_distribution

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                epoch, sub_epoch, mean_epoch_loss, lr))
        
        levels = "▁▂▃▄▅▆▇█"
        if epoch_prototype_quantize_distribution.sum() != 0:
            _epoch_prototype_quantize_distribution = epoch_prototype_quantize_distribution / epoch_prototype_quantize_distribution.sum()
            print('Dynamic Quantize Distribution:', '({})'.format(len(_epoch_prototype_quantize_distribution)), ''.join([levels[int(x*7)] for x in _epoch_prototype_quantize_distribution]))
        for i, pattern_quantize_distribution in enumerate(epoch_pattern_quantize_distributions):
            if pattern_quantize_distribution.sum() != 0:
                _pattern_quantize_distribution = pattern_quantize_distribution / pattern_quantize_distribution.sum()
                print('Cond Quantize Distribution {} ({}):'.format(i, len(_pattern_quantize_distribution)), ''.join([levels[int(x*7)] for x in _pattern_quantize_distribution]))
    if c.reassign_quantize:
        models['cpc_quantizer'].reAssign(epoch_prototype_quantize_distribution)
        for i, pattern_quantize_distribution in enumerate(epoch_pattern_quantize_distributions):
            models['cspc_quantizers'][i].reAssign(pattern_quantize_distribution)
        

def inference_meta_epoch(c, epoch, loader, extractor, models):
    models = models.eval()
    parallel_flows = models['parallel_flows']
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    start = time.time()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            cls = data[-1]
            mask = data[-2]
            label = data[-3]
            inputs = data[:-3]
            inputs = [x.to(c.device) for x in inputs]
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            z_list, jac, diff, prototype_quantize_distribution, pattern_quantize_distributions \
                    = model_forward(c, extractor, models, inputs, mode='test')

            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = - 0.5 * torch.mean(z**2, 1)
                outputs_list[lvl].append(logp)
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))

            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += label.shape[0]

        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_list, size_list


def train(c):
    
    if c.wandb_enable:
        wandb.finish()
        wandb.init(
            project='vqflow', 
            group=c.version_name,
            name='multi_class' if c.multi_class else c.class_name)
        
    if c.dataset == 'mvtec':
        if c.pre_extract:
            Dataset = MVTecFeatureDataset
            testDataset = MVTecDataset
        else:
            Dataset = MVTecDataset
            testDataset = MVTecDataset
        
    if c.dataset == 'visa':
        Dataset = VisADataset
        testDataset = VisADataset

    if c.multi_class:
        train_dataset = MultiClassDataset(c, is_train=True)
        test_datasets = {}
        for class_name in c.class_names:
            setattr(c, 'class_name', class_name)
            test_datasets[class_name] = testDataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True, drop_last=True)
        test_loaders = {k: torch.utils.data.DataLoader(v, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True) for k, v in test_datasets.items()}
    else:
        train_dataset = Dataset(c, is_train=True)
        test_dataset  = Dataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()

    semantic_mlp = nn.Sequential(
        nn.Linear(output_channels[-1], c.dim_cpc),
        nn.BatchNorm1d(c.dim_cpc),
        nn.ReLU(True),
        nn.Linear(c.dim_cpc, c.dim_cpc),
        nn.BatchNorm1d(c.dim_cpc),
        nn.ReLU(True)
    ).to(c.device)

    cpc_quantizer = Quantize(c.dim_cpc, c.k_cpc, thresh=1e-4).to(c.device)

    dim_cspc = c.dim_cpc
    if c.concat_pos and c.quantize_type == 'residual':
        dim_cspc += c.c_pos_conds[0]
    
    feature_mlps = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(output_channel, dim_cspc, 1, 1, 0),
            nn.BatchNorm2d(dim_cspc),
            nn.ReLU(True),
            nn.Conv2d(dim_cspc, dim_cspc, 1, 1, 0),
            nn.BatchNorm2d(dim_cspc),
            nn.ReLU(True),
        ).to(c.device)
        for output_channel in output_channels
    ])

    if c.concat_pos and c.quantize_type == 'naive':
        dim_cspc += c.c_pos_conds[0]

    cspc_quantizers = nn.ModuleList([
        Quantize(dim_cspc, c.k_cspc, thresh=1e-6).to(c.device) for k in range(4)
    ])

    for i in range(len(c.c_conds)):
        c.c_conds[i] = dim_cspc + c.dim_cpc
    
    if c.concat_pos and c.quantize_type == 'naive':
        dim_cspc -= c.c_pos_conds[0]

    class sigma_mu_generator(nn.Module):

        def __init__(self, in_channels, out_channels):
            super(sigma_mu_generator, self).__init__()
            self.sigma = nn.Linear(in_channels, out_channels)
            self.mu = nn.Linear(in_channels, out_channels)

        def forward(self, x):
            sigma = self.sigma(x).unsqueeze(-1).unsqueeze(-1)
            mu = self.mu(x).unsqueeze(-1).unsqueeze(-1)
            return sigma, mu
        
    sigma_mu_generators = nn.ModuleList([
        sigma_mu_generator(c.dim_cpc, out_channel).to(c.device) for out_channel in output_channels[:-1]
    ])

    parallel_flows, fusion_flow = build_vqflow_model(c, output_channels)
    parallel_flows = nn.ModuleList(parallel_flows).to(c.device)
    fusion_flow = fusion_flow.to(c.device)

    semantic_encoder = nn.Sequential(
        nn.Conv2d(output_channels[-1], dim_cspc, 1, 1, 0),
        nn.BatchNorm2d(dim_cspc),
        nn.ReLU(True),
    ).to(c.device)

    upsamplers = nn.ModuleList([
        nn.Sequential(
            nn.Upsample(
                scale_factor=2**k, mode='bilinear', align_corners=False),
            nn.Conv2d(dim_cspc, dim_cspc, 1, 1, 0),
        ).to(c.device)
        for k in range(len(c.c_conds))
    ])[::-1]   
    
    models = nn.ModuleDict({
            'parallel_flows': parallel_flows,
            'fusion_flow': fusion_flow,
            'semantic_encoder': semantic_encoder, 
            'semantic_mlp': semantic_mlp,
            'upsamplers': upsamplers,
            'feature_mlps': feature_mlps, 
            'cspc_quantizers': cspc_quantizers, 
            'cpc_quantizer': cpc_quantizer,
            'sigma_mu_generators': sigma_mu_generators
        })
    
    params = list(models.parameters())
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC.mul', c.meta_epochs)
    det_auroc_obs_add = Score_Observer('Det.AUROC.add', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)
    
    if c.multi_class:
        det_auroc_obss = {k: Score_Observer('Det.AUROC.mul', c.meta_epochs, verbose=True) for k in test_loaders.keys()}
        det_auroc_obss_add = {k: Score_Observer('Det.AUROC.add', c.meta_epochs, verbose=True) for k in test_loaders.keys()}
        loc_auroc_obss = {k: Score_Observer('Loc.AUROC', c.meta_epochs, verbose=False) for k in test_loaders.keys()}
        loc_pro_obss = {k: Score_Observer('Loc.PRO', c.meta_epochs, verbose=False) for k in test_loaders.keys()}

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(models, c.eval_ckpt)
        epoch = start_epoch + 1
        if c.multi_class:
            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    multi_class_test(c, test_loaders, extractor, models, det_auroc_obs, det_auroc_obs_add, loc_auroc_obs, loc_pro_obs, det_auroc_obss, det_auroc_obss_add, loc_auroc_obss, loc_pro_obss, epoch, c.pro_eval)
        else:
            gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, models)

            anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    eval_det_loc(det_auroc_obs, det_auroc_obs_add, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.loc_eval, c.pro_eval)
        
        return
    
    if c.resume:
        last_epoch = load_weights(models, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor, models, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None)
        
        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        if c.multi_class:
            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    multi_class_test(c, test_loaders, extractor, models, det_auroc_obs, det_auroc_obs_add, loc_auroc_obs, loc_pro_obs, det_auroc_obss, det_auroc_obss_add, loc_auroc_obss, loc_pro_obss, epoch, pro_eval)
        else:
            gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, models)

            anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.loc_eval, pro_eval)
            
            if c.wandb_enable:
                wandb.log(
                    {
                        'Det.AUROC': det_auroc,
                        'Loc.AUROC': loc_auroc,
                        'Loc.PRO': loc_pro_auc
                    },
                    step=epoch
                )

        save_weights(epoch, models, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, models, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, models, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, models, 'best_loc_pro', c.ckpt_dir)

def multi_class_test(c, test_loaders, extractor, models, det_auroc_obs, det_auroc_obs_add, loc_auroc_obs, loc_pro_obs, det_auroc_obss, det_auroc_obss_add, loc_auroc_obss, loc_pro_obss, epoch, pro_eval):
    det_aurocs = {}
    det_aurocs_add = {}
    loc_aurocs = {}
    loc_pro_aucs = {}
    for k, v in test_loaders.items():
        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, v, extractor, models)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
                
        # print('Class: {}'.format(k), '-'*50)
        det_auroc, det_auroc_add, loc_auroc, loc_pro_auc, _, _, _ = \
                    eval_det_loc(det_auroc_obss[k], det_auroc_obss_add[k], loc_auroc_obss[k], loc_pro_obss[k], epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.loc_eval, pro_eval)
        det_aurocs[k] = det_auroc
        det_aurocs_add[k] = det_auroc_add
        loc_aurocs[k] = loc_auroc
        loc_pro_aucs[k] = loc_pro_auc
    print('Multi Classes', '-'*50)
    best_det_auroc = det_auroc_obs.update(np.mean(list(det_aurocs.values())), epoch)
    best_det_auroc_add = det_auroc_obs_add.update(np.mean(list(det_aurocs_add.values())), epoch)
    best_det_auroc = best_det_auroc or best_det_auroc_add
    best_loc_auroc = loc_auroc_obs.update(np.mean(list(loc_aurocs.values())), epoch)
    if pro_eval:
        best_loc_pro = loc_pro_obs.update(np.mean(list(loc_pro_aucs.values())), epoch)
    else:
        best_loc_pro = False

    return det_auroc, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro