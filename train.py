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
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc

from models.quantize import Quantize

def extract_features(c, extractor, inputs):
    if c.pre_extract:
        h_list = inputs
    else:
        h_list = extractor(*inputs)
        
    return h_list

def model_forward(c, extractor, parallel_flows, fusion_flow, inputs, multi_class_adapters=None):
    h_list = extract_features(c, extractor, inputs)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    semantic_encoder = getattr(multi_class_adapters, 'semantic_encoder', None)
    meta_controller = getattr(multi_class_adapters, 'meta_controller', None)
    upsamplers = getattr(multi_class_adapters, 'upsamplers', None)
    condition_adapters = getattr(multi_class_adapters, 'condition_adapters', None)
    weight_generator = getattr(multi_class_adapters, 'weight_generator', None)
    quantize_cond = getattr(multi_class_adapters, 'quantize_cond', None)
    quantize_dynamic = getattr(multi_class_adapters, 'quantize_dynamic', None)
    sigma_mu_generators = getattr(multi_class_adapters, 'sigma_mu_generators', None)

    semantic_embedding = semantic_encoder(h_list[-1]) # B, C, H, W 
    # semantic_embedding = semantic_encoder(h_list) # B, C, H, W 
    # if c.quantize_enable and quantize_cond is not None:
    #     semantic_embedding, diff_cond, _ = quantize_cond(semantic_embedding)
    # else:
    #     diff_cond = 0.
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    # semantic_embedding = semantic_encoder(avg_pool(h_list[-1])) # B, C, H, W 
    feat_meta = meta_controller(avg_pool(h_list[-1]).squeeze(-1).squeeze(-1)) # B, C
    if c.quantize_enable and quantize_dynamic is not None:
        feat_meta, diff_dynamic, emb_ind = quantize_dynamic(feat_meta.unsqueeze(-1).unsqueeze(-1))
        feat_meta = feat_meta.squeeze(-1).squeeze(-1)
    else:
        diff_dynamic = 0.
    
    weights_bias = weight_generator(feat_meta)

    z_list = []
    parallel_jac_list = []
    diff_cond = 0.
    dynamic_quantize_distribution = torch.zeros(c.k_dynamic).to(c.device)
    dynamic_quantize_distribution += torch.bincount(emb_ind.reshape(-1), minlength=c.k_dynamic)
    cond_quantize_distributions = {}
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list[-2::-1], parallel_flows[::-1], c.c_conds)):
        cond_quantize_distributions[idx] = torch.zeros(quantize_cond[idx].n_embed).to(c.device)
        y = pool_layer(h)
        B, _, H, W = y.shape
        pos_cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        for flow in parallel_flow:
            flow.subnet.condition = feat_meta
            flow.subnet.weights_bias = weights_bias
        if c.semantic_cond:
            class_embedding = feat_meta.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            _semantic_embedding = upsamplers[idx](semantic_embedding)
            if c.quantize_enable and quantize_cond is not None:
                if c.concat_pos:
                    #! residual quantize
                    class_pos_embedding = torch.cat([class_embedding, pos_cond], dim=1)
                    # _semantic_embedding = torch.cat([_semantic_embedding, pos_cond], dim=1)
                    residual = _semantic_embedding - class_pos_embedding
                    _residual, _diff_cond, emb_ind = quantize_cond[idx](residual)
                    _semantic_embedding = class_pos_embedding + _residual
                else:
                    #* indenpendent quantize
                    _semantic_embedding, _diff_cond, emb_ind = quantize_cond[idx](_semantic_embedding)
                diff_cond += _diff_cond.mean()
                cond_quantize_distributions[idx] += torch.bincount(emb_ind.reshape(-1), minlength=quantize_cond[idx].n_embed)
            if c.concat_dynamic:
                _semantic_embedding = torch.cat([_semantic_embedding, class_embedding], dim=1)
            semantic_cond = condition_adapters[idx](_semantic_embedding)
            cond = torch.cat([pos_cond, semantic_cond], dim=1)
            if idx == 0:
                cond_quantize_distributions[3] = torch.zeros(quantize_cond[-1].n_embed).to(c.device)
                _semantic_embedding = upsamplers[idx](semantic_embedding)
                if c.quantize_enable and quantize_cond is not None:
                    if c.concat_pos:
                        #! residual quantize
                        # class_embedding = torch.cat([class_embedding, pos_cond], dim=1)
                        # _semantic_embedding = torch.cat([_semantic_embedding, pos_cond], dim=1)
                        residual = _semantic_embedding - class_pos_embedding
                        _residual, _diff_cond, emb_ind = quantize_cond[-1](residual)
                        _semantic_embedding = class_pos_embedding + _residual
                    else:
                        #* indenpendent quantize
                        _semantic_embedding, _diff_cond, emb_ind = quantize_cond[-1](_semantic_embedding)
                    diff_cond += _diff_cond.mean()
                    cond_quantize_distributions[3] += torch.bincount(emb_ind.reshape(-1), minlength=quantize_cond[3].n_embed)
                if c.concat_dynamic:
                    _semantic_embedding = torch.cat([_semantic_embedding, class_embedding], dim=1)
                semantic_cond = condition_adapters[-1](_semantic_embedding)
                _cond = torch.cat([pos_cond, semantic_cond], dim=1)
        z, jac = parallel_flow(y, [cond, ])
        mu, sigma = sigma_mu_generators[::-1][idx](feat_meta)
        z = (z - mu) / torch.sqrt(sigma**2 + 1e-8)
        z_list.append(z)
        parallel_jac_list.append(jac)

    # fusion_flow.module_list[3].subnet_1.cross_convs.condition = feat_meta
    # fusion_flow.module_list[3].subnet_2.cross_convs.condition = feat_meta
    z_list, fuse_jac = fusion_flow(z_list[::-1], _cond)
    jac = fuse_jac + sum(parallel_jac_list)
    
    return z_list, jac, diff_dynamic+diff_cond, dynamic_quantize_distribution, cond_quantize_distributions

def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None, multi_class_adapters=None):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.train()
    multi_class_adapters = multi_class_adapters.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_dynamic_quantize_distribution = torch.zeros(c.k_dynamic).to(c.device)
        epoch_cond_quantize_distributions = [torch.zeros(quantize.n_embed).to(c.device) for quantize in multi_class_adapters['quantize_cond']]
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
                    z_list, jac, diff = model_forward(c, extractor, parallel_flows, fusion_flow, inputs, multi_class_adapters)
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
                z_list, jac, diff, dynamic_quantize_distribution, cond_quantize_distributions \
                    = model_forward(c, extractor, parallel_flows, fusion_flow, inputs, multi_class_adapters)
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

            epoch_dynamic_quantize_distribution += dynamic_quantize_distribution
            for i, cond_quantize_distribution in cond_quantize_distributions.items():
                epoch_cond_quantize_distributions[i] += cond_quantize_distribution

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
        _epoch_dynamic_quantize_distribution = epoch_dynamic_quantize_distribution / epoch_dynamic_quantize_distribution.sum()
        print('Dynamic Quantize Distribution:', '({})'.format(len(_epoch_dynamic_quantize_distribution)), ''.join([levels[int(x*7)] for x in _epoch_dynamic_quantize_distribution]))
        for i, cond_quantize_distribution in enumerate(epoch_cond_quantize_distributions):
            _cond_quantize_distribution = cond_quantize_distribution / cond_quantize_distribution.sum()
            print('Cond Quantize Distribution {} ({}):'.format(i, len(_cond_quantize_distribution)), ''.join([levels[int(x*7)] for x in _cond_quantize_distribution]))
        

def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, multi_class_adapters=None):
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()
    multi_class_adapters = multi_class_adapters.eval()
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

            z_list, jac, diff, dynamic_quantize_distribution, cond_quantize_distributions \
                    = model_forward(c, extractor, parallel_flows, fusion_flow, inputs, multi_class_adapters)

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
            project='65001-msflow', 
            group=c.version_name,
            name='multi_class' if c.multi_class else c.class_name)
        
    if c.dataset == 'mvtec':
        Dataset = MVTecFeatureDataset if c.pre_extract else MVTecDataset
    if c.dataset == 'visa':
        Dataset = VisADataset

    if c.multi_class:
        train_dataset = MultiClassDataset(c, is_train=True)
        test_datasets = {}
        for class_name in c.class_names:
            setattr(c, 'class_name', class_name)
            test_datasets[class_name] = Dataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True, drop_last=True)
        test_loaders = {k: torch.utils.data.DataLoader(v, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True) for k, v in test_datasets.items()}
    else:
        train_dataset = Dataset(c, is_train=True)
        test_dataset  = Dataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)
    params = list(fusion_flow.parameters())
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())

    channel_semantic = c.dim_meta
    if c.concat_pos:
        channel_semantic += c.c_conds[0]
    semantic_encoder = nn.Sequential(
        nn.Conv2d(output_channels[-1], channel_semantic, 3, 1, 1),
        nn.BatchNorm2d(channel_semantic),
        nn.ReLU(True),
    ).to(c.device)

    class semantic_encoder_ms(nn.Module):
        def __init__(self, in_channels, out_channel):
            super(semantic_encoder_ms, self).__init__()
            self.downsamplers = nn.ModuleList([
                nn.AvgPool2d(2**k, 2**k) for k in range(len(in_channels))])[::-1]
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[k], out_channel, 3, 1, 1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                ) for k in range(len(in_channels))])
            
            self.scalers = nn.Parameter(torch.ones(len(in_channels)))
            
        def forward(self, xs):
            ys = []
            for x, downsampler, conv, scaler in zip(xs, self.downsamplers, self.convs, self.scalers):
                ys.append(conv(downsampler(x)) * scaler)
            return sum(ys)
        
    # semantic_encoder = semantic_encoder_ms(output_channels, 256).to(c.device)

    meta_controller = nn.Sequential(
        nn.Linear(output_channels[-1], c.dim_meta//2),
        nn.BatchNorm1d(c.dim_meta//2),
        nn.ReLU(True),
        nn.Linear(c.dim_meta//2, c.dim_meta),
        nn.BatchNorm1d(c.dim_meta),
        nn.ReLU(True)
    ).to(c.device)
    
    upsamplers = nn.ModuleList([
        # nn.Sequential(
        #     nn.ConvTranspose2d(256, 256, 4, 2, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        # ).to(c.device)
        nn.Upsample(
            scale_factor=2**k, mode='bilinear', align_corners=False).to(c.device)
        for k in range(len(c.c_semantic_conds))
    ])
    
    channel_adapter = channel_semantic
    if c.concat_dynamic:
        channel_adapter += c.dim_meta
    condition_adapters = nn.ModuleList([
        nn.Conv2d(channel_adapter, c_semantic_cond, 3, 1, 1).to(c.device) for c_semantic_cond in c.c_semantic_conds
    ])
    condition_adapters.append(nn.Conv2d(channel_adapter, c.c_semantic_conds[-1], 3, 1, 1).to(c.device))
    
    weight_generator = nn.Sequential(
            nn.Linear(c.dim_meta, 32),
            nn.ReLU(True),
            nn.Linear(32, (16+1)*16),
        ).to(c.device)
    
    power_op = lambda x: 2 ** x
    linear_op = lambda x: x+1
    constant_op = lambda x: 1

    if c.compute_op == 'power':
        compute_op = power_op
    elif c.compute_op == 'linear':
        compute_op = linear_op
    else:
        compute_op = constant_op

    cond_quantizers = nn.ModuleList([
        Quantize(channel_semantic, c.k_cond * compute_op(k)).to(c.device) for k in range(3)
    ])
    cond_quantizers.append(Quantize(channel_semantic, c.k_cond).to(c.device))

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
        sigma_mu_generator(c.dim_meta, out_channels).to(c.device) for out_channels in output_channels[:-1]
    ])
    
    multi_class_adapters = nn.ModuleDict({
            'semantic_encoder': semantic_encoder, 
            'meta_controller': meta_controller,
            'upsamplers': upsamplers,
            'condition_adapters': condition_adapters, 
            'weight_generator': weight_generator, 
            # 'quantize_cond': Quantize(256, c.k_cond).to(c.device), 
            'quantize_cond': cond_quantizers, 
            'quantize_dynamic': Quantize(c.dim_meta, c.k_dynamic).to(c.device),
            'sigma_mu_generators': sigma_mu_generators
        })
    
    params += multi_class_adapters.parameters()
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)
    
    if c.multi_class:
        det_auroc_obss = {k: Score_Observer('Det.AUROC', c.meta_epochs, verbose=False) for k in test_loaders.keys()}
        loc_auroc_obss = {k: Score_Observer('Loc.AUROC', c.meta_epochs, verbose=False) for k in test_loaders.keys()}
        loc_pro_obss = {k: Score_Observer('Loc.PRO', c.meta_epochs, verbose=False) for k in test_loaders.keys()}

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        if c.multi_class:
            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    multi_class_test(c, test_loaders, extractor, parallel_flows, fusion_flow, multi_class_adapters, det_auroc_obs, loc_auroc_obs, loc_pro_obs, det_auroc_obss, loc_auroc_obss, loc_pro_obss, epoch, pro_eval)
        else:
            gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow, multi_class_adapters)

            anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.loc_eval, pro_eval)
        
        return
    
    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
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
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None, multi_class_adapters)
        
        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        if c.multi_class:
            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_auroc, best_loc_auroc, best_loc_pro = \
                    multi_class_test(c, test_loaders, extractor, parallel_flows, fusion_flow, multi_class_adapters, det_auroc_obs, loc_auroc_obs, loc_pro_obs, det_auroc_obss, loc_auroc_obss, loc_pro_obss, epoch, pro_eval)
        else:
            gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow, multi_class_adapters)

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

        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)

def multi_class_test(c, test_loaders, extractor, parallel_flows, fusion_flow, multi_class_adapters, det_auroc_obs, loc_auroc_obs, loc_pro_obs, det_auroc_obss, loc_auroc_obss, loc_pro_obss, epoch, pro_eval):
    det_aurocs = {}
    loc_aurocs = {}
    loc_pro_aucs = {}
    for k, v in test_loaders.items():
        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, v, extractor, parallel_flows, fusion_flow, multi_class_adapters)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
                
        # print('Class: {}'.format(k), '-'*50)
        det_auroc, loc_auroc, loc_pro_auc, _, _, _ = \
                    eval_det_loc(det_auroc_obss[k], loc_auroc_obss[k], loc_pro_obss[k], epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.loc_eval, pro_eval)
        det_aurocs[k] = det_auroc
        loc_aurocs[k] = loc_auroc
        loc_pro_aucs[k] = loc_pro_auc
    print('Multi Classes', '-'*50)
    best_det_auroc = det_auroc_obs.update(np.mean(list(det_aurocs.values())), epoch)
    best_loc_auroc = loc_auroc_obs.update(np.mean(list(loc_aurocs.values())), epoch)
    if pro_eval:
        best_loc_pro = loc_pro_obs.update(np.mean(list(loc_pro_aucs.values())), epoch)
    else:
        best_loc_pro = False

    return det_auroc, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro