from functools import partial
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, auc
from skimage.measure import label, regionprops

import numpy as np
import torch

from utils import rescale

def eval_det_auroc(det_auroc_obs, epoch, gt_label, anomaly_score):
    det_auroc = roc_auc_score(gt_label, anomaly_score) * 100
    best = det_auroc_obs.update(det_auroc, epoch)
    return det_auroc, best


def eval_loc_auroc(loc_auroc_obs, epoch, gt_mask, anomaly_score_map):
    
    loc_auroc = roc_auc_score(gt_mask.flatten(), anomaly_score_map.flatten()) * 100
    best = loc_auroc_obs.update(loc_auroc, epoch)
    return loc_auroc, best

def eval_seg_pro(loc_pro_obs, epoch, gt_mask, anomaly_score_map, max_step=800):
    expect_fpr = 0.3 # default 30%
    max_th = anomaly_score_map.max()
    min_th = anomaly_score_map.min()
    delta = (max_th - min_th) / max_step
    threds = np.arange(min_th, max_th, delta).tolist()

    pool = Pool(8)
    ret = pool.map(partial(single_process, anomaly_score_map, gt_mask), threds)
    pool.close()
    pros_mean = []
    fprs = []
    for pro_mean, fpr in ret:
        pros_mean.append(pro_mean)
        fprs.append(fpr)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)
    idx = fprs < expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    loc_pro_auc = auc(fprs_selected, pros_mean_selected) * 100
    best = loc_pro_obs.update(loc_pro_auc, epoch)

    return loc_pro_auc, best

def single_process(anomaly_score_map, gt_mask, thred):
    binary_score_maps = np.zeros_like(anomaly_score_map, dtype=np.bool)
    binary_score_maps[anomaly_score_map <= thred] = 0
    binary_score_maps[anomaly_score_map >  thred] = 1
    pro = []
    for binary_map, mask in zip(binary_score_maps, gt_mask):    # for i th image
        for region in regionprops(label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pro.append(tp_pixels / region.area)

    pros_mean = np.array(pro).mean()
    inverse_masks = 1 - gt_mask
    fpr = np.logical_and(inverse_masks, binary_score_maps).sum() / inverse_masks.sum()
    return pros_mean, fpr


def eval_det_loc(det_auroc_obs, det_auroc_obs_add, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, loc_eval, pro_eval):
    gt_label = np.asarray(gt_label_list, dtype=np.bool)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
    det_auroc, best_det_auroc = eval_det_auroc(det_auroc_obs, epoch, gt_label, anomaly_score)
    _anomaly_score_map_add = anomaly_score_map_add.reshape(
        anomaly_score_map_add.shape[0], -1
    ).copy()
    b, hw = _anomaly_score_map_add.shape
    top_k = int(hw * 0.03)
    _anomaly_score_map_add.sort(axis=-1)
    _anomaly_score_map_add = _anomaly_score_map_add[:, ::-1]
    anomaly_score_add = np.mean(
        _anomaly_score_map_add[:, :top_k],
        axis=1)
    det_auroc_add, _ = eval_det_auroc(det_auroc_obs_add, epoch, gt_label, anomaly_score_add)
    if loc_eval:
        loc_auroc, best_loc_auroc = eval_loc_auroc(loc_auroc_obs, epoch, gt_mask, anomaly_score_map_add)
    else:
        loc_auroc, best_loc_auroc = 0., False
    if pro_eval:
        loc_pro_auc, best_loc_pro = eval_seg_pro(loc_pro_obs, epoch, gt_mask, anomaly_score_map_mul)
    else:
        loc_pro_auc, best_loc_pro = 0., False

    return det_auroc, det_auroc_add, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro