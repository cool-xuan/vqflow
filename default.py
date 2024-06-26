import torch
import os

# base
seed = 9826
gpu = '1'
device = torch.device("cuda")
mode = 'train'
eval_ckpt = ''
resume = False

# optimizer
meta_epochs = 25 # totally 100
sub_epochs = 4
lr = 1e-4
lr_decay_milestones = [70, 90]
lr_decay_gamma = 0.33
lr_warmup = True
lr_warmup_from = 0.1
lr_warmup_epochs = 5
batch_size = 16
workers = 16


# dataset
dataset = 'mvtec' # [mvtec, visa]
class_name = 'bottle'
input_size = (384, 384)
img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# model
extractor = 'convnextv2_base' # [resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2]
pool_type = 'avg'
c_conds = [64, 64, 64]
parallel_blocks = [2, 5, 8]
clamp_alpha = 1.9

# evaluation
top_k = 0.03
pro_eval = True
pro_eval_interval = 6

# results
work_dir = './work_dirs'
