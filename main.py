import os, random
import numpy as np
import torch
import argparse
import wandb

from train import train

import pytorch_lightning as pl

def init_seeds(seed=9826):
    pl.seed_everything(seed)

# def init_seeds(seed=9826):
#     random.seed(seed)  # Python的随机性
#     os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
#     np.random.seed(seed)  # numpy的随机性
#     torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
#     torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
#     torch.backends.cudnn.deterministic = True # 选择确定性算法
#     torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
#     torch.backends.cudnn.enabled = False

def parsing_args(c):
    parser = argparse.ArgumentParser(description='msflow')
    parser.add_argument('--dataset', default='mvtec', type=str, 
                        choices=['mvtec', 'visa'], help='dataset name')
    parser.add_argument('--input-size', default=384, type=int)
    parser.add_argument('--pre-extract', action='store_true', default=False, 
                        help='extract features or not.')
    parser.add_argument('--mode', default='train', type=str, 
                        help='train or test.')
    parser.add_argument('--multi-class', action='store_true', default=False, 
                        help='multi-class training or not.')
    parser.add_argument('--amp-enable', action='store_true', default=False, 
                        help='use amp or not.')
    parser.add_argument('--wandb-enable', action='store_true', default=False, 
                        help='use wandb for result logging or not.')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume training or not.')
    parser.add_argument('--eval_ckpt', default='', type=str, 
                        help='checkpoint path for evaluation.')
    parser.add_argument('--class-names', default=['all'], type=str, nargs='+', 
                        help='class names for training')
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--batch-size', default=16, type=int, 
                        help='train batch size')
    parser.add_argument('--meta-epochs', default=30, type=int,
                        help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=2, type=int,
                        help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, 
                        help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, 
                        help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',
                        help='number of flow blocks used in parallel flows.')
    parser.add_argument('--condition-blocks', default=[0, 0, 0], type=int, metavar='L', nargs='+',
                        help='number of flow blocks with dynamic conv.')
    parser.add_argument('--c-conds', default=[64, 64, 64], type=int, metavar='L', nargs='+',
                        help='positional channel number of condition used in parallel flows.')
    parser.add_argument('--c-pos-conds', default=[64, 64, 64], type=int, metavar='L', nargs='+',
                        help='positional channel number of condition used in parallel flows.')
    parser.add_argument('--c-semantic-conds', default=[64, 64, 64], type=int, metavar='L', nargs='+',
                        help='semantic channel number of condition used in parallel flows.')
    parser.add_argument('--loc-eval', action='store_true', default=False, 
                        help='evaluate the loc auc score or not.')
    parser.add_argument('--pro-eval', action='store_true', default=False, 
                        help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=4, type=int, 
                        help='interval for pro evaluation.')
    
    parser.add_argument('--ratio-dynamic', default=4, type=int, 
                        help='shrunk ratio of dynamic conv.')
    parser.add_argument('--dim-cpc', default=256, type=int, 
                        help='dim of dynamic conv.')
    
    parser.add_argument('--quantize-enable', action='store_true', default=False, 
                        help='use quantize or not.')
    parser.add_argument('--quantize-type', type=str, default='residual', 
                        help='residual quantize or naive quantize.')
    parser.add_argument('--concat-pos', action='store_true', default=False, 
                        help='concat pe for cgpc quantization or not.')
    parser.add_argument('--reassign-quantize', action='store_true', default=False, 
                        help='reassign quantize or not.')
    parser.add_argument('--quantize-weight', default=1., type=float, 
                        help='weight for quantize loss.')
    parser.add_argument('--k-cgpc', default=512, type=int,
                        help='number of clusters for quantize of cond features.')
    parser.add_argument('--k-cpc', default=32, type=int,
                        help='number of clusters for quantize of dynamic features.')
    
    parser.add_argument('--mixed-gaussian', action='store_true', default=False, 
                        help='map to mixed gaussian or not.')
    parser.add_argument('--concat-cpc', action='store_true', default=False, 
                        help='concat cpc prototypes as cond features for flows or not.')
    
    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(c, k, v)
    
    if c.dataset == 'mvtec':
        from datasets import MVTEC_CLASS_NAMES
        setattr(c, 'data_path', './data/MVTec')
        if c.class_names == ['all']:
            setattr(c, 'class_names', MVTEC_CLASS_NAMES)
    elif c.dataset == 'visa':
        from datasets import VISA_CLASS_NAMES
        setattr(c, 'data_path', './data/VisA_pytorch/1cls')
        if c.class_names == ['all']:
            setattr(c, 'class_names', VISA_CLASS_NAMES)
        
    c.input_size = (c.input_size, c.input_size)

    return c

def main(c):
    c = parsing_args(c)
    init_seeds(seed=c.seed)
    c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    if c.multi_class:
        print('-+'*5, 'multi-class', '+-'*5)
        print('training on {} classes: {}'.format(len(c.class_names), ", ".join(c.class_names)))
        c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.dataset, 'multi-class')
        train(c)
    else:
        for class_name in c.class_names:
            c.class_name = class_name
            print('-+'*5, class_name, '+-'*5)
            c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.dataset, c.class_name)
            train(c)

if __name__ == '__main__':
    import default as c
    main(c)