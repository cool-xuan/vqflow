import os, random
import numpy as np
import torch
import argparse
import wandb

from train import train

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_args(c):
    parser = argparse.ArgumentParser(description='msflow')
    parser.add_argument('--dataset', default='mvtec', type=str, 
                        choices=['mvtec', 'visa'], help='dataset name')
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
    parser.add_argument('--meta-epochs', default=25, type=int,
                        help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=4, type=int,
                        help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, 
                        help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, 
                        help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',
                        help='number of flow blocks used in parallel flows.')
    parser.add_argument('--condition-blocks', default=[0, 0, 0], type=int, metavar='L', nargs='+',
                        help='number of flow blocks with dynamic conv.')
    parser.add_argument('--pro-eval', action='store_true', default=False, 
                        help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=4, type=int, 
                        help='interval for pro evaluation.')
    
    parser.add_argument('--semantic-cond', action='store_true', default=False, 
                        help='add semantic feature as cond for flow or not.')
    parser.add_argument('--ratio-dynamic', default=8, type=int, 
                        help='shrunk ratio of dynamic conv.')
    parser.add_argument('--dim-meta', default=512, type=int, 
                        help='dim of dynamic conv.')
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
        
    # c.input_size = (256, 256) if c.class_name == 'transistor' and not c.multi_class else (512, 512)
    # c.input_size = (256, 256)
    c.input_size = (512, 512)

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