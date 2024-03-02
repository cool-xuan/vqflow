from functools import partial

import math

import torch
from torch import nn
from torch.nn import functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .freia_utils import FusionCouplingLayer



def subnet_conv(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, dims_in, 3, 1, 1), nn.ReLU(True), nn.Conv2d(dims_in, dims_out, 3, 1, 1))

def subnet_conv_bn(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, dims_in, 3, 1, 1), nn.BatchNorm2d(dims_in), nn.ReLU(True), nn.Conv2d(dims_in, dims_out, 3, 1, 1))

class subnet_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_mid = dim_in
        self.conv1 = nn.Conv2d(dim_in, dim_mid, 3, 1, 1)
        self.ln = nn.LayerNorm(dim_mid)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim_mid, dim_out, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)

        return out
    
class subnet_bottleneck_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_mid = 16
        self.conv1 = nn.Conv2d(dim_in, dim_mid, 3, 1, 1)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv2 = nn.Conv2d(dim_mid, dim_mid, 1, 1, 0)
        self.relu = nn.ReLU(True)
        self.ln2 = nn.LayerNorm(dim_mid)
        self.conv3 = nn.Conv2d(dim_mid, dim_out, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv3(out)

        return out
    
class subnet_dynamic_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out, dim_cpc=512, ratio_dynamic=8):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_mid = dim_in // ratio_dynamic
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_mid, 3, 1, 1)
        self.ln1 = nn.LayerNorm(self.dim_mid)
        self.relu = nn.ReLU(True)
        self.weight_generator = nn.Sequential(
            nn.Linear(dim_cpc, self.dim_mid * 2),
            # nn.LayerNorm(self.dim_mid * 2),
            nn.BatchNorm1d(self.dim_mid * 2),
            nn.ReLU(True),
            nn.Linear(self.dim_mid * 2, (self.dim_mid+1)*self.dim_mid),
        )
        self.ln2 = nn.LayerNorm(self.dim_mid)
        self.conv3 = nn.Conv2d(self.dim_mid, self.dim_out, 3, 1, 1)
        
    def dynamic_conv(self, x):
        b = x.shape[0]
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        weights_bias = self.weight_generator(self.condition)
        weights, bias = weights_bias[:, :self.dim_mid**2], weights_bias[:, self.dim_mid**2:]
        x = F.conv2d(x,
                     weights.reshape(b*self.dim_mid, self.dim_mid, 1, 1), 
                     bias=bias.reshape(b*self.dim_mid),
                     groups=b)
        
        return x.reshape(b, self.dim_mid, x.shape[2], x.shape[3])

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.dynamic_conv(out)
        out = self.ln2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv3(out)

        return out

class subnet_dynamic_conv_ln_shared(nn.Module):

    def __init__(self, dim_in, dim_out, dim_cpc=512, ratio_dynamic=8):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_mid = dim_in // ratio_dynamic
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_mid, 3, 1, 1)
        self.ln1 = nn.LayerNorm(self.dim_mid)
        self.relu = nn.ReLU(True)
        self.ln2 = nn.LayerNorm(self.dim_mid)
        self.conv3 = nn.Conv2d(self.dim_mid, self.dim_out, 3, 1, 1)
        
    def dynamic_conv(self, x):
        b = x.shape[0]
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        weights_bias = self.weights_bias
        weights, bias = weights_bias[:, :self.dim_mid**2], weights_bias[:, self.dim_mid**2:]
        x = F.conv2d(x,
                     weights.reshape(b*self.dim_mid, self.dim_mid, 1, 1), 
                     bias=bias.reshape(b*self.dim_mid),
                     groups=b)
        
        return x.reshape(b, self.dim_mid, x.shape[2], x.shape[3])

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.dynamic_conv(out)
        out = self.ln2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv3(out)

        return out

class subnet_dynamic_conv_bn(nn.Module):

    def __init__(self, dim_in, dim_out, dim_cpc=512, ratio_dynamic=8):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_mid = dim_in // ratio_dynamic
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_mid, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.dim_mid)
        self.relu = nn.ReLU(True)
        self.weight_generator = nn.Sequential(
            nn.Linear(dim_cpc, self.dim_mid * 2),
            nn.BatchNorm1d(self.dim_mid * 2),
            nn.ReLU(True),
            nn.Linear(self.dim_mid * 2, (self.dim_mid+1)*self.dim_mid),
        )
        self.bn2 = nn.BatchNorm2d(self.dim_mid)
        self.conv3 = nn.Conv2d(self.dim_mid, self.dim_out, 3, 1, 1)
        
    def dynamic_conv(self, x):
        b = x.shape[0]
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        weights_bias = self.weight_generator(self.condition)
        weights, bias = weights_bias[:, :self.dim_mid**2], weights_bias[:, self.dim_mid**2:]
        x = F.conv2d(x,
                     weights.reshape(b*self.dim_mid, self.dim_mid, 1, 1), 
                     bias=bias.reshape(b*self.dim_mid),
                     groups=b)
        
        return x.reshape(b, self.dim_mid, x.shape[2], x.shape[3])

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dynamic_conv(out)
        out = self.conv3(self.relu(self.bn2(out)))

        return out

def single_parallel_flows(c_feat, c_cond, n_block, n_cond_block, clamp_alpha, subnet=subnet_conv_ln, subnet_cond=subnet_dynamic_conv_ln):
    flows = Ff.SequenceINN(c_feat, 1, 1)
    print('Build parallel flows: channels:{}, block:{}, cond_block:{}, cond:{}'.format(c_feat, n_block, n_cond_block, c_cond))
    for k in range(n_block):
        _subnet = subnet_cond if k < n_cond_block else subnet # first dynamic conv
        flows.append(Fm.AllInOneBlock, cond=0, cond_shape=(c_cond, 1, 1), subnet_constructor=_subnet, affine_clamping=clamp_alpha,
            global_affine_type='SOFTPLUS')
    return flows

def build_msflow_model(c, c_feats):
    c_conds = c.c_conds
    n_blocks = c.parallel_blocks
    n_cond_blocks = c.condition_blocks
    clamp_alpha = c.clamp_alpha
    parallel_flows = []
    for c_feat, c_cond, n_block, n_cond_block in zip(c_feats, c_conds, n_blocks, n_cond_blocks):
        parallel_flows.append(
            single_parallel_flows(c_feat, c_cond, n_block, n_cond_block, clamp_alpha, subnet=subnet_conv_ln, subnet_cond=partial(subnet_dynamic_conv_ln, ratio_dynamic=c.ratio_dynamic, dim_cpc=c.dim_cpc)))
    
    c_feats = c_feats[:-1]
    print("Build fusion flow with channels", c_feats)
    nodes = list()
    n_inputs = len(c_feats)
    for idx, c_feat in enumerate(c_feats):
        nodes.append(Ff.InputNode(c_feat, 1, 1, name='input{}'.format(idx)))
    for idx in range(n_inputs):
        nodes.append(Ff.Node(nodes[-n_inputs], Fm.PermuteRandom, {}, name='permute_{}'.format(idx)))
    conditions_fusion = Ff.ConditionNode(c_cond, 1, 1, name='condition_fusion')
    nodes.append(Ff.Node([(nodes[-n_inputs+i], 0) for i in range(n_inputs)], FusionCouplingLayer, {'clamp': clamp_alpha, }, conditions=conditions_fusion, name='fusion flow'))
    for idx, c_feat in enumerate(c_feats):
        nodes.append(Ff.OutputNode(eval('nodes[-idx-1].out{}'.format(idx)), name='output_{}'.format(idx)))
    nodes.append(conditions_fusion)
    fusion_flow = Ff.GraphINN(nodes)

    return parallel_flows, fusion_flow
