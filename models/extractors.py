from .resnet import *
import timm

default_cfgs = {
    'convnext_xlarge_384_in22ft1k' : [256, 512, 1024, 2048],
    'convnextv2_base' : [128, 256, 512, 1024],
}

def build_extractor(c):
    if   c.extractor == 'resnet18':
        extractor = resnet18(pretrained=True, progress=True)
    elif c.extractor == 'resnet34':
        extractor = resnet34(pretrained=True, progress=True)
    elif c.extractor == 'resnet50':
        extractor = resnet50(pretrained=True, progress=True)
    elif c.extractor == 'resnext50_32x4d':
        extractor = resnext50_32x4d(pretrained=True, progress=True)
    elif c.extractor == 'wide_resnet50_2':
        extractor = wide_resnet50_2(pretrained=True, progress=True)
    elif c.extractor == 'resnext101_32x8d':
        extractor = resnext101_32x8d(pretrained=True, progress=True)
    elif c.extractor == 'convnext_xlarge_384_in22ft1k':
        # print('convnext_xlarge_384_in22ft1k picked')
        extractor = timm.create_model('convnext_xlarge_384_in22ft1k', pretrained=True, features_only=True)
    elif c.extractor == 'convnextv2_base':
        print('convnextv2_base picked')
        extractor = timm.create_model('convnextv2_base', pretrained=True, features_only=True)
    
    output_channels = []
    if 'wide' in c.extractor:
        for i in range(4):
            output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))
    elif 'resnext' in c.extractor:
        for i in range(4):
            # print(eval('extractor.layer{}'.format(i+1)))
            output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))
    elif c.extractor == 'convnext_xlarge_384_in22ft1k':
        for i in range(4):
            output_channels.append(default_cfgs['convnext_xlarge_384_in22ft1k'][i])
    elif c.extractor == 'convnextv2_base':
        for i in range(4):
            output_channels.append(default_cfgs['convnextv2_base'][i])
    else:
        for i in range(4):
            output_channels.append(extractor.eval('layer{}'.format(i+1))[-1].conv2.out_channels)
            
    print("Channels of extracted features:", output_channels)
    return extractor, output_channels