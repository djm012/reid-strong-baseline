# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from config import cfg
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnet_qrelu import *


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

# class ExpandTemporalDim(nn.Module):
#     def __init__(self, T):
#         super().__init__()
#         self.T = T

#     def forward(self, x_seq: torch.Tensor):
#         # print('----------------expandself.T:',self.T)
#         y_shape = [self.T, int(x_seq.shape[0]/self.T)]
#         y_shape.extend(x_seq.shape[1:])
#         return x_seq.view(y_shape)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        # 添加qrelu
        elif model_name == 'resnet18_qrelu':
            self.in_planes = 512
            self.base = ResNet_qReLU(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2],T=cfg.MODEL.T)
        elif model_name == 'resnet34_qrelu':
            self.in_planes = 512
            self.base = ResNet_qReLU(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3],T=cfg.MODEL.T)
        elif model_name == 'resnet50_qrelu':
            self.base = ResNet_qReLU(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],T=cfg.MODEL.T)
            print("--------------current_T--------------",cfg.MODEL.T)
        elif model_name == 'resnet101_qrelu':
            self.base = ResNet_qReLU(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3],T=cfg.MODEL.T)
        elif model_name == 'resnet152_qrelu':
            self.base = ResNet_qReLU(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3],T=cfg.MODEL.T)
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':

            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            # print(feat.shape)

            # 测试时用
            # if cfg.MODEL.T > 0:
            #     expand = ExpandTemporalDim(T=cfg.MODEL.T)
            #     feat = expand(feat)
            #     feat = feat.mean(0)
            #     print(feat.shape)
                
                # print('---------------------------running bottleneck---------------------------')

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     print(type(param_dict))
    #     for i in param_dict:
    #         if 'classifier' in i:
    #             continue
    #         self.state_dict()[i].copy_(param_dict[i])

    def load_param(self, trained_path):
        try:
            param_dict = torch.load(trained_path)
            if isinstance(param_dict, dict):
                # 如果是字典格式的state_dict
                for i in param_dict:
                    if 'classifier' in i:
                        continue
                    self.state_dict()[i].copy_(param_dict[i])
            elif isinstance(param_dict, nn.Module):
                # 如果保存的是整个模型
                param_dict = param_dict.state_dict()
                for i in param_dict:
                    if 'classifier' in i:
                        continue
                    self.state_dict()[i].copy_(param_dict[i])
            else:
                raise TypeError(f"不支持的参数类型: {type(param_dict)}")
                
        except Exception as e:
            print(f"加载模型参数时发生错误: {str(e)}")
            raise
