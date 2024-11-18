# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .backbones.resnet_qrelu import qReLU

def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    # return model

    model.base.set_T(cfg.MODEL.T)
    print(f'Setting model with T={cfg.MODEL.T}')
    # if 'qrelu' in cfg.MODEL.NAME.lower():
        # if hasattr(model.base, 'set_T'):
        #     all_modules = list(model.base.named_modules())
        #     for name, module in all_modules:
        #         # print('----------------model_name--------------', name)
        #         if isinstance(module, qReLU):
        #             # if ('layer1' in name and ('qrelu1' in name or 'qrelu2' in name or 'qrelu3' in name)) or ('layer2' in name and ('qrelu1' in name or 'qrelu2' in name or 'qrelu3' in name)) or ('layer3' in name and ('qrelu1' in name or 'qrelu2' in name or 'qrelu3' in name)) or name == 'qrelu':
        #             if ('layer1' in name and ('qrelu1' in name or 'qrelu2' in name or 'qrelu3' in name)) or ('layer2' in name and ('qrelu1' in name or 'qrelu2' in name or 'qrelu3' in name)) or name == 'qrelu':
        #             # if ('layer1' in name and ('qrelu1' in name or 'qrelu2' in name or 'qrelu3' in name)) or name == 'qrelu':
        #             # if name == 'qrelu':
        #                 module.set_T(4)
        #                 print(f'Setting {name} with T=4')
        #             else:
        #                 module.set_T(cfg.MODEL.T)
        #                 print(f'Setting {name} with T={cfg.MODEL.T}')
    return model
    # return model

