import sys

import torch

from model.model import resnet18

sys.path.append('..')


def model_builder(cfg):
    if cfg.gpu is not None:
        print('=> use GPU: {}'.format(cfg.gpu))
    else:
        print('=> use CPU')
    print('=> building pre-trained model {}'.format(cfg.model.arch))
    model = resnet18(pretrained=cfg.model.pre_trained)

    device = torch.device(f'cuda:{cfg.gpu}' if cfg.gpu is not None else "cpu")
    model = model.to(device)
    model.eval()

    return model
