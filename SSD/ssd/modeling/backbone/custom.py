# from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
import torch.nn as nn
from .inception_v3 import inception_v3
from .resnet import resnet34
from torchvision.models.utils import load_state_dict_from_url

class Custom(nn.Module):
    print("HOLAA")
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    def __init__(self, cfg):
        super().__init__()
        self.resnet = resnet34(cfg.MODEL.BACKBONE.PRETRAINED)
        self.inception = inception_v3(cfg.MODEL.BACKBONE.PRETRAINED)
        
    def forward(self, x):
        features_1 = self.resnet(x)
        features_2 = self.inception(x)
        features = features_1 + features_2
        return features