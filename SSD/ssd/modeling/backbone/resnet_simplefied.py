# resnet implementation based on https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
            self.out_channels = [512, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]


        self.feature_extractor = list(backbone.children())[:8]

        # for i, module in enumerate(self.feature_extractor):
        #     print(i, module.__class__.__name__)
        # print("-")

        # delete the maxpool layer from resnet
        del self.feature_extractor[3]

        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # change the last conv from resnet to have stride 1
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)  
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class ResNetBackbone(nn.Module):
    def __init__(self, backbone=ResNet('resnet34')):
        super().__init__()

        self.feature_extractor = backbone

        self._build_additional_features(self.feature_extractor.out_channels)

        self._init_weights()

        self.first_foreward_pass = True

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            elif i < 4:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=(2, 3), bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.feature_extractor(x)

        features = [x]
        for l in self.additional_blocks:
            x = l(x)
            features.append(x)

        # print output feature map sizes once
        if (self.first_foreward_pass):
            self.first_foreward_pass = False
            print("Backbone eature output shapes:")
            for f in features:
                print(f.shape)
        return features


def resnet34_simplefied(pretrained=False, **kwargs):
    return ResNetBackbone(backbone=ResNet('resnet34', pretrained=pretrained))


def resnet50_simplefied(pretrained=False, **kwargs):
    return ResNetBackbone(backbone=ResNet('resnet50', pretrained=pretrained))


def resnet101_simplefied(pretrained=False, **kwargs):
    return ResNetBackbone(backbone=ResNet('resnet101', pretrained=pretrained))