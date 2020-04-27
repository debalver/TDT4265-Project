import torch
import torch.nn as nn
from torchvision.models import inception_v3


class Incepion(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        backbone = inception_v3(pretrained=pretrained)
        self.out_channels = [768, 512, 512, 256, 256, 128]


        self.feature_extractor = nn.Sequential(*list(backbone.children())[:13])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class IncepionBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.feature_extractor = backbone

        self._build_additional_features(self.feature_extractor.out_channels)

        self._init_weights()

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

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        #for f in detection_feed:
        #    print(f.shape)
        return detection_feed


def inception_v3_backbone(pretrained=True, **kwargs):
    return IncepionBackbone(Incepion(pretrained=pretrained))