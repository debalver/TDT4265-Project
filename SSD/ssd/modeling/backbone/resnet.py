# Adapted code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

 
    
class MixedArchitecture(nn.Module):
    
    def __init__(self, cfg, resnet):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        norm_layer = nn.BatchNorm2d

        self.resnet = resnet
        
        # Attempt to copy resnet architecture to output a 5x5, 3x3 and 1x1 feature maps 
        # According to SSD paper, there is 5x5 256 channels and 3x3 256 and 1x1 256
       
        
        # The first block takes as input the output from layer4, a 10x10 2048 channels 
        # And should output a 5x5 output_channels[4]    
        self.block1 = nn.Sequential(    
            conv1x1(output_channels[2], 512),
            norm_layer(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False, dilation=1),
            norm_layer(256),
            nn.Conv2d(256, output_channels[4], kernel_size=1, stride=2, padding=1, bias=False, dilation=1),
            norm_layer(output_channels[4]),
        )
        self.relu = nn.ReLU(inplace=True)
        
        
        # Second mixed block, should output a 3x3 output_channels[5]    
        self.block2 = nn.Sequential(    
            conv1x1(output_channels[4], 512),
            norm_layer(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False, dilation=1),
            norm_layer(256),
            conv1x1(256, output_channels[5]),
            norm_layer(output_channels[5]),
        )
        # One more ReLU here 
        
        # Third mixed block, should output a 1x1 output_channels[6]    
        self.block3 = nn.Sequential(    
            conv1x1(output_channels[5], 512),
            norm_layer(512),
            nn.Conv2d(512, 256, kernel_size=(2, 3), stride=1, padding=0, bias=False, dilation=1),
            norm_layer(256),
            conv1x1(256, output_channels[6]),
            norm_layer(output_channels[6])
        )
        # One more ReLU here
        
        self.downsample1 = nn.Sequential(
            conv1x1(output_channels[2], output_channels[4], 2),
            norm_layer(output_channels[4])
        )
        self.downsample2 = nn.Sequential(
            conv1x1(output_channels[4], output_channels[5], 2),
            norm_layer(output_channels[5])
        )
        self.downsample3 = nn.Sequential(
            conv1x1(output_channels[5], output_channels[6], 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            norm_layer(output_channels[6])
        )
        
    
    def forward(self, x):
        """
        The forward function should output features with shape:
            resnet output shape(38x38 128, 19x19 256, 10x10 512, 1x1 512)
            shape(-1, output_channels[4], 5, 5),
            shape(-1, output_channels[5], 3, 3),
            shape(-1, output_channels[6], 1, 1)]
        """
    
        # For some reason .append(x) doesn't work with the mixed architecture, I have to use indexing 
        out_features = [None]*7
        
        # Compute the output from resnet backbone 
        resnet_output = self.resnet(x)
        
        for i, layer in enumerate(resnet_output):
            out_features[i] = layer
            
        # Compute the output for the mixed architecture
        # Take the Resnet ideas with residuals 
        
        x = resnet_output[2]
        identity = self.downsample1(x)
        x = self.block1(x) 
        x += identity
        x = self.relu(x)
        # The 5x5 output
        out_features[4] = x
        identity = self.downsample2(x)
        x = self.block2(x)
        x += identity
        x = self.relu(x)
        # The 3x3 output 
        out_features[5] = x
        identity = self.downsample3(x)
        x = self.block3(x)
        x += identity
        x = self.relu(x)
        # The 1x1 output 
        out_features[6] = x
        
        #print("The 7 output features")
        #for layer in out_features:
        #    print(layer.shape) 
        
        return out_features
        
    
    
class ResNet(nn.Module):

    def __init__(self, cfg, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # The extracted features for the box_head
        features = []

        # See note [TorchScript super()]
        # Preprocessing, no interesting features yet 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # 75x75
        x = self.layer2(x)
        # 38x38 512 
        features.append(x)
        x = self.layer3(x)
        # 19x19 1024 
        features.append(x)
        x = self.layer4(x)
        # 10x10 2048 
        features.append(x)
        x = self.avgpool(x)
        # 1x1 2048
        features.append(x)
        
        return features
        
        # The classifier provided by Resnet is not used, we use our own in box_head
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(cfg, arch, block, layers, progress, **kwargs):
    model = ResNet(cfg, block, layers, **kwargs)
    if cfg.MODEL.BACKBONE.PRETRAINED:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # Transfer learning ! 
        if cfg.MODEL.BACKBONE.TRANSFER_LEARNING:
            for param in model.conv1.parameters(): 
                param.requires_grad = False
            for param in model.bn1.parameters(): 
                param.requires_grad = False
            for param in model.relu.parameters(): 
                param.requires_grad = False
            for param in model.maxpool.parameters(): 
                param.requires_grad = False
            for param in model.layer1.parameters(): 
                param.requires_grad = False
            for param in model.layer2.parameters(): 
                param.requires_grad = False
            for param in model.layer3.parameters(): 
                param.requires_grad = False
            for param in model.layer4.parameters(): 
                param.requires_grad = True
            for param in model.avgpool.parameters(): 
                param.requires_grad = True
            for param in model.fc.parameters(): 
                param.requires_grad = False
        
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(cfg, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(cfg, 'resnet34', BasicBlock, [3, 4, 6, 3], progress,
                   **kwargs)


def resnet50(cfg, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(cfg, 'resnet50', Bottleneck, [3, 4, 6, 3], progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, transfer_learning=False ,progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, transfer_learning, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)