from torch import nn
from ssd.modeling.backbone.vgg import VGG
from ssd.modeling.backbone.basic import BasicModel
from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd import torch_utils
from ssd.modeling.backbone.resnet import resnet101, resnet152, resnet18, resnet34, resnet50, ExtendedResNet, wide_resnet50_2, wide_resnet101_2
from ssd.modeling.backbone.resnet_simplefied import resnet34_simplefied, resnet50_simplefied, resnet101_simplefied
from ssd.modeling.backbone.inception_v3 import inception_v3_backbone

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = SSDBoxHead(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")

    def forward(self, images, targets=None):
        features = self.backbone(images)
        #for fe in features:
        #    print("SHAPE: ", fe.shape)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    print(backbone_name)
    if backbone_name == "basic":
        model = BasicModel(cfg)
        return model
    if backbone_name == "vgg":
        model = VGG(cfg)
        if cfg.MODEL.BACKBONE.PRETRAINED:
            state_dict = load_state_dict_from_url(
                "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth")
            model.init_from_pretrain(state_dict)
        return model
    if backbone_name == "resnet34":
        resnet = resnet34(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet)
        return model
    if backbone_name == "resnet34":
        resnet = resnet34(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet, cfg)
        return model
    if backbone_name == "resnet50":
        resnet = resnet50(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet, cfg)
        return model
    if backbone_name == "resnet101":
        resnet = resnet101(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet)
        return model
    if backbone_name == "resnet152":
        resnet = resnet152(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet)
        return model
    if backbone_name == "wide_resnet50":
        resnet = wide_resnet50_2(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet)
        return model
    if backbone_name == "wide_resnet101":
        resnet = wide_resnet50_2(cfg.MODEL.BACKBONE.PRETRAINED)
        model = ExtendedResNet(resnet)
        return model
    if backbone_name == "inception_v3":
        model = inception_v3_backbone(cfg.MODEL.BACKBONE.PRETRAINED)
        return model
    if backbone_name == "resnet34_simplefied":
        model = resnet34_simplefied(cfg.MODEL.BACKBONE.PRETRAINED)
        return model
    if backbone_name == "resnet50_simplefied":
        model = resnet50_simplefied(cfg.MODEL.BACKBONE.PRETRAINED)
        return model
    if backbone_name == "resnet101_simplefied":
        model = resnet101_simplefied(cfg.MODEL.BACKBONE.PRETRAINED)
        return model
    if backbone_name == "MobileNetV2":
        model = mobilenet_v2(cfg.MODEL.BACKBONE.PRETRAINED, cfg.MODEL.NUM_CLASSES)
        return model

        
    