from .vgg import VGG16
from .darknet import DarkNet53
from .resnet import ResNet18, ResNet34, ResNet101
from .resnet_d import ResNetV1c
from .cspdarknet import CspDarkNet

backbone_dict={
    'vgg':VGG16,
    'darknet': DarkNet53,
    'resnet34': ResNet34,
    'resnet101':ResNet101,
    'resnet101d':ResNetV1c,
    'resnet18':ResNet18,
    'cspdarknet':CspDarkNet
}

def build_visual_encoder(__C):
    vis_enc=backbone_dict[__C.VIS_ENC](__C,multi_scale_outputs=True)
    return vis_enc