import timm
import torch.nn as nn
from timm.models import register_model


@register_model
def cifar_resnet18(pretrained, num_classes, **kwargs):
    model = timm.create_model("resnet18", pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    return model, data_config


@register_model
def cifar_vit_small_patch16_224(pretrained, num_classes, **kwargs):
    model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    return model, data_config
