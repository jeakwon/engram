import timm
import torch.nn as nn

def cifar_resnet18(pretrained, num_classes):
    model = timm.create_model("resnet18", pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def cifar_vit_base_patch16_224(pretrained, num_classes):
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def cifar_mixer_b16_224(pretrained, num_classes):
    model = timm.create_model("mixer_b16_224", pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model