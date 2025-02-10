import timm
import torch.nn as nn

def cifar_resnet18(pretrained, num_classes):
    model = timm.create_model("resnet18", pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model

def cifar_vit(pretrained, num_classes):
    model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def cifar_mixer(pretrained, num_classes):
    model = timm.create_model("mixer_b16_224", pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model