from torchvision import models
import torch.nn as nn


def build_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, num_classes))
    return model
