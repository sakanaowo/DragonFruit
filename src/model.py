import torch
from torchvision import models
import torch.nn as nn

from utils.config import load_config


def build_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, num_classes))
    return model


def load_model(device):
    config = load_config()
    num_classes=config['model']['num_classes']
    model = build_model(num_classes=num_classes, pretrained=False)

    model.load_state_dict(torch.load(config['model']['model_path']))
    model.to(device)
    model.eval()

    return model
