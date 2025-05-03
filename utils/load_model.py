import torch
from torch import nn
from torchvision import models

from utils.config import load_config


def load_model(device):
    config = load_config()
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, config['model']['num_classes'])
    model.load_state_dict(torch.load(config['model']['model_path']))
    model.to(device)
    model.eval()
    return model
