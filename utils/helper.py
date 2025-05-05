import torch
import os
import re
from utils.config import load_config

from matplotlib import pyplot as plt

epochs = list(range(1, 51))


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print('model saved')


# def load_model(model, path):
def load_logs(file_name):
    log_dir = load_config()['output']['log_dir']
    with open(log_dir + "/" + file_name, 'r') as f:
        return f.read()


def extract_metrics_from_log(log_text):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    # Tách từng dòng log
    lines = log_text.strip().split('\n')
    for line in lines:
        match = re.match(
            r"Epoch \[\d+\]: Train Loss: ([\d.]+), Train Accuracy: ([\d.]+)%, Val Loss: ([\d.]+), Val Accuracy: ([\d.]+)%",
            line
        )
        if match:
            train_loss.append(float(match.group(1)))
            train_acc.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            val_acc.append(float(match.group(4)))

    return [train_loss, val_loss, train_acc, val_acc]
