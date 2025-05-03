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

    # Tìm tất cả các dòng chứa thông số
    lines = log_text.strip().split('\n')
    for line in lines:
        match = re.match(r"Train Loss: ([\d.]+)\|Train Acc: ([\d.]+)%", line)
        if match:
            train_loss.append(float(match.group(1)))
            train_acc.append(float(match.group(2)))
        else:
            match = re.match(r"Val Loss: ([\d.]+)\|Val Acc: ([\d.]+)%", line)
            if match:
                val_loss.append(float(match.group(1)))
                val_acc.append(float(match.group(2)))

    return [train_loss, val_loss, train_acc, val_acc]

