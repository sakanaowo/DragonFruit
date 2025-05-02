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


def show_loss(train_loss, val_loss):
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')

    # Hiển thị điểm cực đại (max)
    max_train = max(train_loss)
    max_val = max(val_loss)
    idx_train = train_loss.index(max_train)
    idx_val = val_loss.index(max_val)

    plt.text(epochs[idx_train], max_train + 0.02, f"{max_train:.2f}", ha='center', fontsize=9, color='blue')
    plt.text(epochs[idx_val], max_val + 0.02, f"{max_val:.2f}", ha='center', fontsize=9, color='orange')

    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)

    plt.tight_layout()
    plt.show()



def show_acc(train_acc, val_acc):
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_acc, label='Train Acc', marker='o')
    plt.plot(epochs, val_acc, label='Val Acc', marker='o')

    # Hiển thị điểm cực đại (max)
    max_train = max(train_acc)
    max_val = max(val_acc)
    idx_train = train_acc.index(max_train)
    idx_val = val_acc.index(max_val)

    plt.text(epochs[idx_train], max_train + 0.5, f"{max_train:.2f}%", ha='center', fontsize=9, color='blue')
    plt.text(epochs[idx_val], max_val + 0.5, f"{max_val:.2f}%", ha='center', fontsize=9, color='orange')

    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)

    plt.tight_layout()
    plt.show()

