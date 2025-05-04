import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Thêm đường dẫn để import module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_dataloaders
from utils.config import load_config
from src.model import load_model


def evaluate(model, dataloader, device, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("Classification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    # Normalized Confusion Matrix
    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_norm.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Normalized Confusion Matrix")
    plt.show()

    return acc, report


if __name__ == '__main__':
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    dataloader = get_dataloaders(
        config['data']['train_dir'],
        config['data']['val_dir'],
        config['data']['test_dir'],
        config['data']['batch_size'],
        config['data']['num_workers'],
    )
    class_names = config['data']['class_names']
    evaluate(model, dataloader['test'], device, class_names)
