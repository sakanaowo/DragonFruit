import os

import torch
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
# uncomment to run on colab
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# //
from src.dataset import get_dataloaders
from utils.config import load_config
from utils.load_model import load_model


def evaluate(model, dataloader, device, class_name):
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
    report = classification_report(all_labels, all_preds, target_names=class_name)

    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:", report)

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
    classes_names = config['data']['class_names']
    evaluate(model, dataloader['test'], device, classes_names)
