import torch
import torch.optim as optim
import torch.nn as nn

from utils.config import load_config
from src.dataset import get_dataloaders
from src.model import build_model
from utils.helper import save_model

# uncomment this if you want to run this file on colab
import sys

sys.path.append('/kaggle/working/DragonFruit')


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += torch.sum(predicted == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    dataloaders = get_dataloaders(
        train_dir=config['data']['train_dir'],
        val_dir=config['data']['val_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )

    model = build_model(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    # loss&optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'],
                           weight_decay=config['train']['weight_decay'])

    # scheduler
    scheduler = None
    if config['scheduler']['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['step_size'],
                                              gamma=config['scheduler']['gamma'])

    # trainloop
    best_acc_val = 0.0
    for epoch in range(config['train']['epochs']):
        print(f"Epoch {epoch + 1}/{config['train']['epochs']}")

        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)

        print(f"Train Loss: {train_loss:.4f}|Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}|Val Acc: {val_acc * 100:.2f}%")

        if scheduler:
            scheduler.step()

        if val_acc > best_acc_val:
            best_acc_val = val_acc
            save_model(model, config['output']['model_save_path'])

    print("Training Finished!")


if __name__ == '__main__':
    main()
