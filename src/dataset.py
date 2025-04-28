from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.transform import train_transform, val_test_transform


def get_dataloaders(train_dir, val_dir, test_dir, batch_size, num_workers):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return dataloaders
