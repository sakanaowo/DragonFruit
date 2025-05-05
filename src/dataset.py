import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.transform import train_transform, val_test_transform


def get_dataloaders(train_dir, val_dir, test_dir, batch_size, num_workers):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    targets = [sample[1] for sample in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return dataloaders


def get_val_loader(val_dir, batch_size, num_workers):
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_loader
