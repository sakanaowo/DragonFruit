# chuẩn hóa dữ liệu
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    # Định nghĩa phép biến đổi
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
    }

    # Load dataset
    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms["train"])
    val_ds = datasets.ImageFolder(root=f"{data_dir}/val", transform=data_transforms["val"])

    # Tạo DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
