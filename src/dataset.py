import os
from torch.utils.data import Dataset
from PIL import Image


class DragonFruitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
