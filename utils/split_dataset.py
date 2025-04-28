# src/utils/split_dataset.py

import os
import random
import shutil

def split_dataset(sorted_dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must add up to 1.0"
    categories = os.listdir(sorted_dataset_dir)

    for category in categories:
        category_path = os.path.join(sorted_dataset_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = os.listdir(category_path)
        random.shuffle(images)

        train_split = int(train_ratio * len(images))
        val_split = int(val_ratio * len(images)) + train_split

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for split_name, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            split_dir = os.path.join(output_dir, split_name, category)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in split_images:
                src_path = os.path.join(category_path, img_name)
                dst_path = os.path.join(split_dir, img_name)
                shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    sorted_dataset_dir = "../Sorted_dataset"
    output_dir = "../data"
    split_dataset(sorted_dataset_dir, output_dir)
