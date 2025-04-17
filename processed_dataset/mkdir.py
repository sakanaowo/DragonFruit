import os
import json
import shutil
import random
from collections import defaultdict
from pathlib import Path

# Cấu hình
json_path = '../Dataset/cleaned_labels.json'
output_dir = ''
split_ratio = (0.7, 0.15, 0.15)  # train, val, test

# Tạo thư mục output
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Đọc labels
with open(json_path, 'r') as f:
    data = json.load(f)

# Gom ảnh theo class
label_to_paths = defaultdict(list)
for path, label in data.items():
    label_to_paths[label].append(path)

# Chia và copy ảnh
for label, paths in label_to_paths.items():
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    n_test = n - n_train - n_val

    splits = {
        'train': paths[:n_train],
        'val': paths[n_train:n_train + n_val],
        'test': paths[n_train + n_val:]
    }

    for split, split_paths in splits.items():
        class_dir = os.path.join(output_dir, split, label)
        os.makedirs(class_dir, exist_ok=True)

        for src_rel_path in split_paths:
            src_path = os.path.normpath(src_rel_path)
            filename = os.path.basename(src_path)
            dst_path = os.path.join(class_dir, filename)

            # Copy ảnh nếu tồn tại
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"[⚠️] Không tìm thấy ảnh: {src_path}")

print("✅ Dataset đã được chia và lưu trong thư mục `processed_dataset/`")
