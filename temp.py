# import os
# import json
# import shutil
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
#
# def move_single_image(item, source_root, dest_root, copy=True):
#     img_path, labels = item
#     if not labels:
#         return None
#
#     label = labels[0]
#     src_path = os.path.join(source_root, img_path)
#     dest_dir = os.path.join(dest_root, label)
#     os.makedirs(dest_dir, exist_ok=True)
#     dest_path = os.path.join(dest_dir, os.path.basename(img_path))
#
#     if not os.path.exists(src_path):
#         print(f"⚠️ Source not found: {src_path}")
#         return None
#
#     if copy:
#         shutil.copy2(src_path, dest_path)
#     else:
#         shutil.move(src_path, dest_path)
#
#     # Trả về path mới và label để lưu JSON
#     return dest_path, label
#
# def move_images_by_label_fast(json_path, source_root, dest_root, output_json_path, copy=True, num_workers=8):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#
#     os.makedirs(dest_root, exist_ok=True)
#
#     results = []
#
#     with ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         for item in data.items():
#             futures.append(executor.submit(move_single_image, item, source_root, dest_root, copy))
#
#         for future in tqdm(futures, desc="🚀 Moving images", ncols=100):
#             result = future.result()
#             if result:
#                 results.append(result)
#
#     # Viết ra JSON file mới
#     new_mapping = {path.replace("\\", "/"): label for path, label in results}
#     with open(output_json_path, 'w') as f:
#         json.dump(new_mapping, f, indent=4)
#
#     print(f"\n✅ Done! New label file saved to: {output_json_path}")
#
# # --- Cách gọi:
#
# move_images_by_label_fast(
#     json_path="Sorted_dataset/old_label.json",
#     source_root="",
#     dest_root="Sorted_dataset",
#     output_json_path="Sorted_dataset/sorted_dataset_labels.json",
#     copy=True,           # copy ảnh
#     num_workers=16       # tăng tốc
# )
import json
from collections import Counter

# with open("Sorted_dataset/sorted_dataset_labels.json", "r") as file:
#     data = json.load(file)
#
# dict = Counter()
# for item, label in data.items():
#     dict[label] += 1
# for item, count in dict.items():
#     print(item, count)

import torch
print(torch.__file__)
