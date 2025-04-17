import json
import os

# Đường dẫn đến file labels.json gốc
original_json_path = 'labels.json'

# Đọc dữ liệu gốc
with open(original_json_path, 'r') as f:
    data = json.load(f)

# Tạo dictionary mới với đường dẫn tương đối
cleaned_data = {}

for abs_path, label in data.items():
    # Lấy phần đường dẫn bắt đầu từ 'dataset'
    rel_path = os.path.relpath(abs_path, start="D:\\Dragon Fruit project (CNN)\\")
    rel_path = rel_path.replace("\\", "/").replace("/D:/Dragon Fruit project (CNN)","")
    cleaned_data[rel_path] = label

# Ghi ra file mới
output_path = 'cleaned_labels.json'
with open(output_path, 'w') as f:
    json.dump(cleaned_data, f, indent=2)

print(f"✅ Done! Cleaned labels saved to: {output_path}")
