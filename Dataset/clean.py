import json


def fix_cleaned_labels(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed_data = {}

    for path, labels in data.items():
        new_path = path[1:]
        fixed_data[new_path] = labels

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Đã lưu file đã sửa đường dẫn vào: {output_path}")


# Gọi hàm với đường dẫn cụ thể
fix_cleaned_labels("cleaned_labels.json", "../cleaned_labels_fixed.json")
