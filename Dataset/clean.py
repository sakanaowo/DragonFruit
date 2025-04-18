import json

with open("cleaned_labels.json", 'r', encoding="utf-8") as f:
    data = json.load(f)
i = 1
processed_data = {}
for path, label in data.items():
    # new_path = path.replace("..", "")
    processed_data[path] = [label]
    print(i, path)
    i += 1
with open("cleaned_labels.json", 'w', encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)
