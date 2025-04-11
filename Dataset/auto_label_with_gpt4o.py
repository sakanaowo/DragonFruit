import os
import openai
import base64
from tqdm import tqdm
import json
from PIL import Image
from io import BytesIO

openai.api_key = "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"  # 🔑 Thay bằng API key của bạn

# ✅ Danh sách nhãn cần GPT phân loại
label_list = [
    "Bacterial Diseases",
    "Bacterial Wilt (Fruits)",
    "Fungal Infections (Anthracnose or Stem Canker)",
    "Healthy Leaves",
    "Healthy Fruits",
    "Insect-Infected Fruits",
    "Mealybugs and Scale Insects (Fruits)",
    "Sunburn Damage"
]

# 📁 Folder gốc chứa các thư mục con: bad_fruit, good_leaf, ...
root_image_dir = "D:\Dragon Fruit project (CNN)\Dataset"  # <--- ⚠️ Sửa thành đường dẫn thật đến thư mục gốc

# 📄 Kết quả gán nhãn sẽ lưu ở đây
output_file = "gpt4o_labels.json"

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

results = {}

# Duyệt qua toàn bộ thư mục con và ảnh
all_image_paths = []
for subfolder in os.listdir(root_image_dir):
    subfolder_path = os.path.join(root_image_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(subfolder_path, file_name)
                all_image_paths.append(full_path)

for image_path in tqdm(all_image_paths, desc="Đang gán nhãn"):
    try:
        image_base64 = encode_image(image_path)

        response = openai.chat.completions.create(
            model="gpt-4o",   
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia nông nghiệp, hãy phân tích bệnh trên cây thanh long dựa vào hình ảnh."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Dựa trên hình ảnh này, hãy phân loại thành một trong các loại sau:\n\n" +
                                    "\n".join(f"- {label}" for label in label_list) +
                                    "\n\nChỉ trả lời bằng tên nhãn chính xác nhất."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0
        )

        label = response.choices[0].message.content.strip()
        results[image_path] = label
        print(f"[✅] {os.path.basename(image_path)}: {label}")

    except Exception as e:
        print(f"[❌ GPT lỗi] {image_path}: {e}")
        results[image_path] = "Lỗi"

# 💾 Lưu kết quả
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("🎉 Hoàn tất gán nhãn!")
