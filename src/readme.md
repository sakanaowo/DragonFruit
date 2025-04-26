```
DragonFruit/
│
├── data/                     # Dataset ảnh bệnh (train/val/test)
│   ├── train/
│   │   ├── healthy/
│   │   ├── disease_1/
│   │   ├── disease_2/
│   │   └── ...
│   ├── val/
│   └── test/
│
├── models/                   # Lưu model đã huấn luyện
│   └── resnet50_finetuned.pth
│
├── note/                # Notebook dùng để thử nghiệm nhanh
│   └── train_eval.ipynb
│
├── outputs/                  # Logs, biểu đồ, hình ảnh kết quả
│
├── src/                      # Code chính
│   ├── dataset.py            # Tiền xử lý và tạo DataLoader
│   ├── model.py              # Load và custom ResNet-50
│   ├── train.py              # Vòng lặp huấn luyện chính
│   ├── evaluate.py           # Đánh giá model (độ chính xác, confusion matrix,...)
│   └── predict.py            # Dự đoán một ảnh đầu vào mới
│
├── utils/                    # Các hàm tiện ích (log, plot, seed, metrics,...)
│   └── helper.py
│
├── requirements.txt          # Thư viện cần cài
├── config.yaml               # Cấu hình: learning rate, batch size,...
└── README.md                 # Mô tả project, hướng dẫn chạy
```