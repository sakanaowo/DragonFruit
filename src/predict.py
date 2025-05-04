import argparse
import os

import torch
from PIL import Image

from utils.config import load_config
from src.model import load_model
from utils.transform import val_test_transform


def predict_image(image_path, device):
    config = load_config()
    model = load_model(device)

    class_name = config['data']['class_names']

    image = Image.open(image_path).convert('RGB')
    image = val_test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        predicted_class = class_name[pred.item()]

    return predicted_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image {args.image} not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_class = predict_image(args.image, device)
    print(f"Predicted class: {predicted_class}")
