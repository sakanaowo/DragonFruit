import argparse
import os
from io import BytesIO

import requests
import torch
from PIL import Image
# uncomment this if you want to run this file on colab
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config
from src.model import load_model
from utils.transform import val_test_transform


def predict_image(image, device):
    config = load_config()
    model = load_model(device)

    class_name = config['data']['class_names']

    if image.startswith('http'):
        response = requests.get(image)
        if response.status_code != 200:
            raise Exception(f"Failed to download image from {image}: ", str(response.status_code))
        try:
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            raise Exception(f"Failed to load image from {image}: ", str(e))

    else:
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image {image} not found.")
        image = Image.open(image).convert('RGB')

    image = val_test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_name[predicted.item()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    result = predict_image(args.image, device)
    print(f"Predicted class: {result}")
    # python predict.py --image "url or path"
