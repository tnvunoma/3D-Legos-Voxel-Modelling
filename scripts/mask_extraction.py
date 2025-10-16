import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms

def deeplabv3_segmentation(image_path):
    # Load pretrained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    input_image = Image.open(image_path).convert("RGB")

    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create mini-batch

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # For simplicity: consider non-background classes as foreground and COCO class 0 is background
    mask = (output_predictions != 0).astype(np.uint8) * 255

    # refine mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask