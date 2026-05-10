"""
v1/src/predict.py
Single-image inference for MobileNetV2, CustomCNN, and YOLOv8.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

CLASS_NAMES = {0: "D00 – Longitudinal Crack", 1: "D10 – Transverse Crack",
               2: "D20 – Alligator Crack",    3: "D40 – Pothole",
               4: "Normal Road"}

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_classifier(model, image_path: str, device=None):
    """
    Run a single image through MobileNetV2 or CustomCNN.
    Returns: (class_name, confidence_pct)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()

    return CLASS_NAMES[idx], round(probs[idx].item() * 100, 2)


def predict_yolov8(model_path: str, image_path: str, conf: float = 0.25):
    """
    Run YOLOv8 detection on a single image.
    Returns list of dicts: [{label, confidence, bbox}, ...]
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "label":      r.names[int(box.cls)],
                "confidence": round(float(box.conf) * 100, 2),
                "bbox":       box.xyxy[0].tolist(),
            })
    return detections
