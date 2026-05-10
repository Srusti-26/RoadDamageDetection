"""
v1/src/dataset.py
Dataset loading and preprocessing for RDD2022 India (real zip).
Handles:
  - Pascal VOC XML parsing
  - Cropping bounding-box regions for classification
  - Normal (no-damage) image sampling
  - Train/val split
  - DataLoaders for MobileNetV2 and CustomCNN
"""

import os
import zipfile
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Label mapping ──────────────────────────────────────────────────────────────
# D44 (manhole/joint damage) → mapped to D40 (Pothole) as closest category
# D01, D11 → mapped to D00, D10 (sub-variants)
# D43, D50, rare classes → skipped (too few samples)
RAW_TO_CANONICAL = {
    "D00": "D00", "D01": "D00",          # Longitudinal Crack
    "D10": "D10", "D11": "D10",          # Transverse Crack
    "D20": "D20",                         # Alligator Crack
    "D40": "D40", "D44": "D40",          # Pothole / Manhole
}

CLASS_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "Normal": "Normal Road",
}
CLASS_TO_IDX = {"D00": 0, "D10": 1, "D20": 2, "D40": 3, "Normal": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split: str):
    """Return torchvision transforms for train / val / test splits."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── ZIP extraction & crop pipeline ───────────────────────────────────────────
def extract_and_prepare(
    zip_path: str,
    output_root: str,
    val_split: float = 0.15,
    normal_per_class: int = 400,
    seed: int = 42,
):
    """
    Extract RDD2022_India.zip, parse Pascal VOC XMLs, crop bounding-box
    regions per class, and build the classification folder structure:

        output_root/
          train/D00/*.jpg  train/D10/*.jpg  train/D20/*.jpg
          train/D40/*.jpg  train/Normal/*.jpg
          val/  (same structure)

    Also writes a YOLO-format dataset under output_root/yolo/ for YOLOv8.
    """
    random.seed(seed)
    output_root = Path(output_root)

    print("Opening zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_entries = {e.filename: e for e in zf.infolist()}

        # Collect (image_path_in_zip, xml_path_in_zip) pairs
        xml_entries = [e for e in zf.infolist()
                       if e.filename.startswith("India/train/annotations/xmls/") and e.filename.endswith(".xml")]

        # Parse all annotations
        crops = []       # (pil_crop, canonical_class)
        normal_imgs = [] # full images with no damage

        print(f"Parsing {len(xml_entries)} XML annotations...")
        for xml_entry in xml_entries:
            xml_data = zf.read(xml_entry.filename).decode("utf-8")
            root_el  = ET.fromstring(xml_data)
            filename = root_el.findtext("filename")
            img_key  = f"India/train/images/{filename}"

            if img_key not in all_entries:
                continue

            objects = root_el.findall("object")
            if not objects:
                normal_imgs.append(img_key)
                continue

            # Load image once per file
            img_bytes = zf.read(img_key)
            from io import BytesIO
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            w, h = img.size

            for obj in objects:
                raw_cls = obj.findtext("name", "").strip()
                canonical = RAW_TO_CANONICAL.get(raw_cls)
                if canonical is None:
                    continue  # skip rare/unknown classes

                bb = obj.find("bndbox")
                xmin = max(0, int(float(bb.findtext("xmin"))))
                ymin = max(0, int(float(bb.findtext("ymin"))))
                xmax = min(w, int(float(bb.findtext("xmax"))))
                ymax = min(h, int(float(bb.findtext("ymax"))))

                if xmax <= xmin or ymax <= ymin:
                    continue

                crop = img.crop((xmin, ymin, xmax, ymax))
                crops.append((crop, canonical))

        print(f"  Damage crops   : {len(crops)}")
        print(f"  Normal images  : {len(normal_imgs)}")

        # Sample normal images
        random.shuffle(normal_imgs)
        normal_sample = normal_imgs[:normal_per_class]
        normal_pil = []
        for img_key in normal_sample:
            from io import BytesIO
            img_bytes = zf.read(img_key)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            normal_pil.append((img, "Normal"))

        all_samples = crops + normal_pil
        random.shuffle(all_samples)

        # Train / val split
        n_val = int(len(all_samples) * val_split)
        val_samples   = all_samples[:n_val]
        train_samples = all_samples[n_val:]

        # Save to disk
        for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
            for cls_code in list(CLASS_TO_IDX.keys()):
                (output_root / split_name / cls_code).mkdir(parents=True, exist_ok=True)

        print("Saving crops to disk...")
        counters = {}
        for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
            for img, cls in split_data:
                idx = counters.get((split_name, cls), 0)
                out_path = output_root / split_name / cls / f"{cls}_{idx:05d}.jpg"
                img.save(out_path, quality=90)
                counters[(split_name, cls)] = idx + 1

    print("\n✓ Dataset prepared. Class distribution:")
    dataset_stats(str(output_root))


# ── Dataset class ──────────────────────────────────────────────────────────────
class RDDClassificationDataset(Dataset):
    """
    Expects folder structure:
        root/split/CLASS_CODE/*.jpg
    """
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root) / split
        self.transform = get_transforms(split)
        self.samples = []

        for class_code, idx in CLASS_TO_IDX.items():
            class_dir = self.root / class_code
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((str(img_path), idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found under {self.root}. "
                "Run notebook 01 (extract_and_prepare) first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


# ── DataLoader factory ─────────────────────────────────────────────────────────
def get_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 0):
    """Return train and val DataLoaders. num_workers=0 for Windows compatibility."""
    train_ds = RDDClassificationDataset(data_root, split="train")
    val_ds   = RDDClassificationDataset(data_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    return train_loader, val_loader


# ── Dataset statistics ─────────────────────────────────────────────────────────
def dataset_stats(data_root: str):
    """Print class distribution for train and val splits."""
    for split in ["train", "val"]:
        split_dir = Path(data_root) / split
        if not split_dir.exists():
            continue
        print(f"\n── {split.upper()} ──")
        total = 0
        for code in CLASS_TO_IDX:
            count = len(list((split_dir / code).glob("*.jpg"))) if (split_dir / code).exists() else 0
            print(f"  {code} ({CLASS_NAMES[code]}): {count} images")
            total += count
        print(f"  Total: {total}")
