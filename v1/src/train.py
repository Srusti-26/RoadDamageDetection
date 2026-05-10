"""
v1/src/train.py
Training loop for MobileNetV2 and CustomCNN classifiers.
Saves best model weights and training history.
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    model_name: str,
    save_dir: str = "models",
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = None,
):
    """
    Train a classification model and save the best checkpoint.

    Returns:
        history (dict): train/val loss and accuracy per epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    save_path = Path(save_dir) / f"{model_name}_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        # ── Metrics ────────────────────────────────────────────────────────────
        t_loss = train_loss / train_total
        v_loss = val_loss   / val_total
        t_acc  = train_correct / train_total * 100
        v_acc  = val_correct   / val_total   * 100

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%")

        scheduler.step(v_acc)

        # ── Save best ──────────────────────────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved → {save_path}  (val_acc={v_acc:.2f}%)")

    # Save history
    history_path = Path(save_dir) / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    return history
