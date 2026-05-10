"""
v1/src/evaluate.py
Evaluation utilities: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

CLASS_NAMES = ["D00 – Long. Crack", "D10 – Trans. Crack",
               "D20 – Alligator Crack", "D40 – Pothole", "Normal Road"]


def evaluate_model(model, val_loader, device=None, save_dir="outputs/plots"):
    """
    Run inference on val_loader and compute all metrics.

    Returns:
        metrics (dict): accuracy, precision, recall, f1
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
        "recall":    recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
        "f1":        f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
    }

    print("\n── Evaluation Results ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k.capitalize():12s}: {v:.2f}%")
    print()
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    _plot_confusion_matrix(all_labels, all_preds, save_dir)
    return metrics


def _plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out}")


def plot_training_history(history: dict, model_name: str, save_dir="outputs/plots"):
    """Plot loss and accuracy curves from training history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title(f"{model_name} – Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_title(f"{model_name} – Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / f"{model_name}_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {out}")
