# Version 1 – Road Damage Detection (Educational / Notebook-Based)

## Goal
A clean, well-documented, educational implementation of road damage detection
using three models: YOLOv8, MobileNetV2, and a Custom CNN.

## Folder Structure

```
v1/
├── notebooks/
│   ├── 01_dataset_preparation.ipynb     ← Download & explore dataset
│   ├── 02_model_implementation.ipynb    ← Build all 3 models
│   ├── 03_training_evaluation.ipynb     ← Train & evaluate with metrics
│   └── 04_final_demo.ipynb              ← End-to-end demo notebook
│
├── src/
│   ├── dataset.py       ← Dataset loader & preprocessing
│   ├── models.py        ← MobileNetV2 + Custom CNN definitions
│   ├── train.py         ← Training loop with metrics
│   ├── evaluate.py      ← Evaluation: Accuracy, Precision, Recall, F1
│   └── predict.py       ← Single-image inference
│
├── data/
│   └── (dataset goes here after running notebook 01)
│
├── models/
│   └── (saved .pt / .h5 weights go here after training)
│
├── outputs/
│   ├── plots/           ← Training curves, confusion matrix
│   └── results/         ← Prediction outputs
│
├── requirements.txt
└── README.md
```

## Dataset
- **RDD2022 (Road Damage Dataset 2022)**
- Source: https://github.com/sekilab/RoadDamageDetector
- Classes: D00 (Longitudinal Crack), D10 (Transverse Crack), D20 (Alligator Crack), D40 (Pothole)
- ~18,000 labeled images from Japan, India, Czech Republic

## Models
| Model        | Type       | Purpose                        |
|--------------|------------|--------------------------------|
| YOLOv8n      | Detection  | Bounding box + class detection |
| MobileNetV2  | Classifier | Lightweight image classification|
| Custom CNN   | Classifier | Domain-specific from scratch   |

## How to Run
1. Open notebooks in order: 01 → 02 → 03 → 04
2. Each notebook is self-contained with explanations
3. Run `pip install -r requirements.txt` first

## Requirements
See `requirements.txt`
