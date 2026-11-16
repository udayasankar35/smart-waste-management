"""
Google Colab Training Script for YOLOv8/YOLO11 Multi-Class Waste Detection
==========================================================================
Recommended Datasets from Kaggle:
1. TACO Dataset: https://www.kaggle.com/datasets/kneroma/tacotrashdataset
2. Garbage Classification (12 classes / 6 classes): https://www.kaggle.com/datasets/mostafaabla/garbage-classification
3. Roboflow Universe Waste Detection: https://universe.roboflow.com/search?q=waste+detection

How to run in Google Colab (Free T4 GPU):
----------------------------------------
1. Open Google Colab (https://colab.research.google.com).
2. Change Runtime: Runtime -> Change runtime type -> T4 GPU -> Save.
3. Paste and run each cell below:

CELL 1: Install Ultralytics and dependencies
!pip install -q ultralytics kaggle

CELL 2: Download Dataset using Kaggle API or Roboflow
# Option A: Using Roboflow directly (Easiest & already formatted in YOLOv8)
!pip install -q roboflow
from roboflow import Roboflow
# Replace with your Roboflow workspace or free public waste project
# e.g.: https://universe.roboflow.com/material-identification/waste-detection-7classes
# rf = Roboflow(api_key="YOUR_KEY")
# project = rf.workspace("material-identification").project("waste-detection-7classes")
# version = project.version(1)
# dataset = version.download("yolov8")

CELL 3: Train YOLOv8 / YOLO11 Model
-----------------------------------
"""

from ultralytics import YOLO

def train():
    # 1. Initialize with pretrained weights (yolov8s.pt or yolo11s.pt)
    # Using 's' (small) gives much better accuracy than 'n' (nano) for subtle materials
    model = YOLO("yolo11s.pt")

    # 2. Train with data augmentations to prevent false metal/glass hallucinations
    results = model.train(
        data="waste_data.yaml",   # Path to data.yaml
        epochs=60,                # 50-70 epochs is optimal
        imgsz=640,
        batch=16,
        device=0,                 # GPU
        patience=15,              # Early stopping if no improvement
        save=True,
        workers=4,
        # Key augmentations to prevent whole-screen false positives:
        mosaic=1.0,               # Mix multiple items to learn bounding sizes
        mixup=0.15,               # Blends background to avoid color bias
        scale=0.5,                # Handles both small bottles and large furniture
        degrees=15.0,             # Rotations
        flipud=0.0,
        fliplr=0.5,
        name="waste_yolo11s_run"
    )

    # 3. Evaluate validation metrics
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50   :", metrics.box.map50)

    # 4. Export best weights
    # Trained weights will be saved at: runs/detect/waste_yolo11s_run/weights/best.pt
    print("Training finished! Download best.pt from: runs/detect/waste_yolo11s_run/weights/best.pt")

if __name__ == "__main__":
    print("Upload this script and waste_data.yaml to Google Colab with GPU enabled.")
