#!/usr/bin/env python3
"""
YOLOv8 Waste Model Training Script
AI-Enabled Smart Waste Segregation & Monitoring System

Fine-tunes YOLOv8n on waste classification datasets:
  - TrashNet, TACO, WasteNet, or custom collected images
  - Exports optimized model for Jetson Nano (TensorRT / ONNX)

Usage:
    python train_waste_model.py --data ../waste-detection.v1i.yolov8/data.yaml --epochs 100
    python train_waste_model.py --data ../waste-detection.v1i.yolov8/data.yaml --epochs 50 --export tensorrt
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
    from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Default Dataset Configuration ──────────────────────────────────────────────

DEFAULT_DATA_YAML = """
# Plastic Waste Detection Dataset
# Single class: plastic

path: ../waste-detection.v1i.yolov8
train: train/images
val: valid/images
test: test/images

# Plastic only
nc: 1
names:
  0: plastic
"""

# ─── Augmentation Presets ───────────────────────────────────────────────────────

AUGMENTATION_CONFIG = {
    "hsv_h": 0.015,         # Hue augmentation
    "hsv_s": 0.7,           # Saturation augmentation
    "hsv_v": 0.4,           # Value augmentation
    "degrees": 10.0,        # Rotation ±10°
    "translate": 0.1,       # Translation ±10%
    "scale": 0.5,           # Scale ±50%
    "shear": 2.0,           # Shear ±2°
    "perspective": 0.0001,  # Perspective warp
    "flipud": 0.5,          # Flip up-down 50%
    "fliplr": 0.5,          # Flip left-right 50%
    "mosaic": 1.0,          # Mosaic augmentation
    "mixup": 0.1,           # Mixup augmentation
    "copy_paste": 0.1,      # Copy-paste augmentation
}


def create_dataset_structure(base_path: str):
    """Create dataset directory structure if it doesn't exist"""
    dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)
        logger.info(f"📁 Created: {base_path}/{d}")

    # Write default data.yaml
    data_yaml_path = os.path.join(base_path, "data.yaml")
    if not os.path.exists(data_yaml_path):
        with open(data_yaml_path, "w") as f:
            f.write(DEFAULT_DATA_YAML)
        logger.info(f"📄 Created: {data_yaml_path}")

    return data_yaml_path


def train_model(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    base_model: str = "yolo/weights/weights.pt",
    project: str = "yolo/runs/waste_detection",
    name: str = "train",
    device: str = "0",
    patience: int = 50,
    workers: int = 4,
    resume: bool = False,
):
    """
    Fine-tune YOLOv8n on waste dataset.

    Args:
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        base_model: Pretrained model to start from
        project: Output project directory
        name: Run name
        device: GPU device (0, 1, or 'cpu')
        patience: Early stopping patience
        workers: Number of data loading workers
        resume: Resume training from last checkpoint
    """
    print("\n" + "=" * 60)
    print("  🗑️  YOLOv8 WASTE MODEL TRAINING")
    print("=" * 60)
    print(f"  Base model : {base_model}")
    print(f"  Dataset    : {data_yaml}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  Image size : {img_size}")
    print(f"  Device     : {device}")
    print("=" * 60 + "\n")

    # Load base model
    model = YOLO(base_model)

    # Start training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        device=device,
        patience=patience,
        workers=workers,
        resume=resume,
        # Augmentation
        **AUGMENTATION_CONFIG,
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Saving
        save=True,
        save_period=10,
        # Logging
        verbose=True,
        plots=True,
    )

    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {project}/{name}/")

    return results


def export_model(model_path: str, export_format: str = "onnx", img_size: int = 640):
    """
    Export trained model for deployment.

    Supported formats: onnx, tensorrt, tflite, openvino
    """
    print(f"\n📦 Exporting model to {export_format}...")

    model = YOLO(model_path)

    exported = model.export(
        format=export_format,
        imgsz=img_size,
        half=True if export_format == "engine" else False,  # FP16 for TensorRT
        simplify=True,
        dynamic=False,
    )

    print(f"✅ Exported: {exported}")
    return exported


def validate_model(model_path: str, data_yaml: str, img_size: int = 640):
    """Run validation on the trained model"""
    print(f"\n📊 Validating model: {model_path}")

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, verbose=True)

    print(f"\n📊 Validation Results:")
    print(f"   mAP@50    : {results.box.map50:.4f}")
    print(f"   mAP@50-95 : {results.box.map:.4f}")
    print(f"   Precision  : {results.box.mp:.4f}")
    print(f"   Recall     : {results.box.mr:.4f}")

    return results


def copy_base_model():
    """Copy yolov8n.pt from drone root if available"""
    drone_model = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
    local_model = os.path.join("yolo", "weights", "yolov8n.pt")

    if os.path.exists(drone_model) and not os.path.exists(local_model):
        import shutil
        os.makedirs(os.path.dirname(local_model), exist_ok=True)
        shutil.copy2(drone_model, local_model)
        logger.info(f"📋 Copied base model from {drone_model}")
    elif not os.path.exists(local_model):
        logger.info("⬇️  Base model will be downloaded automatically by ultralytics")


# ─── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🗑️ YOLOv8 Waste Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset structure
  python train_waste_model.py --setup

  # Train model
    python train_waste_model.py --data yolo/training_data/waste/data.yaml --epochs 100

  # Export to TensorRT (Jetson)
    python train_waste_model.py --export tensorrt --model yolo/runs/waste_detection/train/weights/best.pt

  # Validate model
    python train_waste_model.py --validate --model yolo/runs/waste_detection/train/weights/best.pt --data yolo/training_data/waste/data.yaml
        """
    )

    parser.add_argument("--setup", action="store_true", help="Create dataset directory structure")
    parser.add_argument("--data", type=str, default="../waste-detection.v1i.yolov8/data.yaml", help="Data YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolo/weights/weights.pt", help="Base/trained model path")
    import torch
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="GPU device (0, 1, or 'cpu')")
    parser.add_argument("--export", type=str, choices=["onnx", "engine", "tflite", "openvino"],
                        help="Export format")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--resume", action="store_true", help="Resume training")

    args = parser.parse_args()

    # Setup mode
    if args.setup:
        create_dataset_structure("yolo/training_data/waste")
        print("\n✅ Dataset structure created!")
        print("   Place your images in yolo/training_data/waste/images/{train,val}/")
        print("   Place your labels in yolo/training_data/waste/labels/{train,val}/")
        return

    # Copy base model from drone project
    copy_base_model()

    # Export mode
    if args.export:
        export_model(args.model, args.export, args.img)
        return

    # Validate mode
    if args.validate:
        validate_model(args.model, args.data, args.img)
        return

    # Training mode (default)
    if not os.path.exists(args.data):
        print(f"❌ Dataset config not found: {args.data}")
        print("   Run with --setup first to create dataset structure")
        return

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        base_model=args.model,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
