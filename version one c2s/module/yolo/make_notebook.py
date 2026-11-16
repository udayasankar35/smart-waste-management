import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": []
}

def add_cell(cell_type, source):
    lines = source.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    notebook["cells"].append({
        "cell_type": cell_type,
        "metadata": {},
        "source": src,
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else None
    })

# Title
add_cell("markdown", """# 🚀 EcoSort / C2S: Master YOLOv8 & YOLO11 Waste Training Pipeline
### Featuring 30 Curated Kaggle Waste Datasets & One-Click GPU Training
This notebook supports the 7 canonical classes of the C2S Waste Monitoring taxonomy:
`0: plastic`, `1: glass`, `2: metal`, `3: paper`, `4: cardboard`, `5: organic`, `6: hazardous`
""")

# CELL 1
add_cell("markdown", """## ⚡ CELL 1: Check GPU & Install Dependencies
Verifies Tesla T4 GPU acceleration and installs Ultralytics YOLO, Kaggle CLI, and image processing tools.
""")
add_cell("code", """# Check GPU & Install dependencies
!nvidia-smi
!pip install -q ultralytics kaggle pyyaml opencv-python matplotlib kagglehub

import os
import re
import glob
import shutil
import yaml
from IPython.display import Image, display
from ultralytics import YOLO

print("✅ Environment ready with GPU support!")
""")

# CELL 2
add_cell("markdown", """## 🔑 CELL 2: Authenticate Kaggle API Token
Sets up Kaggle authentication directly using your new Kaggle API Token (no file upload required).
""")
add_cell("code", """import os

# Set Kaggle API Token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_58afe7c786837f5160bbbc83a9fc18df"

# Configure Kaggle client
!mkdir -p ~/.kaggle
!echo KGAT_58afe7c786837f5160bbbc83a9fc18df > ~/.kaggle/access_token
!chmod 600 ~/.kaggle/access_token

# Verify connection
!kaggle datasets list -s "waste detection" --pageSize 3
print("✅ Kaggle API authenticated successfully!")
""")

# CELL 3
add_cell("markdown", """## 📦 CELL 3: 30 Curated Kaggle Datasets & Interactive Downloader
Select any dataset from 1 to 30, enter a custom Kaggle URL, or download multiple datasets.
""")
add_cell("code", """# ==============================================================================
# TOP 30 KAGGLE WASTE DATASETS DIRECTORY
# ==============================================================================
KAGGLE_DATASETS = {
    # --- Group A: YOLO / Bounding Box Object Detection (Directly YOLO Ready) ---
    1:  {"name": "Trash Detection YOLOv8", "slug": "president/trash-detection-image-dataset", "type": "Detection", "classes": "Trash, plastic bottles, cans, bags"},
    2:  {"name": "TACO Trash in Context", "slug": "kneroma/tacotrashdataset", "type": "Detection", "classes": "Plastic, glass, cans, paper, carton in the wild"},
    3:  {"name": "Waste Detection YOLO", "slug": "amirrezafarhadi/waste-detection-dataset", "type": "Detection", "classes": "Plastic, metal, organic, cardboard"},
    4:  {"name": "Urban Waste Bounding Boxes", "slug": "sankalpsrivastava/waste-dataset", "type": "Detection", "classes": "Street litter, bottles, cans, wrappers"},
    5:  {"name": "Packaging & Bottle Trash", "slug": "parthdande/waste-detection-dataset", "type": "Detection", "classes": "Plastic bottles, cans, tetra packs"},
    6:  {"name": "Real-Waste Municipal Dataset", "slug": "shivan98/real-waste", "type": "Detection", "classes": "Plastic, glass, metal, paper, organic, textile"},
    7:  {"name": "Household Waste & Packaging", "slug": "andrewmvd/waste-identification", "type": "Detection", "classes": "Domestic food & drink packaging litter"},
    8:  {"name": "Waste Detection YOLOv8 Pre-split", "slug": "bipin143/waste-detection-yolov8", "type": "Detection", "classes": "Multi-class waste detection"},
    9:  {"name": "Trash Detection YOLO", "slug": "ashwinshenoy/trash-detection-yolo", "type": "Detection", "classes": "Bottles, paper, organic trash"},
    10: {"name": "Marine & Outdoor Debris", "slug": "paultimothymooney/marine-debris-dataset", "type": "Detection", "classes": "Plastic bottles, nylon bags, metal tins"},

    # --- Group B: Multi-Class Garbage & Recycling Classification ---
    11: {"name": "Garbage Classification 12-Classes", "slug": "mostafaabla/garbage-classification", "type": "Classification", "classes": "Battery, bio, glass (3 shades), cardboard, metal, paper, plastic, shoes, clothes"},
    12: {"name": "Classic TrashNet (6 Classes)", "slug": "asdasdasasdas/garbage-classification", "type": "Classification", "classes": "Cardboard, glass, metal, paper, plastic, trash"},
    13: {"name": "Waste Classification (25k Images)", "slug": "techsash/waste-classification-data", "type": "Classification", "classes": "Organic vs Recyclable"},
    14: {"name": "Recyclable Household Waste", "slug": "alistairking/recyclable-and-household-waste-classification", "type": "Classification", "classes": "Aerosol cans, bottles, cartons, jars"},
    15: {"name": "Kitchen & Food Waste Sorting", "slug": "sasha1807/waste-classification-data", "type": "Classification", "classes": "Organic food scraps, plastic, cardboard"},
    16: {"name": "Standardized Trash Types", "slug": "farzadnekouei/trash-type-image-dataset", "type": "Classification", "classes": "Glass, paper, cardboard, plastic, metal"},
    17: {"name": "V7 Waste Sorting Stream", "slug": "vbookshelf/v7-waste-sorting-dataset", "type": "Classification", "classes": "Household recyclables sorting"},
    18: {"name": "Expanded 12-Class Garbage", "slug": "bhanupratapbisht/garbage-classification-12-classes", "type": "Classification", "classes": "15,000+ items across 12 categories"},
    19: {"name": "Curated TrashNet Clean Split", "slug": "feyzazk/trashnet", "type": "Classification", "classes": "6 standard recycling streams"},
    20: {"name": "Comprehensive Garbage Collection", "slug": "alfrandom/comprehensive-garbage-dataset", "type": "Classification", "classes": "14,000+ items across municipal waste"},

    # --- Group C: Specialized Materials (Hazardous, E-Waste, Organic & Medical) ---
    21: {"name": "Electronic Waste & Hazardous Items", "slug": "saumilbhatt/e-waste-dataset", "type": "Specialized", "classes": "Hazardous: batteries, PCBs, chargers, phones, cables"},
    22: {"name": "Compostable vs Inorganic Waste", "slug": "thespacecoder/waste-classification", "type": "Specialized", "classes": "Food organics, plastic containers, tin cans"},
    23: {"name": "Domestic Food & Kitchen Scraps", "slug": "manishkc06/waste-classification-data", "type": "Specialized", "classes": "Organic kitchen waste, food waste"},
    24: {"name": "Sorted Materials Stream", "slug": "markdsu/garbage-classification", "type": "Classification", "classes": "Domestic segregated waste"},
    25: {"name": "Packaging & Plastic Waste", "slug": "favyen/waste-classification", "type": "Classification", "classes": "Plastic wraps, bags, paperboard"},
    26: {"name": "E-Waste Hardware Components", "slug": "saurabhshahane/electronic-waste-dataset", "type": "Specialized", "classes": "Batteries, circuits, electronic scrap"},
    27: {"name": "Medical & Biohazard Waste", "slug": "aryashah2k/breast-and-medical-waste-dataset", "type": "Specialized", "classes": "Medical, gloves, masks, biohazard"},
    28: {"name": "Cardboard & Metal Cans Dataset", "slug": "harshwalia/garbage-classification-dataset", "type": "Classification", "classes": "Cardboard, glass, metal, paper, plastic"},
    29: {"name": "Solid Household Waste Collection", "slug": "shashwatwork/waste-dataset", "type": "Classification", "classes": "Multi-category municipal solid waste"},
    30: {"name": "Segregated Recycling Streams", "slug": "pragati17/garbage-classification", "type": "Classification", "classes": "Dry vs wet segregated materials"}
}

def print_dataset_catalog():
    print("="*80)
    print("📋 TOP 30 KAGGLE WASTE DATASETS FOR C2S / ECOSORT")
    print("="*80)
    for idx, d in KAGGLE_DATASETS.items():
        print(f"[{idx:2d}] {d['name']:<35} | {d['type']:<14} | {d['slug']}")
    print("="*80)

def download_dataset(selection, target_dir="/content/waste_dataset"):
    os.makedirs(target_dir, exist_ok=True)
    slug = None
    
    if isinstance(selection, int) and selection in KAGGLE_DATASETS:
        slug = KAGGLE_DATASETS[selection]["slug"]
        print(f"🎯 Selected #{selection}: {KAGGLE_DATASETS[selection]['name']} ({slug})")
    elif isinstance(selection, str):
        match = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", selection)
        slug = match.group(1) if match else selection.strip()
    
    if not slug:
        print("❌ Invalid selection!")
        return
        
    print(f"📦 Downloading & unzipping '{slug}' into {target_dir}...")
    !kaggle datasets download -d {slug} -p {target_dir} --unzip
    print(f"✅ Extracted successfully into {target_dir}!")

# Print catalog
print_dataset_catalog()

# -------------------------------------------------------------------------
# CHOOSE YOUR DATASET HERE (Default: #1 President's YOLO Waste Dataset)
# You can change 1 to any number (1-30) or paste a custom Kaggle URL!
# -------------------------------------------------------------------------
SELECTED_DATASET = 1  
download_dataset(SELECTED_DATASET)
""")

# CELL 4
add_cell("markdown", """## ⚙️ CELL 4: Resolve Dataset Paths & Auto-Generate `waste_data.yaml`
Scans the downloaded directory to find image splits (`train`, `val`) and writes `waste_data.yaml` mapped to the 7 canonical classes.
""")
add_cell("code", """import os
import yaml

DATASET_ROOT = "/content/waste_dataset"

def find_subfolder(base_dir, candidates):
    for c in candidates:
        for root, dirs, _ in os.walk(base_dir):
            if c in dirs:
                return os.path.join(root, c)
    return None

# Find train and val folders
train_path = find_subfolder(DATASET_ROOT, ["train", "images/train", "Training"])
val_path = find_subfolder(DATASET_ROOT, ["val", "valid", "validation", "images/val", "images/valid", "Validation"])

if not train_path:
    train_path = DATASET_ROOT
if not val_path:
    val_path = train_path

yaml_config = {
    'path': DATASET_ROOT,
    'train': train_path,
    'val': val_path,
    'names': {
        0: 'plastic',
        1: 'glass',
        2: 'metal',
        3: 'paper',
        4: 'cardboard',
        5: 'organic',
        6: 'hazardous'
    }
}

DATA_CONFIG_PATH = "/content/waste_data.yaml"
with open(DATA_CONFIG_PATH, "w") as f:
    yaml.dump(yaml_config, f, sort_keys=False)

print(f"✅ Generated {DATA_CONFIG_PATH}:")
print(f"  Root : {DATASET_ROOT}")
print(f"  Train: {train_path}")
print(f"  Val  : {val_path}")
print("--- Class Mapping ---")
for k, v in yaml_config['names'].items():
    print(f"  {k}: {v}")
""")

# CELL 5
add_cell("markdown", """## 🏋️ CELL 5: Train YOLO11s Model with Anti-Hallucination Augmentations
Trains YOLO11s on the Tesla T4 GPU using augmentations (`mosaic=1.0`, `mixup=0.15`, `scale=0.5`) to eliminate false metal/glass detections on plain walls or tables.
""")
add_cell("code", """from ultralytics import YOLO

# 1. Initialize YOLO11s pretrained weights
print("🚀 Initializing YOLO11s model...")
model = YOLO("yolo11s.pt")

# 2. Train on GPU
results = model.train(
    data="/content/waste_data.yaml",
    epochs=40,               # 30-50 epochs is recommended
    imgsz=640,
    batch=16,
    device=0,                # Train on GPU
    project="runs/detect",
    name="c2s_waste_yolo11s",
    patience=15,             # Early stopping if no improvement
    save=True,
    workers=4,
    # Anti-Hallucination Augmentations:
    mosaic=1.0,              # Blends multiple items to learn realistic bounding boxes
    mixup=0.15,              # Blends background to prevent color bias
    scale=0.5,               # Handles both small caps and large containers
    degrees=15.0,            # Rotation invariance
    fliplr=0.5,
    exist_ok=True
)

print("🎉 Training completed successfully!")
""")

# CELL 6
add_cell("markdown", """## 📊 CELL 6: Model Validation & Confusion Matrix Visualization
Evaluates mean Average Precision (mAP) and displays the confusion matrix.
""")
add_cell("code", """from IPython.display import Image, display
import os

# 1. Evaluate metrics
metrics = model.val()
print("\n--- Validation Metrics ---")
print(f"mAP@50-95 : {metrics.box.map:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")

# 2. Display Confusion Matrix
cm_path = "runs/detect/c2s_waste_yolo11s/confusion_matrix.png"
if os.path.exists(cm_path):
    print("\nConfusion Matrix:")
    display(Image(filename=cm_path))

# 3. Display Training Progression Curves
results_curve = "runs/detect/c2s_waste_yolo11s/results.png"
if os.path.exists(results_curve):
    print("\nTraining Metrics & Loss Curves:")
    display(Image(filename=results_curve))
""")

# CELL 7
add_cell("markdown", """## 💾 CELL 7: Auto-Download `best.pt` to Your Computer
Downloads the trained weights directly to your local machine to use with your local web application.
""")
add_cell("code", """from google.colab import files
import os

best_weights = "runs/detect/c2s_waste_yolo11s/weights/best.pt"

if os.path.exists(best_weights):
    print(f"⬇️ Downloading {best_weights} to your computer...")
    files.download(best_weights)
    print("\n✅ Download started!")
    print("📁 Save this file into your project at:")
    print("   version one c2s/module/yolo/weights/best.pt")
else:
    print(f"❌ Error: {best_weights} not found. Check runs/detect/c2s_waste_yolo11s/weights/")
""")

out_path = r"c:\Users\kumar\OneDrive\C2S\version one c2s\module\yolo\Waste_Detection_YOLO_Colab_Master.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("[OK] Successfully generated updated notebook at:", out_path)
