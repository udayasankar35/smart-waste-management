# YOLO Assets

This folder keeps the waste detection model assets and training data.

## Structure

- `weights/` - pretrained and custom model weights
- `training_data/waste/` - images, labels, and `data.yaml` for training
- `runs/` - optional training outputs if you keep local experiments here

Use `python train_waste_model.py --setup` to create the training data subfolders.
