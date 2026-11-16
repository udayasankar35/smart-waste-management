import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import DEFAULT_CONFIG


def test_custom_model_path_is_project_relative_and_absolute():
    expected_prefix = str(MODULE_DIR)
    assert DEFAULT_CONFIG.detection.custom_model_path.startswith(expected_prefix)
    assert os.path.exists(DEFAULT_CONFIG.detection.custom_model_path)


def test_camera_resolution_is_consistent():
    assert DEFAULT_CONFIG.camera.width == 1280
    assert DEFAULT_CONFIG.camera.height == 720
