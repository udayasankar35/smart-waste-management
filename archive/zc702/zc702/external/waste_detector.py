#!/usr/bin/env python3
"""
YOLOv8 Plastic Detection Module
AI-Enabled Smart Waste Segregation & Monitoring System

Uses the custom-trained YOLOv8 plastic detection model:

    yolo/weights/weights.pt

Trained class:
    plastic (bottle)

Usage:
    # Detect plastic in a single image
    python detect_plastic.py --image path/to/image.jpg

    # Detect plastic in all images in a folder
    python detect_plastic.py --folder path/to/folder
"""

import cv2
import numpy as np
import time
import logging
import sys
import os

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "ultralytics",
        "-q"
    ])

    from ultralytics import YOLO


from config import (
    WASTE_CLASS_MAP,
    WASTE_CLASS_NAMES,
    DetectionConfig,
    DEFAULT_CONFIG,
)


logger = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class WasteDetection:
    """Single waste object detection result"""

    class_id: int
    class_name: str
    confidence: float

    bbox: Tuple[float, float, float, float]

    centroid: Tuple[float, float]

    area: float

    timestamp: float = field(default_factory=time.time)
    detected_object: str = ""

    def get_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    def get_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detected_object": self.detected_object or self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(x, 1) for x in self.bbox],
            "centroid": [round(x, 1) for x in self.centroid],
            "area": round(self.area, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class WasteFrame:
    """All waste detections in a single frame"""

    frame_id: int
    timestamp: float
    image_shape: Tuple[int, int, int]

    detections: List[WasteDetection] = field(default_factory=list)

    inference_time_ms: float = 0.0

    def get_by_class(self, class_name: str) -> List[WasteDetection]:
        return [
            d for d in self.detections
            if d.class_name == class_name
        ]

    def get_summary(self) -> Dict[str, int]:
        summary = {}

        for detection in self.detections:
            summary[detection.class_name] = (
                summary.get(detection.class_name, 0) + 1
            )

        return summary

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "num_detections": len(self.detections),
            "inference_time_ms": round(
                self.inference_time_ms,
                2
            ),
            "summary": self.get_summary(),
            "detections": [
                d.to_dict()
                for d in self.detections
            ],
        }


# ─── Waste Detector ───────────────────────────────────────────────────────────

class WasteDetector:
    """
    YOLO11s-based waste detector.

    Loads:
        yolo/weights/waste_yolo11s.pt

    Falls back to the configured generic YOLO model if the
    custom model is unavailable.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # New YOLO11s class → existing application category
    #
    # The trained model detects nine specific classes.
    # The rest of the project historically uses broader categories.
    # ──────────────────────────────────────────────────────────────────────────

    CUSTOM_MATERIAL_MAPPING: Dict[str, str] = {
        # Exact trained classes from best.pt (multi-material model)
        "plastic_bottle": "plastic",
        "glass_bottle": "glass",
        "metal_can": "metal",
        "paper_cardboard": "paper",
        "organic_waste": "organic",

        # New material7-v1 classes (best_v2.pt). They reuse categories the
        # advisor / database / colours already know, so nothing downstream breaks.
        "textile": "dry_waste",
        "e_waste": "hazardous",
        "e waste": "hazardous",
        "ewaste": "hazardous",
        "paper cardboard": "paper",
        "organic waste": "organic",

        # Common items and aliases
        "plastic": "plastic",
        "plastic bottle": "plastic",
        "bottle": "plastic",
        "milk jug": "plastic",
        "glass": "glass",
        "metal": "metal",
        "can": "metal",
        "tin": "metal",
        "paper": "paper",
        "cardboard": "cardboard",
        "organic": "organic",
    }

    # Class names that only exist in the new 7-class model (best_v2.pt).
    # If the loaded weights contain any of them, the detector switches to the
    # "material7-v1" behaviour below. The old best.pt keeps the old behaviour.
    V2_MARKER_CLASSES = ("textile", "e_waste")

    # Label shown / sent to the advisor for the NEW classes. These strings are
    # the same ones COCO_WASTE_MAP already produces, so the advisor knows them.
    V2_OBJECT_LABELS: Dict[str, str] = {
        "plastic": "Plastic",                      # was hard-coded "Plastic Bottle"
        "dry_waste": "Clothing / Fabric Textile",  # textile
        "hazardous": "Electronic Waste (E-Waste)",  # e_waste
    }

    # Old model: plastic got +0.20 (+0.05 more if custom) in the overlap
    # ranking. The new model is trained on balanced material classes, so the
    # bias is switched off for it (set both to 0.0). Raise only if the new
    # model under-detects plastic in your own tests.
    LEGACY_PLASTIC_BONUS = 0.20
    LEGACY_CUSTOM_PLASTIC_BONUS = 0.05

    # ──────────────────────────────────────────────────────────────────────────
    # COCO 80-Class → (Waste Category, Specific Object Label) Mapping
    # ──────────────────────────────────────────────────────────────────────────
    COCO_WASTE_MAP: Dict[str, Tuple[str, str]] = {
        # Cutlery / Tableware & Food Containers
        "spoon": ("dry_waste", "Spoon / Cutlery"),
        "fork": ("dry_waste", "Fork / Cutlery"),
        "knife": ("dry_waste", "Knife / Cutlery"),
        "bowl": ("plastic", "Plastic / Food Container / Bowl"),
        "cup": ("plastic", "Plastic / Paper Cup"),
        # Bottles & Glassware
        "bottle": ("plastic", "Plastic Bottle / Container"),
        "wine glass": ("glass", "Glass Bottle / Stemware"),
        "vase": ("plastic", "Plastic / Ceramic Container or Jar"),
        # Organic / Wet Food Waste
        "banana": ("wet_waste", "Banana / Fruit Peel (Organic Waste)"),
        "apple": ("wet_waste", "Apple / Fruit Food Waste (Organic Waste)"),
        "sandwich": ("wet_waste", "Sandwich / Food Waste (Organic Waste)"),
        "orange": ("wet_waste", "Orange / Citrus Peel (Organic Waste)"),
        "broccoli": ("wet_waste", "Vegetable Scrap (Organic Waste)"),
        "carrot": ("wet_waste", "Carrot / Vegetable Waste (Organic Waste)"),
        "hot dog": ("wet_waste", "Food Leftover (Organic Waste)"),
        "pizza": ("wet_waste", "Pizza / Food Waste (Organic Waste)"),
        "donut": ("wet_waste", "Food Scrap / Bakery (Organic Waste)"),
        "cake": ("wet_waste", "Cake / Food Waste (Organic Waste)"),
        # Textiles & Fabric / Dry Waste
        "tie": ("dry_waste", "Clothing / Fabric Textile"),
        "umbrella": ("dry_waste", "Clothing / Fabric Textile"),
        "backpack": ("dry_waste", "Textile Bag / Backpack"),
        "handbag": ("dry_waste", "Handbag / Textile Bag"),
        # Paper & Cardboard (Books, Notebooks)
        "book": ("paper", "Book / Paper Material"),
        # Metal Tools & Containers
        "scissors": ("metal", "Scissors / Metal Tool"),
        # Small Electronics & Hazardous / E-Waste
        "cell phone": ("hazardous", "Mobile Phone / Electronic Waste (E-Waste)"),
        "laptop": ("hazardous", "Laptop / Electronic Waste (E-Waste)"),
        "mouse": ("hazardous", "Computer Mouse / Electronic Waste"),
        "remote": ("hazardous", "Remote Controller / Electronic Waste"),
        "keyboard": ("hazardous", "Computer Keyboard / Electronic Waste"),
        # Personal Care, Plastic Trays & Miscellaneous
        "toothbrush": ("plastic", "Plastic Toothbrush"),
        "frisbee": ("plastic", "Plastic Tray / Disc / Container"),
        # Molded Rigid Plastics & Bulky Waste (Chairs, Stools, Luggage)
        "chair": ("plastic", "Molded Plastic Chair"),
        "bench": ("plastic", "Molded Plastic Bench"),
        "suitcase": ("plastic", "Molded Plastic Suitcase / Luggage"),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Display colors
    # ──────────────────────────────────────────────────────────────────────────
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "plastic": (0, 165, 255),
        "dry_waste": (255, 170, 0),
        "wet_waste": (50, 220, 80),
        "metal": (200, 200, 210),
        "glass": (0, 230, 180),
        "paper": (240, 220, 140),
        "cardboard": (180, 140, 90),
        "organic": (50, 220, 80),
        "hazardous": (40, 40, 255),
        "unknown": (200, 200, 200),
    }

    # Neon colors used by the HUD
    NEON_COLORS: Dict[str, Tuple[int, int, int]] = {
        "plastic": (0, 200, 255),
        "dry_waste": (255, 180, 20),
        "wet_waste": (60, 240, 100),
        "metal": (210, 210, 230),
        "glass": (20, 240, 200),
        "paper": (250, 230, 150),
        "cardboard": (200, 160, 100),
        "organic": (60, 240, 100),
        "hazardous": (50, 50, 255),
        "unknown": (220, 220, 220),
    }

    # ─── Constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        config: DetectionConfig = None
    ):
        self.config = (
            config
            or DEFAULT_CONFIG.detection
        )

        self.model: Optional[YOLO] = None
        self.general_model: Optional[YOLO] = None
        self.is_custom_model = False

        # Filled in by load_model(): which class set the custom weights use.
        self.custom_class_names: List[str] = []
        self.taxonomy = "legacy"          # "legacy" (old best.pt) or "material7-v1"
        self._plastic_bonus = self.LEGACY_PLASTIC_BONUS
        self._custom_plastic_bonus = self.LEGACY_CUSTOM_PLASTIC_BONUS

        self.frame_count = 0
        self.total_inference_time = 0.0
        self._fps_times: List[float] = []
        self._no_det_frames = 0

        logger.info(
            f"WasteDetector initialised — "
            f"model path: {self.config.model_path}"
        )

    # ─── Model Loading ────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        """
        Load models:
        1. Custom trained model (yolo/weights/weights.pt)
        2. General YOLOv8n model (yolov8n.pt) for multi-class detection (cutlery, textiles, organic, etc.)
        """
        try:
            # 1. Load custom trained model
            if os.path.exists(self.config.custom_model_path):
                self.model = YOLO(self.config.custom_model_path)
                self.is_custom_model = True
                self._detect_taxonomy()
                logger.info(
                    "CUSTOM YOLO WASTE MODEL LOADED: "
                    f"{self.config.custom_model_path} "
                    f"[taxonomy={self.taxonomy}, classes={self.custom_class_names}]"
                )
            elif os.path.exists(self.config.model_path):
                self.model = YOLO(self.config.model_path)
                self.is_custom_model = False
                logger.warning(
                    f"Using configured model: {self.config.model_path}"
                )
            else:
                self.model = YOLO("yolov8n.pt")
                self.is_custom_model = False

            # 2. Load general YOLOv8n model for multi-class detection
            general_candidates = [
                os.path.join(os.path.dirname(__file__), "yolov8n.pt"),
                "yolov8n.pt",
            ]
            self.general_model = None
            for g_path in general_candidates:
                if os.path.exists(g_path):
                    try:
                        self.general_model = YOLO(g_path)
                        logger.info(f"GENERAL YOLOv8n MODEL LOADED: {g_path}")
                        break
                    except Exception as ge:
                        logger.warning(f"Could not load general model at {g_path}: {ge}")

            if self.general_model is None and not self.is_custom_model:
                self.general_model = self.model

            # Warm-up
            self._warm_up()
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def _detect_taxonomy(self) -> None:
        """Read the class names stored inside the custom weights."""
        names = getattr(self.model, "names", None) or {}
        values = names.values() if isinstance(names, dict) else names
        self.custom_class_names = [
            str(n).strip().lower().replace(" ", "_") for n in values
        ]
        if any(m in self.custom_class_names for m in self.V2_MARKER_CLASSES):
            self.taxonomy = "material7-v1"
            self._plastic_bonus = 0.0
            self._custom_plastic_bonus = 0.0
        else:
            self.taxonomy = "legacy"
            self._plastic_bonus = self.LEGACY_PLASTIC_BONUS
            self._custom_plastic_bonus = self.LEGACY_CUSTOM_PLASTIC_BONUS

    # ─── Warm-up ──────────────────────────────────────────────────────────────

    def _warm_up(self):
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            warmup_kwargs = {
                "source": dummy,
                "conf": self.config.confidence_threshold,
                "imgsz": self.config.input_size,
                "device": self.config.device,
                "verbose": False,
            }
            if self.config.device != "cpu" and self.config.half_precision:
                warmup_kwargs["half"] = True

            if self.model is not None:
                self.model.predict(**warmup_kwargs)
            if self.general_model is not None and self.general_model != self.model:
                self.general_model.predict(**warmup_kwargs)

            logger.info("YOLO model warm-up complete")

        except Exception as e:
            logger.warning(f"Warm-up failed ({e}) — switching to CPU inference")
            self.config.device = "cpu"
            self.config.half_precision = False

    # ─── Class Name Normalisation ─────────────────────────────────────────────

    @staticmethod
    def normalize_material_name(raw_name: str) -> str:
        """Map any detected class name to application waste category."""
        if raw_name is None:
            return "unknown"

        normalized = (
            str(raw_name)
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )
        normalized = " ".join(normalized.split())

        if not normalized:
            return "unknown"

        # Exact trained-class mapping
        if normalized in WasteDetector.CUSTOM_MATERIAL_MAPPING:
            return WasteDetector.CUSTOM_MATERIAL_MAPPING[normalized]

        # COCO waste mapping
        if normalized in WasteDetector.COCO_WASTE_MAP:
            return WasteDetector.COCO_WASTE_MAP[normalized][0]

        # Keyword matching
        # Glass (check before plastic so 'glass bottle' -> glass)
        if any(token in normalized for token in [
            "glass", "jar", "wine glass"
        ]):
            return "glass"

        # Metal (check before plastic so 'tin can' -> metal)
        if any(token in normalized for token in [
            "metal", "can", "tin", "aluminium", "aluminum", "foil", "steel",
            "scissors", "blister pack", "pop tab", "aerosol"
        ]):
            return "metal"

        # Plastic (polymers, containers, packaging, films, tubs)
        if any(token in normalized for token in [
            "plastic", "bottle", "bag", "wrapper", "jug", "cup", "container",
            "polythene", "polypropylene", "tupperware", "styrofoam", "foam",
            "crisp", "tub", "squeezable", "film", "gloove", "glove", "straw",
            "rings", "vase"
        ]):
            return "plastic"

        # Paper / Cardboard
        if any(token in normalized for token in [
            "paper", "cardboard", "carton", "box", "newspaper", "book", "magazine",
            "tissue", "toilet tube"
        ]):
            return "cardboard" if any(k in normalized for k in ["cardboard", "carton", "box"]) else "paper"

        # Wet / Organic waste
        if any(token in normalized for token in [
            "apple", "banana", "orange", "fruit", "vegetable", "food", "scrap", "organic",
            "broccoli", "carrot", "sandwich", "pizza", "donut", "cake"
        ]):
            return "organic"

        # Textiles / Cutlery / Dry waste
        if any(token in normalized for token in [
            "cloth", "clothes", "fabric", "textile", "shirt", "towel", "tie", "umbrella",
            "backpack", "spoon", "fork", "knife", "cutlery", "rope", "string", "shoe"
        ]):
            return "dry_waste"

        # Hazardous / E-waste
        if any(token in normalized for token in [
            "battery", "e waste", "ewaste", "phone", "cell phone", "laptop", "remote",
            "electronic", "appliance"
        ]):
            return "hazardous"

        return "unknown"

    def _custom_object_label(self, cat: str) -> str:
        """Display label for a detection from the custom model."""
        if self.taxonomy == "material7-v1" and cat in self.V2_OBJECT_LABELS:
            return self.V2_OBJECT_LABELS[cat]
        # legacy behaviour (old best.pt) is unchanged
        return "Plastic Bottle" if cat == "plastic" else cat.replace("_", " ").title()

    @staticmethod
    def _compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(area1 + area2 - inter_area)
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _salient_material_fallback(self, frame: np.ndarray) -> Optional[WasteDetection]:
        """
        Visual & material feature fallback for waste objects with no canonical bounding box
        (e.g., plastic sheets, folders, film, unclassified plastic containers, or packaging).
        Inspects the primary foreground object's color, specular reflections, and polymer features.
        """
        try:
            h, w = frame.shape[:2]
            sw, sh = 320, 240
            small = cv2.resize(frame, (sw, sh))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(blurred, 25, 110)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_box = None
            max_area = 0
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if (sw * sh * 0.04) < area < (sw * sh * 0.96):
                    if area > max_area:
                        max_area = area
                        best_box = (x, y, x + cw, y + ch)

            if best_box is not None:
                scale_x = w / float(sw)
                scale_y = h / float(sh)
                x1 = int(best_box[0] * scale_x)
                y1 = int(best_box[1] * scale_y)
                x2 = int(best_box[2] * scale_x)
                y2 = int(best_box[3] * scale_y)
            else:
                x1 = int(w * 0.12)
                y1 = int(h * 0.12)
                x2 = int(w * 0.88)
                y2 = int(h * 0.88)

            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                return None

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_chan = hsv[:, :, 0]
            s_chan = hsv[:, :, 1]
            v_chan = hsv[:, :, 2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            mean_s = float(np.mean(s_chan))
            mean_v = float(np.mean(v_chan))

            # Specular gloss (typical for plastic polymers, folders, acrylic, smooth jugs)
            specular_mask = (v_chan > 215) & (s_chan < 55)
            specular_pct = float(np.mean(specular_mask))

            # Transparent / synthetic blue or vibrant plastic (folders, sleeves, files)
            blue_plastic_mask = (h_chan >= 85) & (h_chan <= 135) & (s_chan >= 25) & (v_chan >= 60)
            blue_plastic_pct = float(np.mean(blue_plastic_mask))

            # Cardboard brown check
            brown_mask = (h_chan >= 10) & (h_chan <= 28) & (s_chan >= 35) & (s_chan <= 170) & (v_chan >= 45) & (v_chan <= 210)
            brown_pct = float(np.mean(brown_mask))

            # Metallic sheen check
            laplacian_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            is_metallic = (mean_s < 40 and 60 < mean_v < 235 and laplacian_var > 140 and brown_pct < 0.15)

            # Organic food color check
            green_mask = (h_chan >= 35) & (h_chan <= 85) & (s_chan >= 60)
            orange_mask = ((h_chan <= 25) | (h_chan >= 165)) & (s_chan >= 90) & (v_chan >= 70)
            organic_pct = float(np.mean(green_mask | orange_mask))

            crop_h, crop_w = crop.shape[:2]
            aspect_ratio = float(crop_h) / max(1.0, float(crop_w))

            if brown_pct > 0.45:
                class_name = "cardboard"
                detected_object = "Cardboard / Paper Packaging"
                conf = 0.62
            elif is_metallic:
                class_name = "metal"
                detected_object = "Metal Can / Foil / Scrap"
                conf = 0.60
            elif organic_pct > 0.40:
                class_name = "organic"
                detected_object = "Organic / Food Scrap Waste"
                conf = 0.62
            else:
                class_name = "plastic"
                if blue_plastic_pct > 0.18 or (aspect_ratio < 1.1 and specular_pct > 0.015):
                    detected_object = "Plastic Folder / Sheet / Packaging"
                elif aspect_ratio > 1.25 or (mean_v > 160 and mean_s < 60):
                    detected_object = "Plastic Container / Jug / Canister"
                else:
                    detected_object = "Plastic Recyclable Item"
                conf = 0.65

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = float((x2 - x1) * (y2 - y1))

            return WasteDetection(
                class_id=0,
                class_name=class_name,
                confidence=conf,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                centroid=(float(cx), float(cy)),
                area=area,
                timestamp=time.time(),
                detected_object=detected_object,
            )
        except Exception as e:
            logger.debug(f"Salient material fallback failed: {e}")
            return None

    # ─── Inference ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> WasteFrame:
        timestamp = time.time()
        waste_frame = WasteFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            image_shape=frame.shape,
        )

        if self.model is None and self.general_model is None:
            logger.warning("Models not loaded — returning empty frame")
            return waste_frame

        try:
            start_time = time.time()
            candidates = []
            frame_h, frame_w = frame.shape[:2]
            frame_area = float(frame_h * frame_w)

            # 1. Run general YOLOv8n model (80 COCO classes: spoon, fork, cup, fruit, clothes, chair, etc.)
            if self.general_model is not None:
                coco_conf_thresh = max(0.20, self.config.confidence_threshold - 0.08)
                g_kwargs = {
                    "source": frame,
                    "conf": coco_conf_thresh,
                    "iou": self.config.iou_threshold,
                    "imgsz": self.config.input_size,
                    "device": self.config.device,
                    "verbose": False,
                }
                if self.config.device != "cpu" and self.config.half_precision:
                    g_kwargs["half"] = True

                g_results = self.general_model.predict(**g_kwargs)
                if g_results and len(g_results) > 0 and g_results[0].boxes is not None:
                    g_res = g_results[0]
                    for box in g_res.boxes:
                        g_cls_id = int(box.cls[0].cpu().numpy())
                        g_name = str(g_res.names.get(g_cls_id, "")).strip().lower()
                        if g_name in WasteDetector.COCO_WASTE_MAP:
                            cat, obj_label = WasteDetector.COCO_WASTE_MAP[g_name]
                            gx1, gy1, gx2, gy2 = box.xyxy[0].cpu().numpy()
                            g_conf = float(box.conf[0].cpu().numpy())
                            # Ambiguous items like books or electronics must have high confidence (>= 0.60 for book)
                            # to avoid misidentifying plastic containers, trays, packets, or blister packs
                            if g_name == "book" and g_conf < 0.60:
                                continue
                            if g_name in ("cell phone", "laptop", "mouse", "remote", "keyboard") and g_conf < 0.45:
                                continue
                            candidates.append({
                                "source": "general",
                                "class_id": g_cls_id,
                                "raw_name": g_name,
                                "class_name": cat,
                                "detected_object": obj_label,
                                "confidence": g_conf,
                                "bbox": (float(gx1), float(gy1), float(gx2), float(gy2)),
                            })

            # 2. Run custom trained model (specialized for plastic bottles and waste items)
            if self.model is not None and self.model != self.general_model:
                c_conf_thresh = max(0.18, self.config.confidence_threshold - 0.10)
                c_kwargs = {
                    "source": frame,
                    "conf": c_conf_thresh,
                    "iou": self.config.iou_threshold,
                    "imgsz": self.config.input_size,
                    "device": self.config.device,
                    "verbose": False,
                }
                if self.config.device != "cpu" and self.config.half_precision:
                    c_kwargs["half"] = True

                c_results = self.model.predict(**c_kwargs)
                if c_results and len(c_results) > 0 and c_results[0].boxes is not None:
                    c_res = c_results[0]
                    for box in c_res.boxes:
                        c_cls_id = int(box.cls[0].cpu().numpy())
                        c_name = str(c_res.names.get(c_cls_id, "unknown")).strip().lower()
                        cat = self.CUSTOM_MATERIAL_MAPPING.get(c_name) or self.normalize_material_name(c_name)
                        if cat == "unknown":
                            continue
                        cx1, cy1, cx2, cy2 = box.xyxy[0].cpu().numpy()
                        c_conf = float(box.conf[0].cpu().numpy())
                        c_box_area = (cx2 - cx1) * (cy2 - cy1)
                        # Soda/tin cans and beverage bottles are compact; a box covering > 35% of the entire frame is an overfitted hallucination
                        if c_name in ("metal_can", "glass_bottle") and c_box_area > 0.35 * frame_area:
                            continue
                        # Require at least 0.35 confidence for paper_cardboard to prevent misclassifying plastic objects
                        if cat == "paper" and c_conf < 0.35:
                            continue
                        candidates.append({
                            "source": "custom",
                            "class_id": c_cls_id,
                            "raw_name": c_name,
                            "class_name": cat,
                            "detected_object": self._custom_object_label(cat),
                            "confidence": c_conf,
                            "bbox": (float(cx1), float(cy1), float(cx2), float(cy2)),
                        })

            # Stage 2: Sensitive waste scan if primary pass yielded 0 candidates
            if len(candidates) == 0 and self.general_model is not None:
                try:
                    sens_kwargs = {
                        "source": frame,
                        "conf": 0.16,
                        "iou": self.config.iou_threshold,
                        "imgsz": self.config.input_size,
                        "device": self.config.device,
                        "verbose": False,
                    }
                    if self.config.device != "cpu" and self.config.half_precision:
                        sens_kwargs["half"] = True
                    s_results = self.general_model.predict(**sens_kwargs)
                    if s_results and len(s_results) > 0 and s_results[0].boxes is not None:
                        for box in s_results[0].boxes:
                            s_cls_id = int(box.cls[0].cpu().numpy())
                            s_name = str(s_results[0].names.get(s_cls_id, "")).strip().lower()
                            if s_name in WasteDetector.COCO_WASTE_MAP:
                                cat, obj_label = WasteDetector.COCO_WASTE_MAP[s_name]
                                sx1, sy1, sx2, sy2 = box.xyxy[0].cpu().numpy()
                                s_conf = float(box.conf[0].cpu().numpy())
                                if s_name in ("book", "cell phone", "laptop") and s_conf < 0.40:
                                    continue
                                candidates.append({
                                    "source": "general_sensitive",
                                    "class_id": s_cls_id,
                                    "raw_name": s_name,
                                    "class_name": cat,
                                    "detected_object": obj_label,
                                    "confidence": s_conf,
                                    "bbox": (float(sx1), float(sy1), float(sx2), float(sy2)),
                                })
                except Exception as se:
                    logger.debug(f"Sensitive waste scan error: {se}")

            inference_ms = (time.time() - start_time) * 1000
            waste_frame.inference_time_ms = inference_ms
            self.total_inference_time += inference_ms
            self.frame_count += 1

            # 3. Reconcile overlapping detections
            # Sort candidates: Plastic detections and high-confidence detections get top priority
            def _candidate_priority(c):
                plastic_bonus = self._plastic_bonus if c["class_name"] == "plastic" else 0.0
                custom_bonus = self._custom_plastic_bonus if (c["source"] == "custom" and c["class_name"] == "plastic") else 0.0
                return c["confidence"] + plastic_bonus + custom_bonus

            candidates.sort(key=_candidate_priority, reverse=True)

            reconciled = []
            for cand in candidates:
                overlapping = False
                for rec in reconciled:
                    iou = self._compute_iou(cand["bbox"], rec["bbox"])
                    if iou > 0.25:
                        overlapping = True
                        # Do not allow custom metal or paper to override a recognized plastic object (chair, container, bottle, etc.)
                        if rec["class_name"] == "plastic" and cand["class_name"] in ("metal", "paper", "cardboard", "unknown"):
                            break
                        # If incoming is plastic and existing was paper/cardboard/unknown, plastic takes precedence
                        keep_confident_custom = (
                            self.taxonomy == "material7-v1"
                            and rec.get("source") == "custom"
                            and rec["confidence"] >= 0.50
                        )
                        if (
                            cand["class_name"] == "plastic"
                            and rec["class_name"] in ("paper", "cardboard", "unknown")
                            and not keep_confident_custom
                        ):
                            if cand["confidence"] >= rec["confidence"] * 0.65 or rec["confidence"] < 0.85:
                                rec["class_name"] = "plastic"
                                rec["detected_object"] = cand["detected_object"]
                                rec["confidence"] = max(rec["confidence"], cand["confidence"])
                                rec["bbox"] = cand["bbox"]
                                rec["raw_name"] = cand["raw_name"]
                        # If both detected bottle / plastic, use highest confidence
                        elif (
                            ("bottle" in cand["raw_name"] or cand["class_name"] == "plastic")
                            and ("bottle" in rec["raw_name"] or rec["class_name"] == "plastic")
                            and cand["confidence"] > rec["confidence"]
                        ):
                            rec["confidence"] = cand["confidence"]
                            rec["bbox"] = cand["bbox"]
                            rec["detected_object"] = cand["detected_object"]
                        elif cand["confidence"] > rec["confidence"] + 0.15:
                            rec["class_name"] = cand["class_name"]
                            rec["detected_object"] = cand["detected_object"]
                            rec["confidence"] = cand["confidence"]
                            rec["bbox"] = cand["bbox"]
                            rec["raw_name"] = cand["raw_name"]
                        break
                if not overlapping:
                    reconciled.append(cand)

            # Build final WasteDetection objects
            for item in reconciled:
                x1, y1, x2, y2 = item["bbox"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                area = (x2 - x1) * (y2 - y1)

                detection = WasteDetection(
                    class_id=item["class_id"],
                    class_name=item["class_name"],
                    confidence=item["confidence"],
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    centroid=(float(cx), float(cy)),
                    area=float(area),
                    timestamp=timestamp,
                    detected_object=item["detected_object"],
                )
                waste_frame.detections.append(detection)

            # Stage 3: Salient material fallback if 0 bounding boxes detected
            if len(waste_frame.detections) == 0:
                fallback_det = self._salient_material_fallback(frame)
                if fallback_det is not None:
                    waste_frame.detections.append(fallback_det)

            # ────────────────────────────────────────────────────────────────
            # Zero detection diagnostics
            # ────────────────────────────────────────────────────────────────

            if len(waste_frame.detections) == 0:
                self._no_det_frames += 1
                if (
                    self.config.log_no_detections
                    and self._no_det_frames % 30 == 1
                ):
                    logger.debug(
                        f"No waste detected in last "
                        f"{self._no_det_frames} frames. "
                        f"Confidence threshold: "
                        f"{self.config.confidence_threshold:.2f}"
                    )
            else:
                self._no_det_frames = 0

            # ────────────────────────────────────────────────────────────────
            # FPS tracker
            # ────────────────────────────────────────────────────────────────

            self._fps_times.append(
                time.time()
            )

            if len(self._fps_times) > 60:

                self._fps_times.pop(0)

        except Exception as e:

            logger.error(
                f"Detection error: {e}"
            )

        return waste_frame


    # ─── Visualisation Helpers ────────────────────────────────────────────────

    @staticmethod
    def _alpha_rect(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        alpha: float = 0.45,
    ) -> None:

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return

        sub = img[
            y1:y2,
            x1:x2
        ]

        if sub.size == 0:
            return

        overlay = np.full(
            sub.shape,
            color,
            dtype=np.uint8
        )

        cv2.addWeighted(
            overlay,
            alpha,
            sub,
            1 - alpha,
            0,
            sub
        )

        img[
            y1:y2,
            x1:x2
        ] = sub


    @staticmethod
    def _corner_brackets(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
        length: int = 18,
    ) -> None:

        for cx, cy, sx, sy in [
            (x1, y1, +1, +1),
            (x2, y1, -1, +1),
            (x1, y2, +1, -1),
            (x2, y2, -1, -1),
        ]:

            cv2.line(
                img,
                (cx, cy),
                (cx + sx * length, cy),
                color,
                thickness
            )

            cv2.line(
                img,
                (cx, cy),
                (cx, cy + sy * length),
                color,
                thickness
            )


    def _neon_bbox(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:

        dim_color = tuple(
            max(0, int(c * 0.45))
            for c in color
        )

        cv2.rectangle(
            img,
            (x1 - 2, y1 - 2),
            (x2 + 2, y2 + 2),
            dim_color,
            4
        )

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            1
        )

        self._corner_brackets(
            img,
            x1,
            y1,
            x2,
            y2,
            color,
            thickness=2,
            length=16
        )

        font = cv2.FONT_HERSHEY_DUPLEX

        font_scale = 0.48

        thickness = 1

        (tw, th), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        pad = 5

        lx1 = x1

        ly1 = max(
            0,
            y1 - th - pad * 2 - 1
        )

        lx2 = x1 + tw + pad * 2

        ly2 = y1

        self._alpha_rect(
            img,
            lx1,
            ly1,
            lx2,
            ly2,
            color,
            alpha=0.75
        )

        cv2.rectangle(
            img,
            (lx1, ly1),
            (lx2, ly2),
            color,
            1
        )

        cv2.putText(
            img,
            label,
            (lx1 + pad, ly2 - pad + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            label,
            (lx1 + pad, ly2 - pad + 1),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


    # ─── Main Display ─────────────────────────────────────────────────────────

    def draw_detections(
        self,
        frame: np.ndarray,
        waste_frame: WasteFrame,
        sensor_data: Optional[Dict] = None,
        is_fullscreen: bool = False,
    ) -> np.ndarray:

        img = frame.copy()

        h, w = img.shape[:2]

        # ─────────────────────────────────────────────────────────────────────
        # Dark vignette
        # ─────────────────────────────────────────────────────────────────────

        dark = img.copy()

        cv2.rectangle(
            dark,
            (0, 0),
            (w, h),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            dark,
            0.12,
            img,
            0.88,
            0,
            img
        )

        # ─────────────────────────────────────────────────────────────────────
        # Bounding boxes
        # ─────────────────────────────────────────────────────────────────────

        for detection in waste_frame.detections:

            x1, y1, x2, y2 = [
                int(v)
                for v in detection.bbox
            ]

            color = self.NEON_COLORS.get(
                detection.class_name,
                (220, 220, 220)
            )

            obj_title = (
                detection.detected_object.split(" (")[0]
                if getattr(detection, "detected_object", None)
                else detection.class_name
            )

            label = (
                f"{obj_title.upper()}  "
                f"{detection.confidence:.0%}"
            )

            self._neon_bbox(
                img,
                x1,
                y1,
                x2,
                y2,
                color,
                label
            )

        # ─────────────────────────────────────────────────────────────────────
        # Top status bar
        # ─────────────────────────────────────────────────────────────────────

        bar_h = 46

        self._alpha_rect(
            img,
            0,
            0,
            w,
            bar_h,
            (10, 10, 30),
            alpha=0.72
        )

        cv2.rectangle(
            img,
            (0, bar_h - 1),
            (w, bar_h),
            (0, 180, 255),
            1
        )

        fps = self.get_fps()

        model_tag = (
            ("WASTE AI v2" if self.taxonomy == "material7-v1" else "YOLO WASTE")
            if self.is_custom_model
            else "FALLBACK MODEL"
        )

        num_det = len(
            waste_frame.detections
        )

        inf_ms = (
            waste_frame.inference_time_ms
        )

        status_color = (
            (50, 255, 120)
            if num_det == 0
            else (0, 200, 255)
        )

        dot_color = (
            (50, 255, 50)
            if num_det == 0
            else (0, 80, 255)
        )

        font = cv2.FONT_HERSHEY_DUPLEX

        font_sm = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            img,
            "PLASTIC  DETECTOR",
            (12, 30),
            font,
            0.62,
            (0, 180, 255),
            1,
            cv2.LINE_AA
        )

        cv2.line(
            img,
            (220, 8),
            (220, bar_h - 8),
            (0, 100, 160),
            1
        )

        cv2.putText(
            img,
            f"FPS {fps:5.1f}",
            (230, 30),
            font,
            0.54,
            (80, 255, 200),
            1,
            cv2.LINE_AA
        )

        cv2.line(
            img,
            (330, 8),
            (330, bar_h - 8),
            (0, 100, 160),
            1
        )

        cv2.putText(
            img,
            f"INF {inf_ms:5.0f}ms",
            (340, 30),
            font,
            0.54,
            (255, 180, 50),
            1,
            cv2.LINE_AA
        )

        cv2.line(
            img,
            (455, 8),
            (455, bar_h - 8),
            (0, 100, 160),
            1
        )

        cv2.putText(
            img,
            f"DET {num_det:3d}",
            (465, 30),
            font,
            0.54,
            status_color,
            1,
            cv2.LINE_AA
        )

        cv2.line(
            img,
            (545, 8),
            (545, bar_h - 8),
            (0, 100, 160),
            1
        )

        cv2.putText(
            img,
            model_tag,
            (555, 30),
            font_sm,
            0.46,
            (200, 200, 60),
            1,
            cv2.LINE_AA
        )

        # Live indicator

        cv2.circle(
            img,
            (w - 60, bar_h // 2),
            6,
            dot_color,
            -1
        )

        cv2.circle(
            img,
            (w - 60, bar_h // 2),
            8,
            dot_color,
            1
        )

        cv2.putText(
            img,
            "LIVE",
            (w - 48, bar_h // 2 + 5),
            font_sm,
            0.42,
            dot_color,
            1,
            cv2.LINE_AA
        )

        # ─────────────────────────────────────────────────────────────────────
        # Detection panel
        # ─────────────────────────────────────────────────────────────────────

        panel_w = 190

        panel_x = w - panel_w

        panel_y = bar_h + 4

        panel_h = min(
            len(waste_frame.detections) * 38 + 40,
            h - bar_h - 55
        )

        if waste_frame.detections:

            self._alpha_rect(
                img,
                panel_x,
                panel_y,
                panel_x + panel_w,
                panel_y + panel_h,
                (5, 10, 30),
                alpha=0.68
            )

            cv2.rectangle(
                img,
                (panel_x, panel_y),
                (panel_x + panel_w, panel_y + panel_h),
                (0, 100, 180),
                1
            )

            cv2.putText(
                img,
                "DETECTIONS",
                (panel_x + 8, panel_y + 20),
                font_sm,
                0.44,
                (0, 160, 255),
                1,
                cv2.LINE_AA
            )

            cv2.line(
                img,
                (panel_x + 4, panel_y + 26),
                (panel_x + panel_w - 4, panel_y + 26),
                (0, 80, 140),
                1
            )

            for i, detection in enumerate(
                waste_frame.detections[:6]
            ):

                row_y = (
                    panel_y
                    + 44
                    + i * 36
                )

                dcol = self.NEON_COLORS.get(
                    detection.class_name,
                    (200, 200, 200)
                )

                cv2.circle(
                    img,
                    (panel_x + 12, row_y - 6),
                    5,
                    dcol,
                    -1
                )

                cv2.putText(
                    img,
                    detection.class_name.upper(),
                    (panel_x + 24, row_y - 2),
                    font_sm,
                    0.44,
                    dcol,
                    1,
                    cv2.LINE_AA
                )

                bar_full = panel_w - 28

                bar_fill = int(
                    bar_full
                    * detection.confidence
                )

                bar_y = row_y + 6

                self._alpha_rect(
                    img,
                    panel_x + 14,
                    bar_y - 6,
                    panel_x + 14 + bar_full,
                    bar_y + 2,
                    (30, 30, 60),
                    alpha=0.9
                )

                cv2.rectangle(
                    img,
                    (panel_x + 14, bar_y - 6),
                    (
                        panel_x + 14 + bar_fill,
                        bar_y + 2
                    ),
                    dcol,
                    -1
                )

                cv2.putText(
                    img,
                    f"{detection.confidence:.0%}",
                    (
                        panel_x
                        + 14
                        + bar_full
                        + 2,
                        bar_y + 2
                    ),
                    font_sm,
                    0.38,
                    dcol,
                    1,
                    cv2.LINE_AA
                )

        # ─────────────────────────────────────────────────────────────────────
        # Bottom sensor bar
        # ─────────────────────────────────────────────────────────────────────

        bbar_h = 40

        by_start = h - bbar_h

        self._alpha_rect(
            img,
            0,
            by_start,
            w,
            h,
            (10, 10, 30),
            alpha=0.72
        )

        cv2.rectangle(
            img,
            (0, by_start),
            (w, by_start + 1),
            (0, 100, 180),
            1
        )

        if sensor_data:

            bin_pct = sensor_data.get(
                "bin_fill_pct",
                0
            )

            temp_c = sensor_data.get(
                "temperature_c",
                0
            )

            hum_pct = sensor_data.get(
                "humidity_pct",
                0
            )

            gas = sensor_data.get(
                "gas_analog",
                0
            )

            src = (
                "[SIM]"
                if sensor_data.get(
                    "is_simulated",
                    True
                )
                else "[REAL]"
            )

            src_col = (
                (0, 180, 255)
                if sensor_data.get(
                    "is_simulated",
                    True
                )
                else (50, 255, 100)
            )

            bfill_max = 200

            bfill_px = int(
                bfill_max
                * min(
                    bin_pct / 100.0,
                    1.0
                )
            )

            if bin_pct < 70:
                bfill_col = (50, 255, 100)

            elif bin_pct < 85:
                bfill_col = (50, 180, 255)

            else:
                bfill_col = (50, 50, 255)

            cv2.putText(
                img,
                "BIN",
                (10, by_start + 26),
                font_sm,
                0.42,
                (160, 160, 160),
                1,
                cv2.LINE_AA
            )

            self._alpha_rect(
                img,
                40,
                by_start + 10,
                40 + bfill_max,
                by_start + 28,
                (30, 30, 60),
                alpha=0.9
            )

            cv2.rectangle(
                img,
                (40, by_start + 10),
                (
                    40 + bfill_px,
                    by_start + 28
                ),
                bfill_col,
                -1
            )

            cv2.rectangle(
                img,
                (40, by_start + 10),
                (
                    40 + bfill_max,
                    by_start + 28
                ),
                (80, 80, 120),
                1
            )

            cv2.putText(
                img,
                f"{bin_pct:.0f}%",
                (248, by_start + 26),
                font_sm,
                0.44,
                bfill_col,
                1,
                cv2.LINE_AA
            )

            sensors_txt = [
                (
                    f"T {temp_c:.1f}C",
                    (255, 150, 80)
                ),
                (
                    f"H {hum_pct:.0f}%",
                    (80, 200, 255)
                ),
                (
                    f"GAS {gas}",
                    (255, 80, 80)
                ),
                (
                    src,
                    src_col
                ),
            ]

            sx = 295

            for txt, col in sensors_txt:

                cv2.putText(
                    img,
                    txt,
                    (sx, by_start + 26),
                    font_sm,
                    0.44,
                    col,
                    1,
                    cv2.LINE_AA
                )

                sx += 100

        # ─────────────────────────────────────────────────────────────────────
        # Corner decorations
        # ─────────────────────────────────────────────────────────────────────

        accent = (0, 180, 255)

        corner_len = 30

        corners = [
            (0, bar_h, +1, +1),
            (w, bar_h, -1, +1),
            (0, by_start, +1, -1),
            (w, by_start, -1, -1),
        ]

        for cx, cy, sx, sy in corners:

            cv2.line(
                img,
                (cx, cy),
                (cx + sx * corner_len, cy),
                accent,
                2
            )

            cv2.line(
                img,
                (cx, cy),
                (cx, cy + sy * corner_len),
                accent,
                2
            )

        # ─────────────────────────────────────────────────────────────────────
        # Frame ID
        # ─────────────────────────────────────────────────────────────────────

        cv2.putText(
            img,
            f"FRAME #{waste_frame.frame_id:06d}",
            (12, h - bbar_h - 8),
            font_sm,
            0.38,
            (60, 80, 100),
            1,
            cv2.LINE_AA
        )

        return img


    # ─── Statistics ──────────────────────────────────────────────────────────

    def get_fps(self) -> float:

        if len(self._fps_times) < 2:
            return 0.0

        elapsed = (
            self._fps_times[-1]
            - self._fps_times[0]
        )

        if elapsed <= 0:
            return 0.0

        return (
            len(self._fps_times) - 1
        ) / elapsed


    def get_statistics(self) -> Dict:

        avg_t = (
            self.total_inference_time
            / self.frame_count
            if self.frame_count > 0
            else 0
        )

        return {
            "model": self.config.custom_model_path
            if self.is_custom_model
            else self.config.model_path,

            "is_custom": self.is_custom_model,

            "taxonomy": self.taxonomy,

            "classes": list(self.custom_class_names),

            "total_frames": self.frame_count,

            "avg_inference_ms": round(
                avg_t,
                2
            ),

            "current_fps": round(
                self.get_fps(),
                1
            ),

            "confidence_threshold":
                self.config.confidence_threshold,

            "zero_detection_frames":
                self._no_det_frames,
        }


# ─── Standalone Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print()
    print("WASTE DETECTOR — YOLO")
    print("=" * 50)

    detector = WasteDetector()

    if not detector.load_model():

        print("Could not load model")

        sys.exit(1)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Could not open webcam")

        sys.exit(1)

    print(
        "Webcam opened — hold waste objects "
        "in front of camera"
    )

    frame_id = 0

    WIN_NAME = (
        "AI Waste Detection | "
        "[F] Fullscreen [Q] Quit"
    )

    is_fullscreen = False

    cv2.namedWindow(
        WIN_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WIN_NAME,
        960,
        600
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        waste_frame = detector.detect(
            frame,
            frame_id
        )

        annotated = detector.draw_detections(
            frame,
            waste_frame
        )

        if waste_frame.detections:

            for detection in (
                waste_frame.detections
            ):

                print(
                    f"[{frame_id}] "
                    f"{detection.class_name:12s} "
                    f"{detection.confidence:.0%} "
                    f"bbox={detection.bbox}"
                )

        prop = (
            cv2.WINDOW_FULLSCREEN
            if is_fullscreen
            else cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            WIN_NAME,
            cv2.WND_PROP_FULLSCREEN,
            prop
        )

        cv2.imshow(
            WIN_NAME,
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q")
        ):
            break

        if key in (
            ord("f"),
            ord("F")
        ):

            is_fullscreen = (
                not is_fullscreen
            )

        frame_id += 1

    cap.release()

    cv2.destroyAllWindows()

    print()
    print(
        f"Stats: {detector.get_statistics()}"
    )