"""
Configuration Module
AI-Enabled Smart Waste Segregation & Monitoring System
Centralized configuration for Jetson Nano + Raspberry Pi Pico W.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import os


# ─── Waste Classification Classes ───────────────────────────────────────────────

class WasteType(Enum):
    """Waste categories detected by the AI system"""
    PLASTIC    = "plastic"
    GLASS      = "glass"
    METAL      = "metal"
    PAPER      = "paper"
    CARDBOARD  = "cardboard"
    ORGANIC    = "organic"
    HAZARDOUS  = "hazardous"
    DRY_WASTE  = "dry_waste"
    WET_WASTE  = "wet_waste"
    UNKNOWN    = "unknown"


# YOLO class index → WasteType mapping (multi-material trained best.pt model)
WASTE_CLASS_MAP: Dict[int, str] = {
    0: "plastic",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "organic",
}

WASTE_CLASS_NAMES: List[str] = list(WASTE_CLASS_MAP.values())


# ─── Hardware Pin Configuration (Raspberry Pi Pico W) ───────────────────────────

class PicoPins:
    """
    GPIO pin assignments for Raspberry Pi Pico W sensors.
    These MUST match the physical wiring on the Pico.
    """

    # HC-SR04 Ultrasonic Distance Sensor
    ULTRASONIC_TRIG = 3     # GP3  → TRIG pin of HC-SR04
    ULTRASONIC_ECHO = 2     # GP2  → ECHO pin of HC-SR04

    # DHT11 Temperature & Humidity Sensor
    DHT11_DATA = 15         # GP15 → DATA pin of DHT11

    # MQ Gas Sensor Module
    GAS_ANALOG  = 26        # GP26 (ADC0) → AO (analog out) of MQ sensor
    GAS_DIGITAL = 16        # GP16         → DO (digital out) of MQ sensor

    # HW-103 Soil Moisture Sensor
    MOISTURE_ANALOG = 27    # GP27 (ADC1) → AO of moisture sensor

    # Status LED (on-board)
    STATUS_LED = "LED"      # Built-in LED ("LED" on Pico W)


# ─── Sensor Thresholds ──────────────────────────────────────────────────────────

@dataclass
class SensorThresholds:
    """Threshold values for sensor-based waste classification & alerts"""

    # Bin fill level (ultrasonic) — percentage
    bin_full_threshold: float  = 85.0       # Alert when bin > 85% full
    bin_max_distance_cm: float = 30.0       # Empty-bin distance (sensor to bottom)

    # Humidity (DHT11) — percentage
    wet_waste_humidity:    float = 65.0     # Above this → likely wet waste present
    high_humidity_alert:   float = 85.0     # Environmental alert level

    # Gas sensor (MQ) — raw ADC uint16 (0–65535)
    gas_safe_level:    int = 400            # Normal air quality
    gas_warning_level: int = 700            # Warning — increased gas
    gas_danger_level:  int = 1000           # Danger / possible toxic gas

    # Moisture sensor (HW-103) — raw ADC uint16 (0–65535)
    dry_waste_threshold: int = 20000        # Below this  → dry waste  (low moisture)
    wet_waste_threshold: int = 40000        # Above this  → wet waste  (high moisture)


# ─── Camera Configuration ────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    """USB Webcam configuration for Jetson Nano"""
    device_index:  int = 0          # /dev/video0
    width:         int = 1280
    height:        int = 720
    fps:           int = 30
    warmup_frames: int = 10         # Discard first N frames on startup


# ─── Detection Configuration ────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """
    YOLOv8 waste detection configuration.

    !! IMPORTANT — Custom waste model training !!
    The system ships with a generic YOLOv8n pretrained on COCO.
    For true plastic/waste detection accuracy, train a custom model:
      1. Collect waste images (plastic bags, bottles, paper, etc.)
         OR download a public dataset:
         - TACO: http://tacodataset.org/
         - OpenImages waste subset
         - Roboflow Universe: waste-detection collections
      2. Label with Roboflow or LabelImg using classes:
         plastic, recyclable, metal, dry_waste, wet_waste, hazardous
      3. Run: python train_waste_model.py
    4. Place trained weights at: yolo/weights/waste_yolov8n.pt
      The system will auto-detect and use the custom model.
    """
    model_path:          str   = os.path.join(
        os.path.dirname(__file__), "yolo", "weights",
        "ecosort_waste_yolo11s.pt" if os.path.exists(os.path.join(os.path.dirname(__file__), "yolo", "weights", "ecosort_waste_yolo11s.pt")) else "weights.pt"
    )
    custom_model_path:   str   = os.path.join(
        os.path.dirname(__file__), "yolo", "weights",
        "ecosort_waste_yolo11s.pt" if os.path.exists(os.path.join(os.path.dirname(__file__), "yolo", "weights", "ecosort_waste_yolo11s.pt")) else "weights.pt"
    )
    confidence_threshold: float = 0.28                      # Tuned for high-recall waste detection
    iou_threshold:        float = 0.45
    input_size:           int   = 640                       # Model input resolution (px)
    device:               str   = "cpu"                     # GPU device ("0") or "cpu"
    half_precision:       bool  = False                     # Set True for Jetson GPU FP16
    log_no_detections:    bool  = True                      # Log when 0 objects detected


# ─── Database Configuration ─────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    """SQLite database configuration"""
    db_path:               str = "database/waste_monitoring.db"
    max_records:           int = 100_000
    backup_interval_hours: int = 24


# ─── Report Configuration ───────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    """Report generation configuration"""
    output_dir:               str  = "reports/json"
    json_enabled:             bool = True
    csv_enabled:              bool = False
    pdf_enabled:              bool = False
    report_interval_minutes:  int  = 60        # Auto-generate every hour


# ─── Alert Configuration ────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    """Alert / notification configuration"""
    enabled:          bool = True

    # Email (SMTP)
    email_enabled:    bool = False
    smtp_server:      str  = "smtp.gmail.com"
    smtp_port:        int  = 587
    sender_email:     str  = ""
    sender_password:  str  = ""
    recipient_emails: List[str] = field(default_factory=list)

    # Console / log alerts (always active)
    console_alerts:   bool  = True

    # Cooldown between repeated alerts (seconds)
    alert_cooldown_sec: float = 300.0          # 5 minutes


# ─── Vehicle Configuration ──────────────────────────────────────────────────────

@dataclass
class VehicleConfig:
    """Garbage collection vehicle metadata"""
    vehicle_id:   str = "GV-12"
    vehicle_name: str = "Smart Waste Truck #12"
    route_id:     str = "ROUTE-A1"


# ─── Serial Communication ───────────────────────────────────────────────────────

@dataclass
class SerialConfig:
    """
    Serial (UART) communication: Raspberry Pi Pico W → Jetson Nano

    strict_sensor_mode:
        True  → System ABORTS if Pico is not physically connected.
                 No simulated sensor data is ever used.
        False → System falls back to simulated readings and logs a warning.
                 Useful for development without hardware.
    """
    port:               str   = "/dev/ttyACM0"   # USB serial from Pico (ACM0 or ACM1)
    baudrate:           int   = 115200
    timeout:            float = 1.0
    read_interval_sec:  float = 1.0              # Read sensor data every 1 second
    strict_sensor_mode: bool  = False            # Set True in production
    log_simulated_data: bool  = True             # Warn on every simulated reading


# ─── Llama / ExecuTorch Configuration ─────────────────────────────────────────

@dataclass
class LlamaConfig:
    """
    Llama 3.2 On-Device LLM Configuration.
    Bharat AI-SoC Challenge 2026–27 mandatory requirement.

    Supports two runtimes:
      1. llama-cpp-python + GGUF  — universal (Windows/Linux/Mac/Arm SoC)
      2. ExecuTorch + .pte        — Arm SoC evaluation target

    Model download:
        python download_model.py        # downloads ~750 MB GGUF to models/
    """
    # ── GGUF runtime (primary — cross-platform local inference) ──────────
    gguf_model_path: str  = os.path.join(os.path.dirname(__file__), "models", "llama-3.2-1b-instruct-q4_k_m.gguf")

    # ── ExecuTorch runtime (Arm SoC challenge target) ───────────────────
    pte_model_path:  str  = os.path.join(os.path.dirname(__file__), "models", "llama3_2_1b.pte")
    model_path:      str  = os.path.join(os.path.dirname(__file__), "models", "llama3_2_1b.pte")
    tokenizer_path:  str  = os.path.join(os.path.dirname(__file__), "models", "tokenizer.model")

    # ── General ─────────────────────────────────────────────────────────
    enabled:         bool  = True
    backend:         str   = "auto"   # "auto", "gguf", "executorch", "fallback"
    max_tokens:      int   = 350
    context_length:  int   = 2048
    temperature:     float = 0.6
    top_p:           float = 0.9


# ─── Master System Configuration ────────────────────────────────────────────────

@dataclass
class SystemConfig:
    """Master configuration combining all subsystems"""
    camera:    CameraConfig    = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sensors:   SensorThresholds = field(default_factory=SensorThresholds)
    database:  DatabaseConfig  = field(default_factory=DatabaseConfig)
    reports:   ReportConfig    = field(default_factory=ReportConfig)
    alerts:    AlertConfig     = field(default_factory=AlertConfig)
    vehicle:   VehicleConfig   = field(default_factory=VehicleConfig)
    serial:    SerialConfig    = field(default_factory=SerialConfig)
    llama:     LlamaConfig     = field(default_factory=LlamaConfig)

    # Logging
    log_level: str = "INFO"
    log_file:  str = "logs/waste_system.log"

    def __post_init__(self):
        """Create necessary directories on startup"""
        for d in [
            "database",
            "reports/json",
            "logs",
            "yolo/weights",
            "yolo/training_data/waste",
            "evidence",
            "models",
        ]:
            os.makedirs(d, exist_ok=True)


# ─── Default Configuration ──────────────────────────────────────────────────────

DEFAULT_CONFIG = SystemConfig()

