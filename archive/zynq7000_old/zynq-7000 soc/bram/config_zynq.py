"""
Configuration Module — Zynq-7000 SoC (ZedBoard Target)
AI-Enabled Smart Waste Segregation & Monitoring System

Target Hardware:
  - Board: Avnet / Digilent ZedBoard (Zynq-7000 XC7Z020-CLG484-1)
  - Processing System: Dual-Core ARM Cortex-A9 @ 667 MHz
  - On-Board RAM: 512 MB DDR3
  - External Memory: 8 GB High-Speed Storage (SD Card / USB Storage / eMMC)
  - On-Chip Memory (BRAM / OCM): 256 KB PS OCM + 630 KB PL Block RAM

Memory Hierarchy:
  1. BRAM / OCM (Fastest, zero-wait):
     - Critical control loop, fast I/O handlers, alert triggers, sensor fusion,
       and AXI-BRAM buffer structures (mapped at 0x40000000 in PL).
  2. 512 MB Physical DDR3:
     - Linux kernel, active application process execution, OpenCV frame buffers.
  3. 8 GB External Memory (Mounted at /mnt/ext_storage or rootfs):
     - YOLOv8 weights (best.pt, yolov8n.pt), Llama 3.2 1B Instruct model,
       SQLite persistent database, web assets, and 2GB virtual swap space.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ─── Target System Path Mapping ──────────────────────────────────────────────────
# Files in BRAM/OCM run from /opt/c2s (or local RAM-disk)
APP_ROOT         = "/opt/c2s"

# 8 GB External Memory Mount (SD Card ext4 partition or USB Drive)
EXT_STORAGE_ROOT = os.environ.get("C2S_EXT_STORAGE", "/mnt/ext_storage")

MODELS_DIR       = os.path.join(EXT_STORAGE_ROOT, "models")
DATABASE_DIR     = os.path.join(EXT_STORAGE_ROOT, "database")
LOGS_DIR         = os.path.join(EXT_STORAGE_ROOT, "logs")
REPORTS_DIR      = os.path.join(EXT_STORAGE_ROOT, "reports")
WEB_DIR          = os.path.join(EXT_STORAGE_ROOT, "web")

# AXI BRAM Physical Memory Map (Configured in Vivado Block Design)
AXI_BRAM_BASE_ADDR = 0x40000000
AXI_BRAM_HIGH_ADDR = 0x4003FFFF  # 256 KB memory space


# ─── Waste Classification Classes ───────────────────────────────────────────────

class WasteType(Enum):
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


WASTE_CLASS_MAP: Dict[int, str] = {
    0: "plastic",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "organic",
}

WASTE_CLASS_NAMES: List[str] = list(WASTE_CLASS_MAP.values())


# ─── ZedBoard Peripheral & Pin Mapping ──────────────────────────────────────────

class ZedBoardHardware:
    """
    Hardware assignments for ZedBoard (Zynq-7000):
      - Console UART: /dev/ttyPS0 (MIO 14..15 -> Micro-USB J14 on ZedBoard)
      - Sensor UART (Pico W):
          Option A (USB OTG Port J13 via hub): /dev/ttyACM0 or /dev/ttyUSB0
          Option B (Pmod JA / JB via EMIO UART1): /dev/ttyPS1
      - Camera: /dev/video0 (UVC Webcam via USB OTG Port J13)
      - AXI-GPIO:
          Channel 1: 8 User LEDs (LD0-LD7) at PL pins T22, T21, U22, U21, V22, W22, U19, U14
          Channel 2: 8 User DIP Switches (SW0-SW7) at PL pins F22, G22, H22, F21, H19, H18, H17, M15
    """
    UART_PICO_PORT       = "/dev/ttyACM0"  # Default Pico W USB serial on Linux
    UART_PICO_PORT_FALLBACK = "/dev/ttyPS1"
    UART_BAUDRATE        = 115200

    CAMERA_DEVICE        = "/dev/video0"

    # AXI GPIO Base in Linux (via UIO or sysfs gpiochip)
    AXI_GPIO_BASE_ADDR   = 0x41200000


# ─── Sensor Thresholds (BRAM Stored) ───────────────────────────────────────────

@dataclass
class SensorThresholds:
    bin_full_threshold: float  = 85.0       # Alert when bin > 85% full
    bin_max_distance_cm: float = 30.0       # Distance when bin is empty
    wet_waste_humidity:    float = 65.0     # Humidity % suggesting wet waste
    high_humidity_alert:   float = 85.0     # Critical humidity
    gas_safe_level:    int = 400            # MQ gas sensor raw ADC
    gas_warning_level: int = 700
    gas_danger_level:  int = 1000
    dry_waste_threshold: int = 20000        # Soil moisture sensor raw ADC
    wet_waste_threshold: int = 40000


# ─── Camera Configuration ────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    """Optimized for Zynq-7000 Cortex-A9 processing"""
    device_index:  int = 0          # /dev/video0
    width:         int = 640        # 640x480 standard resolution
    height:        int = 480
    fps:           int = 15         # 15 fps smooth processing for embedded A9
    warmup_frames: int = 5


# ─── Detection Configuration (External 8GB Storage) ─────────────────────────────

@dataclass
class DetectionConfig:
    """
    YOLOv8 configuration for Zynq-7000.
    Models are stored in external 8GB memory (MODELS_DIR) to preserve BRAM/OCM.
    """
    # Primary custom model
    model_path: str = os.path.join(MODELS_DIR, "best.pt")
    # Fallback lightweight model
    fallback_model_path: str = os.path.join(MODELS_DIR, "yolov8n.pt")
    custom_model_path: str = os.path.join(MODELS_DIR, "best.pt")

    confidence_threshold: float = 0.28
    iou_threshold:        float = 0.45
    input_size:           int   = 416       # 416x416 balance of accuracy and latency on Cortex-A9
    device:               str   = "cpu"     # ARM Cortex-A9 CPU
    half_precision:       bool  = False
    log_no_detections:    bool  = True


# ─── Database Configuration (External 8GB Storage) ──────────────────────────────

@dataclass
class DatabaseConfig:
    """SQLite database placed on 8GB external memory to prevent BRAM saturation"""
    db_path:               str = os.path.join(DATABASE_DIR, "waste_monitoring.db")
    max_records:           int = 100_000
    backup_interval_hours: int = 24


# ─── Report Configuration ───────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    output_dir:               str  = REPORTS_DIR
    json_enabled:             bool = True
    csv_enabled:              bool = True
    pdf_enabled:              bool = False
    report_interval_minutes:  int  = 60


# ─── Alert Configuration ────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    enabled:          bool = True
    email_enabled:    bool = False
    smtp_server:      str  = ""
    smtp_port:        int  = 587
    sender_email:     str  = ""
    sender_password:  str  = ""
    recipient_emails: List[str] = field(default_factory=list)
    console_alerts:   bool  = True
    alert_cooldown_sec: float = 300.0


# ─── Vehicle / Station Metadata ─────────────────────────────────────────────────

@dataclass
class VehicleConfig:
    vehicle_id:   str = "ZEDBOARD-ZYNQ01"
    vehicle_name: str = "Smart Waste Node (ZedBoard Zynq-7000)"
    route_id:     str = "ROUTE-ZED-01"


# ─── Serial Communication ───────────────────────────────────────────────────────

@dataclass
class SerialConfig:
    port:               str   = ZedBoardHardware.UART_PICO_PORT
    baudrate:           int   = ZedBoardHardware.UART_BAUDRATE
    timeout:            float = 1.0
    read_interval_sec:  float = 1.0
    strict_sensor_mode: bool  = False
    log_simulated_data: bool  = True


# ─── Llama On-Device LLM (Leveraging 8GB External Storage & Swap) ───────────────

@dataclass
class LlamaConfig:
    """
    On-device LLM configuration.
    With 8GB external memory, we can host the full GGUF model and support virtual swap.
    """
    gguf_model_path: str  = os.path.join(MODELS_DIR, "llama-3.2-1b-instruct-q4_k_m.gguf")
    pte_model_path:  str  = os.path.join(MODELS_DIR, "llama3_2_1b.pte")
    model_path:      str  = os.path.join(MODELS_DIR, "llama3_2_1b.pte")
    tokenizer_path:  str  = os.path.join(MODELS_DIR, "tokenizer.model")

    enabled:         bool  = True
    backend:         str   = "auto"      # "auto", "gguf", or "fallback"
    max_tokens:      int   = 200
    context_length:  int   = 1024
    temperature:     float = 0.6
    top_p:           float = 0.9


# ─── Master System Configuration ────────────────────────────────────────────────

@dataclass
class SystemConfig:
    camera:    CameraConfig     = field(default_factory=CameraConfig)
    detection: DetectionConfig  = field(default_factory=DetectionConfig)
    sensors:   SensorThresholds = field(default_factory=SensorThresholds)
    database:  DatabaseConfig   = field(default_factory=DatabaseConfig)
    reports:   ReportConfig     = field(default_factory=ReportConfig)
    alerts:    AlertConfig      = field(default_factory=AlertConfig)
    vehicle:   VehicleConfig    = field(default_factory=VehicleConfig)
    serial:    SerialConfig     = field(default_factory=SerialConfig)
    llama:     LlamaConfig      = field(default_factory=LlamaConfig)

    log_level: str = "INFO"
    log_file:  str = os.path.join(LOGS_DIR, "waste_system.log")

    def __post_init__(self):
        """Ensure directories exist on the 8GB external filesystem"""
        for directory in [DATABASE_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR]:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass


DEFAULT_CONFIG = SystemConfig()

