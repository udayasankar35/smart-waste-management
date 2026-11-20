"""
Configuration Module — ZC702 Target Build
AI-Enabled Smart Waste Segregation & Monitoring System

Adapted for Xilinx ZC702 (Zynq-7000 XC7Z020) running PetaLinux/Yocto via Vivado.

Memory mapping:
  BRAM  (4.9 MB total)  → stores this config + lightweight Python modules
  DDR3  (1 GB)          → stores YOLO models, LLM, database, web assets

UART mapping (Zynq PS):
  Pico W serial → /dev/ttyPS0 (MIO UART1, 115200 baud)
  Debug console → /dev/ttyPS1 or via JTAG UART

Camera:
  USB UVC webcam → /dev/video0  (via USB3.0 or USB2.0 on ZC702 J7 USB port)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import os

# ─── ZC702 Storage Root ──────────────────────────────────────────────────────────
# The PS ARM cores run Linux; mount SD card or eMMC as root FS.
# External DDR3 (1 GB) is the main storage. Models are loaded from /mnt/models/

ZC702_ROOT       = "/opt/c2s"                     # App root on target Linux FS
ZC702_MODELS_DIR = "/mnt/models"                   # DDR3 side — large models
ZC702_DB_DIR     = "/mnt/data/database"            # DDR3 side — SQLite DB
ZC702_LOGS_DIR   = "/mnt/data/logs"               # DDR3 side — log files
ZC702_REPORTS_DIR= "/mnt/data/reports"            # DDR3 side — JSON/CSV reports


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


# ─── ZC702 GPIO / MIO Pin Configuration ─────────────────────────────────────────

class ZC702Pins:
    """
    Zynq-7000 ZC702 MIO / EMIO pin assignments.

    MIO connections (Zynq PS side — via PS7 IP in Vivado block design):
      - MIO14/15  → UART0 (console)
      - MIO48/49  → UART1 (Pico W serial, 115200 baud)
      - MIO50-53  → SPI0  (optional SPI sensor)
      - MIO40-43  → ETH0  (Gigabit Ethernet, used for web dashboard)

    EMIO connections (through PL fabric — requires Vivado AXI-GPIO IP):
      - EMIO[0]   → STATUS_LED  (drives PL LED DS23)
      - EMIO[1]   → SENSOR_TRIG (HC-SR04 ultrasonic trigger)
      - EMIO[2]   → SENSOR_ECHO (HC-SR04 ultrasonic echo)

    NOTE: HC-SR04, DHT11, MQ gas, moisture sensors connect via
    Raspberry Pi Pico W which streams JSON over UART1 (/dev/ttyPS0).
    """
    # Pico W Serial UART (PS MIO)
    UART_PORT = "/dev/ttyPS0"       # Zynq PS UART1 on ZC702
    UART_BAUD = 115200

    # EMIO-based LEDs (PL side, routed through Vivado AXI-GPIO)
    STATUS_LED_GPIO_NUM = 906       # Linux gpio-sysfs number (varies by kernel)

    # USB Camera
    CAMERA_DEV = "/dev/video0"      # USB UVC cam on ZC702 USB port J7


# ─── Sensor Thresholds ──────────────────────────────────────────────────────────

@dataclass
class SensorThresholds:
    bin_full_threshold: float  = 85.0
    bin_max_distance_cm: float = 30.0
    wet_waste_humidity:    float = 65.0
    high_humidity_alert:   float = 85.0
    gas_safe_level:    int = 400
    gas_warning_level: int = 700
    gas_danger_level:  int = 1000
    dry_waste_threshold: int = 20000
    wet_waste_threshold: int = 40000


# ─── Camera Configuration ────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    """USB UVC Webcam on ZC702"""
    device_index:  int = 0           # /dev/video0
    width:         int = 640         # Reduced from 1280 to ease DDR3 bandwidth
    height:        int = 480         # Reduced from 720
    fps:           int = 15          # Reduced FPS for Zynq PS A9 processing
    warmup_frames: int = 5


# ─── Detection Configuration ────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """
    YOLOv8 detection config for ZC702.

    Models stored in DDR3 (/mnt/models/).
    Use yolov8n (smallest, 6MB) on the Zynq A9 CPU.
    best.pt (22MB) only if performance allows.
    """
    model_path: str = os.path.join(ZC702_MODELS_DIR, "best.pt")
    custom_model_path: str = os.path.join(ZC702_MODELS_DIR, "best.pt")
    confidence_threshold: float = 0.28
    iou_threshold:        float = 0.45
    input_size:           int   = 416          # Reduced from 640 for A9 speed
    device:               str   = "cpu"        # Zynq PS A9 — no GPU
    half_precision:       bool  = False
    log_no_detections:    bool  = True


# ─── Database Configuration ─────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    """SQLite in DDR3 (external memory — not in BRAM)"""
    db_path:               str = os.path.join(ZC702_DB_DIR, "waste_monitoring.db")
    max_records:           int = 10_000        # Reduced from 100K for embedded
    backup_interval_hours: int = 24


# ─── Report Configuration ───────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    output_dir:               str  = ZC702_REPORTS_DIR
    json_enabled:             bool = True
    csv_enabled:              bool = False
    pdf_enabled:              bool = False
    report_interval_minutes:  int  = 60


# ─── Alert Configuration ────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    enabled:          bool = True
    email_enabled:    bool = False      # No SMTP on embedded target
    smtp_server:      str  = ""
    smtp_port:        int  = 587
    sender_email:     str  = ""
    sender_password:  str  = ""
    recipient_emails: List[str] = field(default_factory=list)
    console_alerts:   bool  = True
    alert_cooldown_sec: float = 300.0


# ─── Vehicle Configuration ──────────────────────────────────────────────────────

@dataclass
class VehicleConfig:
    vehicle_id:   str = "ZC702-01"
    vehicle_name: str = "Zynq Smart Waste Node #1"
    route_id:     str = "ROUTE-ZC702"


# ─── Serial Configuration (Pico W → ZC702 UART1) ────────────────────────────────

@dataclass
class SerialConfig:
    """
    Serial connection from Pico W to Zynq ZC702.
    Physical: Pico W USB/UART TX → ZC702 MIO49 (UART1 RX).
    On Linux: /dev/ttyPS0
    """
    port:               str   = ZC702Pins.UART_PORT   # /dev/ttyPS0
    baudrate:           int   = 115200
    timeout:            float = 1.0
    read_interval_sec:  float = 1.0
    strict_sensor_mode: bool  = False
    log_simulated_data: bool  = True


# ─── Llama / LLM Configuration ─────────────────────────────────────────────────

@dataclass
class LlamaConfig:
    """
    Llama 3.2 1B on ZC702 Zynq A9.
    GGUF model stored in DDR3 (/mnt/models/).
    ExecuTorch PTE runtime preferred for Arm SoC challenge.
    """
    gguf_model_path: str  = os.path.join(ZC702_MODELS_DIR, "llama-3.2-1b-instruct-q4_k_m.gguf")
    pte_model_path:  str  = os.path.join(ZC702_MODELS_DIR, "llama3_2_1b.pte")
    model_path:      str  = os.path.join(ZC702_MODELS_DIR, "llama3_2_1b.pte")
    tokenizer_path:  str  = os.path.join(ZC702_MODELS_DIR, "tokenizer.model")
    enabled:         bool  = True
    backend:         str   = "gguf"       # gguf is safest on generic ARM Linux
    max_tokens:      int   = 200          # Reduced for latency on A9
    context_length:  int   = 1024         # Reduced from 2048
    temperature:     float = 0.6
    top_p:           float = 0.9


# ─── Master System Configuration ────────────────────────────────────────────────

@dataclass
class SystemConfig:
    camera:    CameraConfig    = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sensors:   SensorThresholds = field(default_factory=SensorThresholds)
    database:  DatabaseConfig  = field(default_factory=DatabaseConfig)
    reports:   ReportConfig    = field(default_factory=ReportConfig)
    alerts:    AlertConfig     = field(default_factory=AlertConfig)
    vehicle:   VehicleConfig   = field(default_factory=VehicleConfig)
    serial:    SerialConfig    = field(default_factory=SerialConfig)
    llama:     LlamaConfig     = field(default_factory=LlamaConfig)

    log_level: str = "INFO"
    log_file:  str = os.path.join(ZC702_LOGS_DIR, "waste_system.log")

    def __post_init__(self):
        """Create necessary directories on startup (target Linux FS)"""
        for d in [
            ZC702_DB_DIR,
            ZC702_REPORTS_DIR,
            ZC702_LOGS_DIR,
            ZC702_MODELS_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


DEFAULT_CONFIG = SystemConfig()

