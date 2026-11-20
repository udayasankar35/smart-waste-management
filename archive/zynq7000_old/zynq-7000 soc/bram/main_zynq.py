#!/usr/bin/env python3
"""
Zynq-7000 SoC (ZedBoard) Main System Entry Point
AI-Enabled Smart Waste Segregation & Monitoring System

Runs on the Dual ARM Cortex-A9 cores under Linux:
  - Critical control & fusion code kept in BRAM / OCM / Fast RAM
  - Heavy models (YOLO & Llama) & persistent SQLite DB stored in 8 GB External Memory
  - Direct hardware status indication on ZedBoard LEDs (LD0-LD7)

Usage on ZedBoard:
    python3 main_zynq.py                   # Standard autonomous headless run
    python3 main_zynq.py --test            # 10-frame self test
    python3 main_zynq.py --serial /dev/ttyUSB0   # Override Pico serial port
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

# ── Dynamic Path Setup ────────────────────────────────────────────────────────
# Add current directory (BRAM modules) and External storage modules to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# External memory path (where waste_detector.py, waste_advisor.py live)
EXT_DIR = os.environ.get("C2S_EXT_STORAGE", "/mnt/ext_storage")
sys.path.insert(0, EXT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "external"))

# Import BRAM modules
from config_zynq import DEFAULT_CONFIG, SystemConfig, ZedBoardHardware
from alert_system import AlertSystem
from data_fusion import DataFusionEngine, PicoSerialReader
from database import WasteDatabase
from report_generator import ReportGenerator
from webcam_capture import WebcamCapture

# Import External modules (from 8GB external memory)
try:
    from waste_detector import WasteDetector
except ImportError:
    # Look in adjacent external directory
    sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "external")))
    from waste_detector import WasteDetector


# ─── ZedBoard Hardware LED Controller ──────────────────────────────────────────

class ZedBoardLEDs:
    """Controls the 8 onboard LEDs (LD0..LD7) on ZedBoard via sysfs GPIO if exported"""
    def __init__(self):
        self.available = False
        # Try standard Zynq Linux GPIO base (typically 906-913 or 960-967 depending on kernel)
        for base in [906, 960, 1016]:
            if os.path.exists(f"/sys/class/gpio/gpiochip{base}"):
                self.base = base
                self.available = True
                break

    def set_heartbeat(self, state: bool):
        """Toggle LD0 as system heartbeat"""
        if not self.available:
            return
        try:
            pin = self.base
            path = f"/sys/class/gpio/gpio{pin}/value"
            if not os.path.exists(path):
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(pin))
                with open(f"/sys/class/gpio/gpio{pin}/direction", "w") as f:
                    f.write("out")
            with open(path, "w") as f:
                f.write("1" if state else "0")
        except Exception:
            pass


# ─── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging(config: SystemConfig):
    log_file = config.log_file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "node": "ZedBoard_Zynq7000",
                "message": record.getMessage(),
            }
            return json.dumps(payload)

    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)
    except Exception as e:
        print(f"Warning: Could not open log file on external memory: {e}")

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=handlers,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

logger = logging.getLogger("ZynqMain")


# ─── Graceful Shutdown ─────────────────────────────────────────────────────────

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal caught — terminating gracefully...")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── Main Execution Pipeline ──────────────────────────────────────────────────

def run_system(config: SystemConfig, test_mode: bool = False) -> bool:
    global running

    leds = ZedBoardLEDs()
    logger.info("=" * 60)
    logger.info("  C2S SMART WASTE SYSTEM — ZYNQ-7000 (ZedBoard)")
    logger.info("=" * 60)
    logger.info(f"Target Board:        ZedBoard (XC7Z020)")
    logger.info(f"External Memory:     8 GB attached (Storage/Models)")
    logger.info(f"Model Path:          {config.detection.model_path}")
    logger.info(f"Database Path:       {config.database.db_path}")
    logger.info(f"Pico Sensor UART:    {config.serial.port}")
    logger.info("=" * 60)

    # 1. Initialize Subsystems
    detector      = WasteDetector(config.detection)
    camera        = WebcamCapture(config.camera)
    serial_reader = PicoSerialReader(config.serial)
    fusion        = DataFusionEngine(config.sensors, config.vehicle)
    db            = WasteDatabase(config.database.db_path)
    alerts        = AlertSystem(config.alerts, config.sensors)

    # 2. Load AI Model from External 8GB Storage
    logger.info("Loading YOLOv8 weights from 8GB external memory...")
    if not detector.load_model():
        logger.warning(f"Could not load primary model at {config.detection.model_path}")
        logger.info(f"Attempting fallback model: {config.detection.fallback_model_path}")
        config.detection.model_path = config.detection.fallback_model_path
        if not detector.load_model():
            logger.error("Failed to load any YOLO model. Aborting.")
            return False

    # 3. Open Camera (or fallback to test mode if camera not yet plugged into USB OTG)
    if not test_mode:
        if not camera.open():
            logger.warning("Camera not detected on USB OTG — running in simulated camera mode")
            test_mode = True

    # 4. Connect to Raspberry Pi Pico W Serial
    try:
        pico_connected = serial_reader.connect()
    except Exception as e:
        logger.warning(f"Serial connection notice: {e}")
        pico_connected = False

    logger.info(f"Sensors: {'CONNECTED (Real UART)' if pico_connected else 'SIMULATED (Fallback)'}")
    logger.info("Entering processing loop...")

    frame_id         = 0
    last_report_time = time.time()
    report_interval  = config.reports.report_interval_minutes * 60
    heartbeat_state  = False

    try:
        while running:
            loop_start = time.time()

            # Heartbeat LED toggle
            heartbeat_state = not heartbeat_state
            leds.set_heartbeat(heartbeat_state)

            # Step 1: Capture Frame
            if test_mode:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

            # Step 2: YOLO AI Waste Classification
            waste_frame = detector.detect(frame, frame_id)

            # Step 3: Sensor Data Acquisition
            try:
                sensor_reading = serial_reader.read_sensors()
            except RuntimeError as e:
                logger.error(f"Sensor error: {e}")
                break

            # Step 4: Multi-Sensor Data Fusion
            records = fusion.fuse(waste_frame, sensor_reading)

            # Step 5: Store in SQLite (8GB External Storage)
            source_tag = "simulated" if sensor_reading.is_simulated else "real"
            for record in records:
                record_dict = record.to_dict()
                db.insert_waste_record(record_dict)

                db.insert_sensor_reading({
                    "timestamp":     record.timestamp,
                    "distance_cm":   sensor_reading.distance_cm,
                    "bin_fill_pct":  sensor_reading.bin_fill_pct,
                    "temperature_c": sensor_reading.temperature_c,
                    "humidity_pct":  sensor_reading.humidity_pct,
                    "gas_analog":    sensor_reading.gas_analog,
                    "gas_digital":   sensor_reading.gas_digital,
                    "moisture_raw":  sensor_reading.moisture_raw,
                    "data_source":   source_tag,
                })

                # Step 6: Threshold & Safety Alerts
                alerts.check_and_alert(record_dict, db)

            # Step 7: Periodic Health & Snapshot Reporting
            now = time.time()
            if now - last_report_time > report_interval:
                logger.info("Generating periodic report snapshot to external memory...")
                recent = db.get_recent_records(500)
                stats  = db.get_daily_statistics()
                payloads = ReportGenerator(config.reports).generate_all(
                    [dict(r) for r in recent], stats
                )
                if "json" in payloads:
                    db.insert_json_report("waste_monitoring", payloads["json"])
                last_report_time = now

            # Progress output
            frame_id += 1
            if frame_id % 30 == 0:
                stats = detector.get_statistics()
                logger.info(
                    f"[ZYNQ Frame {frame_id:5d}] | "
                    f"FPS: {stats['current_fps']:4.1f} | "
                    f"Detections: {len(waste_frame.detections):2d} | "
                    f"Bin Fill: {sensor_reading.bin_fill_pct:5.1f}% | "
                    f"Gas ADC: {sensor_reading.gas_analog:4d}"
                )

            if test_mode and frame_id >= 10:
                logger.info(f"Self-test complete ({frame_id} frames). Exiting.")
                break

            # Sleep to match camera FPS constraint
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / config.camera.fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as exc:
        logger.error(f"Execution error: {exc}", exc_info=True)
    finally:
        camera.stop()
        serial_reader.close()
        leds.set_heartbeat(False)
        logger.info("Zynq-7000 processing stopped cleanly.")

    return True


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="C2S Waste Monitoring System — Zynq-7000 (ZedBoard)"
    )
    parser.add_argument("--test",   action="store_true", help="Run 10-frame dry-run self test")
    parser.add_argument("--serial", type=str,            help="Override Pico UART port (/dev/ttyACM0, /dev/ttyUSB0)")
    parser.add_argument("--model",  type=str,            help="Override model path in 8GB external memory")
    parser.add_argument("--strict", action="store_true", help="Abort if Pico sensor is disconnected")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.serial:
        config.serial.port = args.serial
    if args.model:
        config.detection.model_path = args.model
    if args.strict:
        config.serial.strict_sensor_mode = True

    setup_logging(config)
    success = run_system(config, test_mode=args.test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

