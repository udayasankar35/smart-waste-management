#!/usr/bin/env python3
"""
ZC702 Main Entry Point (Adapted)
AI-Enabled Smart Waste Segregation & Monitoring System

This is main.py adapted for the ZC702 Zynq-7000 target:
  - Uses config_zc702.py instead of config.py
  - UART port is /dev/ttyPS0 (Zynq PS UART1 → Pico W)
  - Camera on /dev/video0 (USB UVC via PS USB OTG)
  - All models and DB are in DDR3 (/mnt/models/, /mnt/data/)
  - Runs headless by default (no X11 display on PetaLinux)
  - Reduced FPS (15) and input size (416) for Cortex-A9 performance

Usage on ZC702:
    python3 main_zc702.py                  # Headless, DDR3 storage
    python3 main_zc702.py --test           # Self-test, no hardware
    python3 main_zc702.py --serial /dev/ttyPS0  # Explicit port
"""

import argparse
import logging
import os
import sys
import time
import signal
import json
from datetime import datetime

# ZC702 path setup — /opt/c2s is the app root
sys.path.insert(0, "/opt/c2s")

# Import ZC702 config instead of default config
from config_zc702     import DEFAULT_CONFIG, SystemConfig
from waste_detector   import WasteDetector
from webcam_capture   import WebcamCapture
from data_fusion      import DataFusionEngine, PicoSerialReader
from database         import WasteDatabase
from report_generator import ReportGenerator
from alert_system     import AlertSystem


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(config: SystemConfig):
    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })

    file_handler    = logging.FileHandler(config.log_file)
    file_handler.setFormatter(JsonFormatter())
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=[file_handler, console_handler],
    )


logger = logging.getLogger(__name__)

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_system(config: SystemConfig, headless: bool = True, test_mode: bool = False) -> bool:
    global running

    logger.info(json.dumps({
        "event": "zc702_system_start",
        "vehicle_id": config.vehicle.vehicle_id,
        "model_path": config.detection.model_path,
        "serial_port": config.serial.port,
        "headless": headless,
        "mode": "test" if test_mode else "live",
    }))

    detector      = WasteDetector(config.detection)
    camera        = WebcamCapture(config.camera)
    serial_reader = PicoSerialReader(config.serial)
    fusion        = DataFusionEngine(config.sensors, config.vehicle)
    db            = WasteDatabase(config.database.db_path)
    alerts        = AlertSystem(config.alerts, config.sensors)

    if not detector.load_model():
        logger.error("Failed to load YOLO model")
        return False

    if not test_mode:
        if not camera.open():
            logger.warning("Camera unavailable — switching to test mode")
            test_mode = True

    try:
        pico_connected = serial_reader.connect()
    except RuntimeError as e:
        logger.critical(str(e))
        return False

    logger.info(f"Pico W: {'connected' if pico_connected else 'simulated'}")
    logger.info("ZC702 system ready — entering main loop")

    frame_id         = 0
    last_report_time = time.time()
    report_interval  = config.reports.report_interval_minutes * 60

    try:
        while running:
            loop_start = time.time()

            # 1. Capture frame
            if test_mode:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

            # 2. AI Detection
            waste_frame = detector.detect(frame, frame_id)

            # 3. Read Pico W sensors
            try:
                sensor_reading = serial_reader.read_sensors()
            except RuntimeError as e:
                logger.critical(f"Pico disconnected: {e}")
                running = False
                break

            # 4. Fuse AI + sensors
            records = fusion.fuse(waste_frame, sensor_reading)

            # 5. Store in DDR3 database
            source_tag = "simulated" if sensor_reading.is_simulated else "real"
            for record in records:
                record_dict = record.to_dict()
                db.insert_waste_record(record_dict)
                db.insert_sensor_reading({
                    "timestamp":    record.timestamp,
                    "distance_cm":  sensor_reading.distance_cm,
                    "bin_fill_pct": sensor_reading.bin_fill_pct,
                    "temperature_c": sensor_reading.temperature_c,
                    "humidity_pct": sensor_reading.humidity_pct,
                    "gas_analog":   sensor_reading.gas_analog,
                    "gas_digital":  sensor_reading.gas_digital,
                    "moisture_raw": sensor_reading.moisture_raw,
                    "data_source":  source_tag,
                })
                alerts.check_and_alert(record_dict, db)

            # 6. Periodic reports
            now = time.time()
            if now - last_report_time > report_interval:
                recent      = db.get_recent_records(500)
                daily_stats = db.get_daily_statistics()
                report_payloads = ReportGenerator(config.reports).generate_all(
                    [dict(r) for r in recent], daily_stats,
                )
                if "json" in report_payloads:
                    db.insert_json_report("waste_monitoring", report_payloads["json"])
                last_report_time = now

            frame_id += 1
            if frame_id % 30 == 0:
                stats = detector.get_statistics()
                logger.info(f"Frame {frame_id:5d} | FPS: {stats['current_fps']:4.1f} | "
                            f"Det: {len(waste_frame.detections):2d} | "
                            f"Bin: {sensor_reading.bin_fill_pct:5.1f}%")

            if test_mode and frame_id >= 10:
                logger.info("Test mode complete")
                break

            elapsed    = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / config.camera.fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"Main loop error: {e}")
        import traceback; traceback.print_exc()

    finally:
        camera.stop()
        serial_reader.close()
        logger.info("ZC702 system stopped cleanly")

    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="C2S Smart Waste System — ZC702 Zynq-7000"
    )
    parser.add_argument("--test",     action="store_true", help="Test mode")
    parser.add_argument("--headless", action="store_true", default=True, help="No display")
    parser.add_argument("--strict",   action="store_true", help="Abort if Pico not connected")
    parser.add_argument("--serial",   type=str, help="Override serial port")
    parser.add_argument("--model",    type=str, help="Override YOLO model path")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.serial:
        config.serial.port = args.serial
    if args.strict:
        config.serial.strict_sensor_mode = True
    if args.model:
        config.detection.model_path = args.model

    setup_logging(config)
    success = run_system(config, headless=True, test_mode=args.test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

