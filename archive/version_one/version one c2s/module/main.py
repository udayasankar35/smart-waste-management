#!/usr/bin/env python3
"""
Main System Entry Point
AI-Enabled Smart Waste Segregation & Monitoring System

Orchestrates the full pipeline on NVIDIA Jetson Nano:
  1. Capture frames from USB webcam
  2. Run YOLOv8 waste classification
  3. Read sensor data from Raspberry Pi Pico W (serial USB)
  4. Fuse AI + sensor data
  5. Store in SQLite database
  6. Check alerts
  7. Generate periodic reports

Usage:
    python main.py                   # Normal operation (simulation fallback if no Pico)
    python main.py --strict          # STRICT MODE — abort if Pico not connected
    python main.py --test            # Dry-run test (no hardware required)
    python main.py --headless        # No GUI window
    python main.py --serial /dev/ttyUSB0   # Override serial port
"""

import argparse
import logging
import os
import sys
import time
import signal
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config          import DEFAULT_CONFIG, SystemConfig
from waste_detector  import WasteDetector
from webcam_capture  import WebcamCapture
from data_fusion     import DataFusionEngine, PicoSerialReader
from database        import WasteDatabase
from report_generator import ReportGenerator
from alert_system    import AlertSystem


# ─── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(config: SystemConfig):
    os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            return json.dumps(payload)

    file_handler = logging.FileHandler(config.log_file)
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=[
            file_handler,
            console_handler,
        ]
    )

logger = logging.getLogger(__name__)


# ─── Graceful Shutdown ───────────────────────────────────────────────────────────

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received — finishing current frame")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def set_fullscreen(cv2, window_name: str, enabled: bool) -> None:
    """Apply fullscreen immediately so the display manager can resize the window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if enabled:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, 1920, 1080)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    cv2.waitKey(1)


def select_gallery_image() -> str | None:
    """Open the operating-system image picker without adding a dependency."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        image_path = filedialog.askopenfilename(
            title="Select a waste photo to analyse",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        root.destroy()
        return image_path or None
    except Exception as exc:
        logger.error(f"Could not open image picker: {exc}")
        return None


def inspect_gallery_image(detector, image_path: str, frame_id: int, window_name: str,
                          is_fullscreen: bool) -> tuple[bool, bool]:
    """Display material detections for one selected photo; C resumes the camera."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read gallery image: {image_path}")
        return False, is_fullscreen
    result = detector.detect(image, frame_id)
    annotated = detector.draw_detections(image, result, is_fullscreen=is_fullscreen)
    cv2.putText(annotated, "GALLERY | [C] Camera [G] Another photo [F] Fullscreen [Q] Quit",
                (14, max(70, annotated.shape[0] - 52)), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    while True:
        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("c"), ord("C"), 27):
            return False, is_fullscreen
        if key in (ord("q"), ord("Q")):
            return True, is_fullscreen
        if key in (ord("f"), ord("F")):
            is_fullscreen = not is_fullscreen
            set_fullscreen(cv2, window_name, is_fullscreen)
        if key in (ord("g"), ord("G")):
            next_path = select_gallery_image()
            if next_path:
                return inspect_gallery_image(detector, next_path, frame_id + 1,
                                             window_name, is_fullscreen)


# ─── Main Pipeline ───────────────────────────────────────────────────────────────

def run_image_mode(config: SystemConfig, image_path: str) -> bool:
    """Analyse one gallery image without requiring camera or sensor hardware."""
    import cv2
    detector = WasteDetector(config.detection)
    if not detector.load_model():
        return False
    window_name = "AI Smart Waste System | Gallery"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.camera.width, config.camera.height)
    set_fullscreen(cv2, window_name, False)
    try:
        inspect_gallery_image(detector, image_path, 0, window_name, False)
    finally:
        cv2.destroyAllWindows()
    return True


def run_system(
    config:   SystemConfig,
    headless: bool = False,
    test_mode: bool = False,
) -> bool:
    """Main system loop"""
    global running

    # ── Startup Banner ────────────────────────────────────────────────────────
    logger.info(
        json.dumps(
            {
                "event": "system_start",
                "vehicle_id": config.vehicle.vehicle_id,
                "vehicle_name": config.vehicle.vehicle_name,
                "route_id": config.vehicle.route_id,
                "model_path": config.detection.model_path,
                "serial_port": config.serial.port,
                "strict_sensor_mode": config.serial.strict_sensor_mode,
                "database": config.database.db_path,
                "mode": "test" if test_mode else "live",
                "headless": headless,
            }
        )
    )

    # ── Initialise Components ─────────────────────────────────────────────────
    detector      = WasteDetector(config.detection)
    camera        = WebcamCapture(config.camera)
    serial_reader = PicoSerialReader(config.serial)
    fusion        = DataFusionEngine(config.sensors, config.vehicle)
    db            = WasteDatabase(config.database.db_path)
    alerts        = AlertSystem(config.alerts, config.sensors)

    # ── Load AI Model ─────────────────────────────────────────────────────────
    if not detector.load_model():
        logger.error("❌ Failed to load YOLO model — aborting")
        return False

    # ── Open Camera ──────────────────────────────────────────────────────────
    if not test_mode:
        if not camera.open():
            logger.warning("⚠️  Webcam unavailable — switching to test mode")
            test_mode = True

    # ── Connect Pico W Serial ─────────────────────────────────────────────────
    # In strict mode this will raise RuntimeError and abort if Pico is absent.
    try:
        pico_connected = serial_reader.connect()
    except RuntimeError as e:
        logger.critical(str(e))
        logger.info("Connect the Pico W to USB and restart, or run without strict mode")
        return False

    if pico_connected:
        logger.info("Pico W connected; real sensor data enabled")
    else:
        logger.warning("Pico W not connected; using simulated sensor data")

    # ── Import cv2 for display (only if needed) ───────────────────────────────
    is_fullscreen = False
    WIN_NAME     = "AI Smart Waste System  |  [G] Gallery  [F] Fullscreen  [Q] Quit"
    if not headless and not test_mode:
        import cv2
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, config.camera.width, config.camera.height)
        set_fullscreen(cv2, WIN_NAME, False)

    logger.info("✅ All components initialised — entering main loop")

    frame_id          = 0
    last_report_time  = time.time()
    report_interval   = config.reports.report_interval_minutes * 60

    try:
        while running:
            loop_start = time.time()

            # ── 1. Capture frame ──────────────────────────────────────────────
            if test_mode:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    if not headless:
                        import cv2
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord("q") or key == ord("Q"):
                            break
                    time.sleep(0.05)
                    continue

            # ── 2. AI waste detection ─────────────────────────────────────────
            waste_frame = detector.detect(frame, frame_id)

            # ── 3. Read sensor data from Pico W ───────────────────────────────
            try:
                sensor_reading = serial_reader.read_sensors()
            except RuntimeError as e:
                # Strict mode: Pico disconnected mid-run
                logger.critical(f"❌ {e}")
                running = False
                break

            # ── 4. Fuse AI + sensors ──────────────────────────────────────────
            records = fusion.fuse(waste_frame, sensor_reading)

            # ── 5. Store in database ──────────────────────────────────────────
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

                # ── 6. Check alerts ───────────────────────────────────────────
                alerts.check_and_alert(record_dict, db)

            # ── 7. Display (if GUI mode) ─────────────────────────────────────────
            if not headless and not test_mode:
                # Build sensor dict for premium HUD bottom bar
                sensor_dict = {
                    "bin_fill_pct":   sensor_reading.bin_fill_pct,
                    "temperature_c":  sensor_reading.temperature_c,
                    "humidity_pct":   sensor_reading.humidity_pct,
                    "gas_analog":     sensor_reading.gas_analog,
                    "moisture_raw":   sensor_reading.moisture_raw,
                    "is_simulated":   sensor_reading.is_simulated,
                }
                annotated = detector.draw_detections(
                    frame, waste_frame,
                    sensor_data=sensor_dict,
                    is_fullscreen=is_fullscreen,
                )
                cv2.putText(annotated, "[G] Gallery photo  [F] Fullscreen  [Q] Quit",
                            (14, max(65, annotated.shape[0] - 52)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WIN_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    break
                if key == ord("f") or key == ord("F"):
                    is_fullscreen = not is_fullscreen
                    set_fullscreen(cv2, WIN_NAME, is_fullscreen)
                if key == ord("g") or key == ord("G"):
                    image_path = select_gallery_image()
                    if image_path:
                        should_quit, is_fullscreen = inspect_gallery_image(
                            detector, image_path, frame_id, WIN_NAME, is_fullscreen
                        )
                        if should_quit:
                            break

            # ── 8. Periodic JSON snapshots ────────────────────────────────────
            now = time.time()
            if now - last_report_time > report_interval:
                logger.info("Generating periodic JSON snapshot...")
                recent      = db.get_recent_records(500)
                daily_stats = db.get_daily_statistics()
                report_payloads = ReportGenerator(config.reports).generate_all(
                    [dict(r) for r in recent],
                    daily_stats,
                )
                if "json" in report_payloads:
                    db.insert_json_report("waste_monitoring", report_payloads["json"])
                last_report_time = now

            # ── Per-frame logging (every 30 frames) ──────────────────────────
            frame_id += 1
            if frame_id % 30 == 0:
                src_label = "SIM" if sensor_reading.is_simulated else "REAL"
                stats     = detector.get_statistics()
                logger.info(
                    f"Frame {frame_id:5d} | "
                    f"FPS: {stats['current_fps']:5.1f} | "
                    f"Det: {len(waste_frame.detections):2d} | "
                    f"Bin: {sensor_reading.bin_fill_pct:5.1f}% | "
                    f"Gas: {sensor_reading.gas_analog:5d} | "
                    f"Sensor: [{src_label}]"
                )

            # ── Test mode limit ───────────────────────────────────────────────
            if test_mode and frame_id >= 10:
                logger.info(f"Test mode complete after {frame_id} frames")
                break

            # Frame rate control
            elapsed    = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / config.camera.fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.stop()
        serial_reader.close()
        if not headless and not test_mode:
            cv2.destroyAllWindows()

        # Final JSON snapshot
        logger.info("Generating final JSON snapshot...")
        recent      = db.get_recent_records(500)
        daily_stats = db.get_daily_statistics()
        report_payloads = ReportGenerator(config.reports).generate_all(
            [dict(r) for r in recent],
            daily_stats,
        )
        if "json" in report_payloads:
            db.insert_json_report("waste_monitoring", report_payloads["json"])

        # Session summary
        det_stats     = detector.get_statistics()
        fusion_stats  = fusion.get_statistics()
        alert_summary = alerts.get_alert_summary()

        logger.info(
            json.dumps(
                {
                    "event": "session_summary",
                    "frames_processed": det_stats["total_frames"],
                    "avg_inference_ms": round(det_stats["avg_inference_ms"], 1),
                    "total_records": fusion_stats["total_records"],
                    "real_sensor_reads": fusion_stats.get("real_records", "N/A"),
                    "simulated_reads": fusion_stats.get("simulated_records", "N/A"),
                    "waste_distribution": fusion_stats.get("waste_distribution", {}),
                    "total_alerts": alert_summary["total_alerts"],
                    "reports": "stored_in_database",
                }
            )
        )

    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🗑️  Smart Waste System — Jetson Nano + Raspberry Pi Pico W",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         Normal operation
  python main.py --strict                Abort if Pico not connected (production)
  python main.py --test --headless       Run self-test without any hardware
  python main.py --serial /dev/ttyUSB0  Custom serial port
  python main.py --vehicle GV-07        Override vehicle ID
        """
    )
    parser.add_argument("--test",     action="store_true",
                        help="Test mode — no real hardware required")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI display window")
    parser.add_argument("--strict",   action="store_true",
                        help="Strict sensor mode — abort if Pico not connected")
    parser.add_argument("--vehicle",  type=str,
                        help="Override vehicle ID")
    parser.add_argument("--serial",   type=str,
                        help="Override serial port (e.g. /dev/ttyACM1)")
    parser.add_argument("--image",    type=str,
                        help="Analyse one saved image without opening the camera")
    parser.add_argument("--model",    type=str,
                        help="Override YOLO model weights path (.pt or .engine)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    # Apply overrides
    if args.vehicle:
        config.vehicle.vehicle_id = args.vehicle
    if args.serial:
        config.serial.port = args.serial
    if args.strict:
        config.serial.strict_sensor_mode = True
    if args.model:
        config.detection.custom_model_path = args.model
        config.detection.model_path = args.model

    setup_logging(config)
    if args.image:
        success = run_image_mode(config, args.image)
        sys.exit(0 if success else 1)
    success = run_system(config, headless=args.headless, test_mode=args.test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
