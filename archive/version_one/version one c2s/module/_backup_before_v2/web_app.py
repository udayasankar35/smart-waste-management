#!/usr/bin/env python3
"""
Web Application — Flask Backend
AI-Powered Smart Waste Classification & Disposal Assistant

Full-stack web interface for testing the waste detection system:
  - Webcam live feed with real-time AI detection
  - Image gallery upload and analysis
  - Database dashboard with statistics
  - AI chatbot for waste disposal advisory

Bharat AI-SoC Challenge 2026–27 — PS2

Usage:
    python web_app.py                  # Start on http://localhost:5000
    python web_app.py --port 8080      # Custom port
    python web_app.py --debug          # Debug mode
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
import io
from datetime import datetime

# ── Fix Windows console handle crash with Click / Flask ─────────────────────────
# Under Windows (especially Python 3.14+ or when native C-libraries manipulate
# low-level stdout/stderr handles), click._winconsole caches Windows console
# handles that can become invalid, causing WriteConsoleW to fail with
# `OSError: Windows error: 6` (ERROR_INVALID_HANDLE).
# Bypassing _winconsole routes output through standard Python sys.stdout / sys.stderr,
# which natively support Unicode safely without crashing.
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
        import click._winconsole
        import click._compat
        click._winconsole._get_windows_console_stream = lambda *args, **kwargs: None
        click._compat._get_windows_console_stream = lambda *args, **kwargs: None
    except Exception:
        pass

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# ── Add parent directory to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_CONFIG, SystemConfig, DetectionConfig
from waste_detector import WasteDetector
from database import WasteDatabase
from waste_advisor import WasteAdvisor
from llama_engine import LlamaEngine

# ─── Logging ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Flask App ───────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "web", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "web", "static"),
)

# ─── Global Components ───────────────────────────────────────────────────────────

config = DEFAULT_CONFIG
detector: WasteDetector = None
db: WasteDatabase = None
advisor: WasteAdvisor = None
llama_engine: LlamaEngine = None
camera = None
frame_counter = 0


def init_components(model_path: str = None):
    """Initialize AI model, database, Llama engine, and chatbot."""
    global detector, db, advisor, llama_engine

    logger.info("🚀 Initialising components...")

    # AI Detector
    det_config = DetectionConfig()
    if model_path:
        det_config.custom_model_path = model_path
        det_config.model_path = model_path
    det_config.device = "cpu"
    detector = WasteDetector(det_config)
    if not detector.load_model():
        logger.error("❌ Failed to load YOLO model")
        sys.exit(1)
    logger.info("✅ YOLO model loaded")

    # Database
    db_path = os.path.join(os.path.dirname(__file__), "database", "waste_monitoring.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = WasteDatabase(db_path)
    logger.info("✅ Database connected")

    # Llama 3.2 Local LLM Engine
    llama_engine = LlamaEngine(config=config.llama)
    llama_engine.initialize()
    logger.info(f"✅ Llama Engine: {llama_engine.load_status}")

    # Chatbot (rule-based advisor — used alongside Llama)
    advisor = WasteAdvisor(llama_engine=llama_engine)
    logger.info("✅ Waste Advisor ready")


def ensure_db() -> WasteDatabase:
    """Ensure database connection is active."""
    global db
    if db is None:
        db_path = os.path.join(os.path.dirname(__file__), "database", "waste_monitoring.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db = WasteDatabase(db_path)
    return db


def ensure_components():
    """Ensure all core subsystems are initialized on first access."""
    global detector, db, advisor, llama_engine
    if detector is None or db is None or advisor is None:
        init_components()


# ─── Routes: Pages ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Main web interface."""
    return render_template("index.html")


# ─── Routes: AI Detection API ────────────────────────────────────────────────────


@app.route("/api/detect", methods=["POST"])
def detect_waste():
    """
    Detect waste in an uploaded image.
    Accepts: multipart/form-data with 'image' file
             OR JSON with 'image' as base64 string
    """
    global frame_counter

    try:
        image = None

        # Handle file upload
        if "image" in request.files:
            file = request.files["image"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Handle base64 image (from webcam capture)
        elif request.is_json:
            data = request.get_json()
            if "image" in data:
                # Remove data URL prefix if present
                img_data = data["image"]
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                file_bytes = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "No valid image provided"}), 400

        ensure_components()
        if detector is None:
            return jsonify({"error": "Detector not initialized"}), 500

        # Run detection
        frame_counter += 1
        waste_frame = detector.detect(image, frame_counter)

        # Build response
        detections = []
        for det in waste_frame.detections:
            det_dict = det.to_dict()
            detections.append(det_dict)

        # Get annotated image as base64
        annotated = detector.draw_detections(image, waste_frame)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        # Store in database
        active_db = ensure_db()
        for det in waste_frame.detections:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "waste_type": det.class_name,
                "ai_confidence": det.confidence,
                "ai_class": det.class_name,
                "humidity": -1.0,
                "temperature": -1.0,
                "gas_level": 0,
                "moisture_level": 0,
                "bin_fill_level": -1.0,
                "vehicle_id": "WEB-UI",
            }
            active_db.insert_waste_record(record)

        # If no detections, store a no-detection record
        if not waste_frame.detections:
            active_db.insert_waste_record({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "waste_type": "no_detection",
                "ai_confidence": 0.0,
                "ai_class": "",
                "vehicle_id": "WEB-UI",
            })

        # Get advisory and object guess for the top detection
        advice = None
        analysis = None
        if detections:
            top = detections[0]
            detected_obj = top.get("detected_object") or ""
            analysis = advisor.guess_object_and_analysis(
                top["class_name"],
                top["confidence"],
                top.get("bbox"),
                detected_object=detected_obj,
                image=image,
            )
            advice_result = advisor.get_detection_advice(
                top["class_name"],
                top["confidence"],
                top.get("bbox"),
                detected_object=detected_obj,
                image=image,
            )
            advice = advice_result["message"]

            all_det_list = [
                {
                    "object": d.get("detected_object") or d.get("class_name", ""),
                    "detected_object": d.get("detected_object") or d.get("class_name", ""),
                    "class_name": d.get("class_name", ""),
                    "confidence": float(d.get("confidence", 0.0)),
                    "bbox": d.get("bbox"),
                }
                for d in detections
            ]
            det_summary_str = f"{len(detections)} object(s) captured in picture: " + ", ".join([
                f"{d.get('detected_object') or d.get('class_name')} ({float(d.get('confidence', 0.0))*100:.1f}%)"
                for d in detections
            ])

            advisor.set_last_detection({
                "waste_type": top["class_name"],
                "confidence": top["confidence"],
                "bbox": top.get("bbox"),
                "detected_object": detected_obj,
                "object_guess": analysis["guessed_object"],
                "plastic_type": analysis["plastic_type"],
                "recycling_steps": analysis["recycling_steps"],
                "household_reuse": analysis["household_reuse"],
                "decomposition_time": analysis["decomposition_time"],
                "num_detections": len(detections),
                "all_detections": all_det_list,
                "detections_summary": det_summary_str,
                "has_picture": True,
            })

        return jsonify({
            "success": True,
            "frame_id": waste_frame.frame_id,
            "num_detections": len(detections),
            "inference_ms": round(waste_frame.inference_time_ms, 1),
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "advice": advice,
            "object_guess": analysis["guessed_object"] if analysis else None,
            "plastic_type": analysis["plastic_type"] if analysis else None,
            "recycling_steps": analysis["recycling_steps"] if analysis else [],
            "household_reuse": analysis["household_reuse"] if analysis else [],
            "decomposition_time": analysis["decomposition_time"] if analysis else None,
            "summary": waste_frame.get_summary(),
        })

    except Exception as e:
        logger.error(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── Routes: Chatbot API ─────────────────────────────────────────────────────────


@app.route("/api/chat", methods=["POST"])
def chat():
    """AI chatbot endpoint — hybrid routing with WasteAdvisor + Llama 3.2."""
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        detection_context = data.get("detection_context")
        force_llama = data.get("force_llama", False)

        if not message:
            return jsonify({"error": "Empty message"}), 400

        ensure_components()

        # Build detection context from last advisor detection if not provided
        ctx = detection_context
        if not ctx and advisor and advisor.last_detection:
            ctx = advisor.last_detection
        elif ctx and advisor and advisor.last_detection:
            merged_ctx = dict(advisor.last_detection)
            merged_ctx.update(ctx)
            ctx = merged_ctx

        # If user explicitly requested generative Llama inference
        if force_llama and llama_engine and llama_engine.active_backend in ("llama-cpp-gguf", "executorch-pte"):
            llama_resp = llama_engine.generate(message, detection_context=ctx)
            return jsonify({
                "success": True,
                "response": llama_resp["text"],
                "type": "llama",
                "engine": llama_resp["engine"],
                "backend": llama_resp["backend"],
                "inference_ms": llama_resp["inference_time_ms"],
                "is_fallback": llama_resp["is_fallback"],
                "waste_info": None,
            })

        # 1. First consult expert WasteAdvisor for instant, accurate domain knowledge (<5ms)
        advisor_resp = None
        if advisor:
            advisor_resp = advisor.get_response(message, detection_context=ctx)

        # If advisor has a concrete response (greeting, disposal, recycling, reuse, resin codes, etc.)
        if advisor_resp and advisor_resp.get("type") not in ("clarification", "fallback"):
            return jsonify({
                "success": True,
                "response": advisor_resp["message"],
                "type": advisor_resp["type"],
                "waste_info": advisor_resp.get("waste_info"),
                "engine": "WasteAdvisor Knowledge Engine",
                "inference_ms": 2.5,
            })

        # 2. For open-ended questions where advisor needed fallback, try active Llama 3.2 local model
        if llama_engine and llama_engine.active_backend in ("llama-cpp-gguf", "executorch-pte"):
            try:
                llama_resp = llama_engine.generate(message, detection_context=ctx)
                if llama_resp.get("text"):
                    return jsonify({
                        "success": True,
                        "response": llama_resp["text"],
                        "type": "llama",
                        "engine": llama_resp["engine"],
                        "backend": llama_resp["backend"],
                        "inference_ms": llama_resp["inference_time_ms"],
                        "is_fallback": llama_resp["is_fallback"],
                        "waste_info": None,
                    })
            except Exception as le:
                logger.warning(f"Llama generation fallback: {le}")

        # 3. Return advisor response (including smart fallback guidance)
        if advisor_resp:
            return jsonify({
                "success": True,
                "response": advisor_resp["message"],
                "type": advisor_resp["type"],
                "waste_info": advisor_resp.get("waste_info"),
                "engine": "WasteAdvisor Knowledge Engine",
            })

        return jsonify({
            "success": True,
            "response": "I am here to help you segregate, recycle, and reuse waste! Ask me about any item, or upload a photo in the Detect tab.",
            "type": "help",
            "waste_info": None,
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/llama/status")
def llama_status():
    """Get Llama engine status for dashboard diagnostics."""
    if llama_engine:
        return jsonify({"success": True, **llama_engine.get_info()})
    return jsonify({"success": False, "status": "Engine not initialised"})


# ─── Routes: Database API ────────────────────────────────────────────────────────


@app.route("/api/stats")
def get_stats():
    """Get dashboard statistics."""
    try:
        active_db = ensure_db()
        total = active_db.get_total_records()
        distribution_raw = active_db.get_waste_distribution()  # Dict[str, int]
        daily_raw = active_db.get_daily_statistics()            # Dict for today
        recent_alerts = active_db.get_recent_alerts(10)

        # Transform distribution dict → list of {waste_type, count}
        distribution = [
            {"waste_type": wt, "count": cnt}
            for wt, cnt in distribution_raw.items()
        ]

        # Transform daily stats → list with today's entry
        today = time.strftime("%Y-%m-%d")
        daily_stats = [{"date": today, "count": daily_raw.get("total", 0)}]

        return jsonify({
            "success": True,
            "total_records": total,
            "waste_distribution": distribution,
            "daily_stats": daily_stats,
            "recent_alerts": recent_alerts,
        })

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/records")
def get_records():
    """Get recent waste records."""
    try:
        active_db = ensure_db()
        limit = request.args.get("limit", 50, type=int)
        waste_type = request.args.get("type", None)

        if waste_type:
            records = active_db.get_records_by_type(waste_type, limit)
        else:
            records = active_db.get_recent_records(limit)

        return jsonify({
            "success": True,
            "count": len(records),
            "records": records,
        })

    except Exception as e:
        logger.error(f"Records error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sensors")
def get_sensors():
    """Get recent sensor readings."""
    try:
        active_db = ensure_db()
        limit = request.args.get("limit", 50, type=int)
        readings = active_db.get_recent_sensor_readings(limit)
        return jsonify({
            "success": True,
            "count": len(readings),
            "readings": readings,
        })
    except Exception as e:
        logger.error(f"Sensor error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts")
def get_alerts():
    """Get recent alerts."""
    try:
        active_db = ensure_db()
        limit = request.args.get("limit", 50, type=int)
        alerts = active_db.get_recent_alerts(limit)
        return jsonify({
            "success": True,
            "count": len(alerts),
            "alerts": alerts,
        })
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Routes: Webcam Stream ───────────────────────────────────────────────────────


def _open_camera(device_index: int = 0) -> Optional[cv2.VideoCapture]:
    """Helper to open camera with DirectShow fallback on Windows."""
    try:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                return cap
        cap = cv2.VideoCapture(device_index)
        if cap and cap.isOpened():
            return cap
    except Exception as e:
        logger.error(f"Error opening camera {device_index}: {e}")
    return None


def generate_frames():
    """Generator for webcam MJPEG stream with AI overlay."""
    global camera, frame_counter
    camera = _open_camera(0)

    if not camera or not camera.isOpened():
        logger.warning("Webcam not available")
        # Send a placeholder frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Webcam not available", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)
        _, buffer = cv2.imencode(".jpg", placeholder)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1)
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_counter += 1

            # Run detection every 3rd frame to save CPU
            if frame_counter % 3 == 0:
                waste_frame = detector.detect(frame, frame_counter)
                annotated = detector.draw_detections(frame, waste_frame)
            else:
                annotated = frame

            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    finally:
        if camera:
            camera.release()


@app.route("/api/webcam/stream")
def webcam_stream():
    """MJPEG stream of webcam with AI overlay."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/webcam/capture", methods=["POST"])
def webcam_capture():
    """Capture a single frame from webcam and run full detection."""
    global camera, frame_counter

    try:
        cap = _open_camera(0)
        if not cap or not cap.isOpened():
            return jsonify({"error": "Webcam not available"}), 503

        # Read a few frames to stabilize
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({"error": "Failed to capture frame"}), 500

        frame_counter += 1
        waste_frame = detector.detect(frame, frame_counter)

        detections = [d.to_dict() for d in waste_frame.detections]

        # Get annotated image
        annotated = detector.draw_detections(frame, waste_frame)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        # Store in DB
        for det in waste_frame.detections:
            db.insert_waste_record({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "waste_type": det.class_name,
                "ai_confidence": det.confidence,
                "ai_class": det.class_name,
                "vehicle_id": "WEB-CAM",
            })

        # Get advice and object estimation
        advice = None
        analysis = None
        if detections:
            top = detections[0]
            detected_obj = top.get("detected_object") or ""
            analysis = advisor.guess_object_and_analysis(
                top["class_name"],
                top["confidence"],
                top.get("bbox"),
                detected_object=detected_obj,
                image=frame,
            )
            advice_result = advisor.get_detection_advice(
                top["class_name"],
                top["confidence"],
                top.get("bbox"),
                detected_object=detected_obj,
                image=frame,
            )
            advice = advice_result["message"]
            advisor.set_last_detection({
                "waste_type": top["class_name"],
                "confidence": top["confidence"],
                "bbox": top.get("bbox"),
                "detected_object": detected_obj,
                "object_guess": analysis["guessed_object"],
                "plastic_type": analysis["plastic_type"],
                "recycling_steps": analysis["recycling_steps"],
                "household_reuse": analysis["household_reuse"],
                "decomposition_time": analysis["decomposition_time"],
            })

        return jsonify({
            "success": True,
            "num_detections": len(detections),
            "inference_ms": round(waste_frame.inference_time_ms, 1),
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "advice": advice,
            "object_guess": analysis["guessed_object"] if analysis else None,
            "plastic_type": analysis["plastic_type"] if analysis else None,
            "recycling_steps": analysis["recycling_steps"] if analysis else [],
            "household_reuse": analysis["household_reuse"] if analysis else [],
            "decomposition_time": analysis["decomposition_time"] if analysis else None,
        })

    except Exception as e:
        logger.error(f"Capture error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/webcam/status")
def webcam_status():
    """Check if webcam is available."""
    try:
        cap = _open_camera(0)
        available = cap is not None and cap.isOpened()
        if available:
            ret, _ = cap.read()
            available = ret
            cap.release()
        return jsonify({"available": available})
    except Exception:
        return jsonify({"available": False})


# ─── Main ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌐 Smart Waste Web Interface")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--model", type=str, default=None, help="Override YOLO model weights path (.pt)")
    args = parser.parse_args()

    init_components(args.model)

    # Ensure Click console stream patch is active before Flask starts
    if sys.platform == "win32":
        try:
            import click._winconsole
            import click._compat
            click._winconsole._get_windows_console_stream = lambda *a, **k: None
            click._compat._get_windows_console_stream = lambda *a, **k: None
        except Exception:
            pass

    logger.info(f"🌐 Starting web server on http://localhost:{args.port}")
    logger.info("   Open your browser to access the interface.")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("🛑 Web server stopped by user (Ctrl+C).")
    except Exception as e:
        logger.exception(f"💥 Web server encountered an error: {e}")
