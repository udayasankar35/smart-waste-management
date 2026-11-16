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
import binascii
import threading
from datetime import datetime
from typing import Optional

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
from werkzeug.exceptions import HTTPException

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

# Reject huge uploads before they are read into RAM (phone photos are 3-12 MB).
MAX_UPLOAD_MB = 25
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def _too_large(_err):
    return jsonify({"error": f"Image too large (max {MAX_UPLOAD_MB} MB)"}), 413


@app.errorhandler(404)
def _not_found(_err):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "API endpoint not found"}), 404
    return render_template("404.html"), 404

# ─── Global Components ───────────────────────────────────────────────────────────

config = DEFAULT_CONFIG
detector: WasteDetector = None
db: WasteDatabase = None
advisor: WasteAdvisor = None
llama_engine: LlamaEngine = None
camera = None            # kept for compatibility; the shared camera lives in _camera_hub
frame_counter = 0
_infer_lock = threading.Lock()   # YOLO models are not thread-safe: one inference at a time


def init_components(model_path: str = None):
    """Initialize AI model, database, Llama engine, and chatbot."""
    global detector, db, advisor, llama_engine

    logger.info("🚀 Initialising components...")

    # AI Detector
    det_config = DetectionConfig()
    if model_path and not os.path.isfile(model_path):
        # Without this check a typo silently falls back to the generic COCO model.
        logger.error(f"❌ --model file not found: {model_path}")
        sys.exit(1)
    if model_path:
        det_config.custom_model_path = model_path
        det_config.model_path = model_path
    det_config.device = "cpu"
    detector = WasteDetector(det_config)
    if not detector.load_model():
        logger.error("❌ Failed to load YOLO model")
        sys.exit(1)
    logger.info(f"✅ YOLO model loaded (taxonomy={getattr(detector, 'taxonomy', 'n/a')})")

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


@app.route("/favicon.ico")
def favicon():
    """Serve favicon.ico directly to prevent 404 console errors."""
    return send_from_directory(
        app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )


@app.route("/robots.txt")
def robots_txt():
    """Search engine and AI crawler rules."""
    return send_from_directory(
        app.static_folder, "robots.txt", mimetype="text/plain"
    )


@app.route("/sitemap.xml")
def sitemap_xml():
    """XML sitemap for SEO."""
    return send_from_directory(
        app.static_folder, "sitemap.xml", mimetype="application/xml"
    )


@app.route("/llms.txt")
def llms_txt():
    """Standard markdown context for LLM crawlers."""
    return send_from_directory(
        app.static_folder, "llms.txt", mimetype="text/plain"
    )


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
            data = request.get_json(silent=True)
            if isinstance(data, dict) and isinstance(data.get("image"), str):
                # Remove data URL prefix if present
                img_data = data["image"]
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(img_data)
                except (binascii.Error, ValueError):
                    return jsonify({"error": "Image is not valid base64"}), 400
                file_bytes = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "No valid image provided"}), 400

        ensure_components()
        if detector is None:
            return jsonify({"error": "Detector not initialized"}), 500

        # Run detection (one inference at a time)
        with _infer_lock:
            frame_counter += 1
            waste_frame = detector.detect(image, frame_counter)

        # Build response
        detections = []
        for det in waste_frame.detections:
            det_dict = det.to_dict()
            detections.append(det_dict)

        # Get annotated image as base64
        with _infer_lock:
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

        # If no detections, store a no-detection record and clear previous advisor context
        if not waste_frame.detections:
            active_db.insert_waste_record({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "waste_type": "no_detection",
                "ai_confidence": 0.0,
                "ai_class": "",
                "vehicle_id": "WEB-UI",
            })
            if advisor:
                advisor.set_last_detection(None)

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

    except HTTPException:
        raise                       # let Flask answer 413 etc. instead of turning it into a 500
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


class _CameraHub:
    """
    ONE shared camera reader for every stream / capture / status request.

    Before: each request opened its own cv2.VideoCapture(0). A second browser
    tab, or the Capture button while the live stream was running, fought for the
    same device: the second open failed or the first stream froze.

    Now: the first viewer starts a background reader thread, everybody reads the
    newest frame from it, and the camera is released a few seconds after the last
    viewer leaves.
    """

    IDLE_RELEASE_SEC = 4.0      # keep the camera open this long after the last viewer
    WARMUP_FRAMES = 5           # discarded after opening (auto-exposure settles)

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._cond = threading.Condition()
        self._clients = 0
        self._thread = None
        self._frame = None
        self._frame_id = 0          # only ever grows, even after a restart
        self._failed = False        # last attempt to open the camera failed
        self._last_client_ts = 0.0

    @property
    def failed(self) -> bool:
        return self._failed

    def acquire(self) -> None:
        """Register a viewer; starts the reader thread if it is not running."""
        with self._cond:
            self._clients += 1
            self._last_client_ts = time.time()
            if self._thread is None or not self._thread.is_alive():
                self._failed = False
                self._thread = threading.Thread(
                    target=self._run, name="camera-hub", daemon=True
                )
                self._thread.start()

    def release(self) -> None:
        with self._cond:
            self._clients = max(0, self._clients - 1)
            self._last_client_ts = time.time()

    def wait_frame(self, after_id: int = -1, timeout: float = 2.0):
        """
        Wait for a frame newer than `after_id`.
        Returns (frame_id, frame), or (after_id, None) on timeout / camera failure.
        """
        deadline = time.time() + timeout
        with self._cond:
            while not self._failed and (
                self._frame is None or self._frame_id <= after_id
            ):
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if self._frame is not None and self._frame_id > after_id:
                return self._frame_id, self._frame
            return after_id, None

    # ── reader thread ──────────────────────────────────────────────────────
    def _run(self) -> None:
        while True:
            cap = _open_camera(self.device_index)
            if cap is None or not cap.isOpened():
                logger.warning("Webcam not available")
                with self._cond:
                    self._failed = True
                    self._thread = None
                    self._cond.notify_all()
                return

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
                for _ in range(self.WARMUP_FRAMES):
                    cap.read()

                while True:
                    with self._cond:
                        if (
                            self._clients == 0
                            and time.time() - self._last_client_ts > self.IDLE_RELEASE_SEC
                        ):
                            break
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue
                    with self._cond:
                        self._frame = frame
                        self._frame_id += 1
                        self._cond.notify_all()
            finally:
                cap.release()

            with self._cond:
                if self._clients > 0:
                    # someone arrived while we were shutting down: open it again
                    continue
                self._frame = None
                self._thread = None
                self._cond.notify_all()
                return


_camera_hub = _CameraHub(0)
_stream_cache = {"fid": -1, "jpeg": None}


def _grab_frame(timeout: float = 4.0):
    """Newest camera frame, or None if the camera is unavailable."""
    _camera_hub.acquire()
    try:
        _, frame = _camera_hub.wait_frame(-1, timeout=timeout)
        return frame
    finally:
        _camera_hub.release()


def _stream_jpeg(frame, fid: int):
    """
    Annotate (every 3rd frame, to save CPU) and JPEG-encode a frame ONCE.
    Every viewer of the stream receives the same bytes, so ten open tabs cost
    the same as one.
    """
    with _infer_lock:
        if _stream_cache["fid"] != fid:
            out = frame
            if fid % 3 == 0 and detector is not None:
                waste_frame = detector.detect(frame, fid)
                out = detector.draw_detections(frame, waste_frame)
            ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _stream_cache["fid"] = fid
            _stream_cache["jpeg"] = buffer.tobytes() if ok else None
        return _stream_cache["jpeg"]


def generate_frames():
    """Generator for webcam MJPEG stream with AI overlay (shared camera)."""
    _camera_hub.acquire()
    try:
        last_id = -1
        waited = 0
        while True:
            fid, frame = _camera_hub.wait_frame(last_id, timeout=2.0)

            if frame is None:
                if _camera_hub.failed:
                    # Send a placeholder frame, and retry the camera every ~5 s
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Webcam not available", (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)
                    _, buffer = cv2.imencode(".jpg", placeholder)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buffer.tobytes() + b"\r\n")
                    time.sleep(1)
                    waited += 1
                    if waited % 5 == 0:
                        _camera_hub.release()
                        _camera_hub.acquire()
                continue

            waited = 0
            last_id = fid
            jpeg = _stream_jpeg(frame, fid)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    finally:
        _camera_hub.release()


@app.route("/api/webcam/stream")
def webcam_stream():
    """MJPEG stream of webcam with AI overlay."""
    ensure_components()
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/webcam/capture", methods=["POST"])
def webcam_capture():
    """Capture a single frame from webcam and run full detection."""
    global frame_counter

    try:
        ensure_components()

        frame = _grab_frame(timeout=4.0)
        if frame is None:
            return jsonify({"error": "Webcam not available"}), 503

        with _infer_lock:
            frame_counter += 1
            waste_frame = detector.detect(frame, frame_counter)
            annotated = detector.draw_detections(frame, waste_frame)

        detections = [d.to_dict() for d in waste_frame.detections]

        # Get annotated image
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        # Store in DB
        active_db = ensure_db()
        for det in waste_frame.detections:
            active_db.insert_waste_record({
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
        return jsonify({"available": _grab_frame(timeout=3.0) is not None})
    except Exception:
        return jsonify({"available": False})


# ─── Routes: Model info ──────────────────────────────────────────────────────────


@app.route("/api/model/info")
def model_info():
    """Which weights are loaded and which class set they use (confirms best_v2.pt)."""
    if detector is None:
        return jsonify({"success": False, "status": "Detector not initialised"})
    return jsonify({"success": True, **detector.get_statistics()})


# ─── Main ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌐 Smart Waste Web Interface")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address (default: this computer only). "
                             "Use 0.0.0.0 to open the site from a phone / another PC on the same Wi-Fi")
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
