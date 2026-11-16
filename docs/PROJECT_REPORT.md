# C2S Smart Waste Classification & Disposal System — Project Audit Report

## A. Header

- **Project Root Path:** `c:\Users\kumar\OneDrive\C2S`
- **Operating System:** Microsoft Windows 11 Pro / Enterprise (Build 10.0.26300 AMD64)
- **Python Version (Active System / Virtualenv):** Python 3.14.7 (64-bit AMD64)
- **Node.js Version:** v24.19.0
- **Audit Date:** 2026-10-13
- **Git Status:** Not a Git repository (`fatal: not a git repository (or any of the parent directories): .git`). No Git commits, branches, or tracking metadata exist in the workspace root or subdirectories.
- **My Hardware Note:**
  - `BOARD / SoC: not provided   RAM: not provided GB   Accelerator (GPU/NPU/none): not provided   OS: not provided   Camera: not provided`

---

## B. What This Project Is

This project is an AI-enabled smart waste segregation, classification, and monitoring system created for the **Bharat AI-SoC Challenge 2026–27 (Problem Statement 2: AI-Powered Waste Classification & Disposal Assistant)**, organised by Arm, MeitY C2S Programme, and IIT Delhi (`version one c2s/module/README.md:1-26`). It combines YOLOv8 edge computer vision for material classification (plastics, glass, metal, paper, organic) with IoT sensor telemetry collected from a Raspberry Pi Pico W via MicroPython firmware (measuring ultrasonic bin fill percentage, DHT11 temperature/humidity, MQ gas ADC levels, and HW-103 soil moisture). Sensor and vision inputs are merged in a multi-sensor data fusion engine and stored in an on-device SQLite database that drives safety alerts and automated reports (`version one c2s/module/data_fusion.py:45-80`, `version one c2s/module/alert_system.py:40-65`). For user disposal guidance and environmental education, the application integrates an on-device Llama 3.2 1B Instruct model (executed locally via `llama-cpp-python` GGUF and ExecuTorch stubs) backed by a 2,300-line expert `WasteAdvisor` knowledge engine (`version one c2s/module/llama_engine.py:47-70`, `version one c2s/module/waste_advisor.py:65-80`). The system features two execution surfaces: a real-time OpenCV desktop application with HUD overlays (`version one c2s/module/main.py:172-205`) and a full-stack Flask web application providing live MJPEG webcam streaming, single-image detection analysis, interactive analytics, and conversational waste advice (`version one c2s/module/web_app.py:81-85`).

---

## C. File and Folder Map

| Path | Type | Size | What it is and what it is meant to do | Evidence | Connects to |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `version one c2s/module/web_app.py` | app code | 27,300 B | Primary Flask web server hosting REST APIs for vision detection, Llama/WasteAdvisor chatbot, live webcam streaming, and telemetry. | `web_app.py:81-85` | Imports `waste_detector.py`, `database.py`, `waste_advisor.py`, `llama_engine.py`, `config.py`; serves `index.html`, `app.js`, `style.css`. |
| `version one c2s/module/main.py` | app code | 20,867 B | Desktop OpenCV monitoring application running the camera feed, HUD rendering, serial sensor listener, and gallery inspector. | `main.py:172-205` | Imports `waste_detector.py`, `webcam_capture.py`, `data_fusion.py`, `database.py`, `alert_system.py`, `config.py`. |
| `version one c2s/module/waste_detector.py` | app code | 50,842 B | Dual-model YOLO detector orchestrating custom multi-material YOLOv8s with general YOLOv8n, candidate priority sorting, and HUD graphics. | `waste_detector.py:143-152` | Loaded by `web_app.py`, `main.py`, `detect_plastic.py`; loads `best.pt`, `weights.pt`, `yolov8n.pt`. |
| `version one c2s/module/waste_advisor.py` | app code | 132,422 B | Rule-based domain knowledge base delivering instant (<5ms) advice on recycling bins, resin codes, decomposition, reuse, and composting. | `waste_advisor.py:65-80` | Imported by `web_app.py`, `executorch_llama_engine.py`; wraps `llama_engine.py`. |
| `version one c2s/module/llama_engine.py` | app code | 27,395 B | On-device Llama 3.2 1B inference engine managing GGUF (`llama-cpp-python`) and ExecuTorch runtimes with structured prompt formatting. | `llama_engine.py:47-70` | Imported by `web_app.py`, `waste_advisor.py`; loads `llama-3.2-1b-instruct-q4_k_m.gguf`. |
| `version one c2s/module/executorch_llama_engine.py` | app code | 9,794 B | ExecuTorch runtime wrapper and standalone fallback for Llama 3.2 targeting Arm SoC challenge requirements. | `executorch_llama_engine.py:40-60` | Standalone test target; interfaces with `waste_advisor.py`. |
| `version one c2s/module/data_fusion.py` | app code | 20,566 B | Multi-sensor fusion engine parsing Pico W serial UART packets and correlating visual detections with moisture and gas states. | `data_fusion.py:45-80` | Imported by `main.py`; feeds `database.py` and `alert_system.py`. |
| `version one c2s/module/database.py` | app code | 14,193 B | SQLite database abstraction layer managing tables for detections, sensor logs, hardware alerts, and reports. | `database.py:32-60` | Imported by `web_app.py`, `main.py`, `report_generator.py`; reads/writes `waste_monitoring.db`. |
| `version one c2s/module/alert_system.py` | app code | 9,533 B | Automated alert evaluator detecting toxic gas surges, bin overflow levels, and hazardous waste items. | `alert_system.py:40-65` | Imported by `main.py`; logs alerts into SQLite via `database.py`. |
| `version one c2s/module/webcam_capture.py` | app code | 6,775 B | OpenCV camera wrapper supporting reconnect routines, frame warmup, resolution configuration, and buffer flushing. | `webcam_capture.py:30-55` | Imported by `main.py`, `web_app.py`. |
| `version one c2s/module/detect_plastic.py` | app code | 7,112 B | CLI utility for running batch inference on individual images or folders with annotated image outputs. | `detect_plastic.py:25-45` | Imports `waste_detector.py`, `config.py`. |
| `version one c2s/module/train_waste_model.py` | app code | 9,686 B | Ultralytics YOLOv8 training pipeline script with hyperparameter configurations for custom waste datasets. | `train_waste_model.py:25-50` | Uses `ultralytics.YOLO`; consumes dataset `data.yaml`. |
| `version one c2s/module/download_model.py` | app code | 5,906 B | Automated downloader pulling Llama 3.2 1B Instruct GGUF weights directly from Hugging Face Hub. | `download_model.py:20-40` | Writes to `models/llama-3.2-1b-instruct-q4_k_m.gguf`. |
| `version one c2s/module/report_generator.py` | app code | 3,505 B | Automated reporting utility producing periodic JSON, CSV, and ReportLab PDF summaries of waste metrics. | `report_generator.py:20-40` | Imports `database.py`, `config.py`; saves to `reports/`. |
| `version one c2s/module/config.py` | config | 11,735 B | Central configuration dataclasses defining pin numbers, sensor thresholds, detection models, Llama settings, and database paths. | `config.py:15-38` | Imported by all primary modules in `version one c2s/module`. |
| `version one c2s/module/requirements.txt` | config | 1,637 B | Python dependency manifest specifying minimum required library versions including Flask and llama-cpp-python. | `requirements.txt:1-36` | Installed via `pip`. |
| `version one c2s/module/README.md` | docs | 16,424 B | Comprehensive documentation outlining the Bharat AI-SoC Challenge PS2 problem statement, system architecture, and setup instructions. | `README.md:1-50` | Standalone documentation. |
| `version one c2s/module/pico_sensors/main.py` | app code | 9,661 B | MicroPython firmware executed on Raspberry Pi Pico W reading ultrasonic, DHT11, MQ gas, and moisture sensors and streaming JSON over UART. | `pico_sensors/main.py:20-50` | Streams JSON lines over USB serial to `data_fusion.py`. |
| `version one c2s/module/web/templates/index.html` | app code | 82,544 B | Single-page HTML5 web dashboard structure containing tabs for Detect, Live Webcam, Chatbot, Analytics, and Records. | `index.html:1-100` | Rendered by `web_app.py:152`; loads `style.css` and `app.js`. |
| `version one c2s/module/web/static/css/style.css` | app code | 65,895 B | Custom Vanilla CSS styling defining dark-mode aesthetic, HUD badges, layout grids, modal windows, and responsive containers. | `style.css:1-50` | Linked in `index.html`. |
| `version one c2s/module/web/static/js/app.js` | app code | 54,402 B | Frontend client JavaScript logic managing camera streams, image file uploads, chat requests, and dynamic DOM rendering. | `app.js:1-50` | Loaded in `index.html`; communicates with backend `/api/*` endpoints. |
| `version one c2s/module/web/static/images/` | other | 1,010,482 B | Static web media assets (7 WebP files: `hero-bottles.webp`, `plastic-bottles-dark.webp`, `sorting-bins.webp`, etc.). | (name only) | Displayed as static illustrations in `index.html`. |
| `version one c2s/module/yolo/weights/best.pt` | model | 22,510,826 B | Trained YOLOv8s PyTorch weights detecting 5 classes (`plastic_bottle`, `glass_bottle`, `metal_can`, `paper_cardboard`, `organic_waste`). | `waste_detector.py:295` | Loaded by `WasteDetector.load_model()`. |
| `version one c2s/module/yolo/weights/weights.pt` | model | 20,342,981 B | Earlier trained YOLOv8s model detecting single class `bottle`. | `waste_detector.py:302` | Fallback model in `WasteDetector`. |
| `version one c2s/module/yolov8n.pt` | model | 6,549,796 B | Standard Ultralytics pretrained COCO 80-class YOLOv8n model utilized for general object and secondary plastic detection. | `waste_detector.py:314` | Loaded by `WasteDetector` as `general_model`. |
| `version one c2s/module/models/llama-3.2-1b-instruct-q4_k_m.gguf` | model | 807,694,464 B | 4-bit quantized GGUF weights of Llama 3.2 1B Instruct for local on-device generative text reasoning. | `llama_engine.py:139` | Loaded by `LlamaEngine._try_load_gguf()`. |
| `version one c2s/module/database/waste_monitoring.db` | database | 7,311,360 B | Active SQLite database containing 9,811 waste records, 9,599 sensor logs, 112 alerts, and 35 JSON reports. | `database.py:35` | Read/written by `web_app.py`, `main.py`, `database.py`. |
| `version one c2s/module/database/README.md` | docs | 231 B | Brief note describing SQLite database initialization and storage policies. | `README.md:1-10` | Standalone documentation. |
| `version one c2s/module/data/waste_monitoring.db` | database | 36,864 B | Default seed/test SQLite database (30 records, 30 sensor logs, 8 alerts). | `config.py:139` | Referenced when relative path `database/waste_monitoring.db` resolves to `./data`. |
| `version one c2s/module/tests/test_executorch_llama.py` | app code | 1,678 B | Unit test validating initialization, prompt creation, and generation for ExecuTorch Llama engine. | `test_executorch_llama.py:1-40` | Tests `executorch_llama_engine.py`. |
| `version one c2s/module/tests/test_material_mapping.py` | app code | 529 B | Unit test validating material name normalization in `WasteDetector`. | `test_material_mapping.py:1-20` | Tests `waste_detector.py`. |
| `version one c2s/module/tests/test_waste_advisor.py` | app code | 6,063 B | Comprehensive test suite validating 17 functional query categories in `WasteAdvisor`. | `test_waste_advisor.py:1-50` | Tests `waste_advisor.py`. |
| `version one c2s/module/reports/` | generated | 47,814 B | Output directory containing 9 generated files (3 CSV, 3 JSON, 3 PDF reports). | `report_generator.py:40` | Created by `report_generator.py`. |
| `version one c2s/module/logs/waste_system.log` | generated | 1,778,726 B | System runtime execution log recording telemetry, detection events, and initialization diagnostics. | `main.py:58`, `web_app.py:75` | Written by Python `logging`. |
| `version one c2s/module/test_images/` | dataset | 280,002 B | 3 test images (`Plastic.jpeg`, `clothes.jpeg`, `spoon.jpeg`) for visual verification. | `app.js:275-290` | Sample images loaded in web UI. |
| `version one c2s/module/results/` | generated | 612,817 B | 3 annotated detection output images (`Plastic_detected.jpeg`, etc.). | `detect_plastic.py:80` | Generated by `detect_plastic.py`. |
| `version one c2s/module/version one/` | junk | 418 B (meta) | Abandoned virtual environment directory containing orphaned `pyvenv.cfg` and `.gitignore`. | `pyvenv.cfg:1-5` | Inactive/unused virtualenv artifact. |
| `version one c2s/tests/test_config_paths.py` | app code | 662 B | Unit test validating file path resolution in `config.py`. | `test_config_paths.py:1-25` | Tests `config.py`. |
| `version one c2s/yolo/weights/best.pt` | model | 22,510,826 B | Duplicate copy of YOLOv8s 5-class model located outside `module/`. | (name only) | Identical duplicate of `module/yolo/weights/best.pt`. |
| `version one c2s/waste-detection.v1i.yolov8/` | dataset | ~25 MB | Single-class `plastic` Roboflow dataset (330 files across `train`, `valid`, `test`) used to train `weights.pt`. | `data.yaml:1-16` | Standalone dataset folder. |
| `P-yesist-v2/P-yesist-v2/module/main.py` | app code | 21,337 B | Desktop OpenCV monitoring entry point for the P-yesist-v2 variant. | `main.py:167-205` | Imports `waste_detector.py`, `data_fusion.py`, `database.py`, `alert_system.py`. |
| `P-yesist-v2/P-yesist-v2/module/waste_detector.py` | app code | 50,842 B | YOLO detector for P-yesist-v2 (synced with dual-model fusion and plastic priority logic). | `waste_detector.py:143-152` | Loaded by P-yesist-v2 `main.py` and `detect_plastic.py`. |
| `P-yesist-v2/P-yesist-v2/module/config.py` | config | 10,231 B | Configuration dataclasses for P-yesist-v2 (does not define LlamaConfig). | `config.py:15-38` | Imported by P-yesist-v2 modules. |
| `P-yesist-v2/P-yesist-v2/module/data_fusion.py` | app code | 20,566 B | Multi-sensor fusion engine for P-yesist-v2. | `data_fusion.py:45-80` | Imported by P-yesist-v2 `main.py`. |
| `P-yesist-v2/P-yesist-v2/module/database.py` | app code | 14,193 B | SQLite database manager for P-yesist-v2. | `database.py:32-60` | Imported by P-yesist-v2 `main.py`. |
| `P-yesist-v2/P-yesist-v2/module/alert_system.py` | app code | 9,533 B | Alert management module for P-yesist-v2. | `alert_system.py:40-65` | Imported by P-yesist-v2 `main.py`. |
| `P-yesist-v2/P-yesist-v2/module/webcam_capture.py` | app code | 6,775 B | Camera capture wrapper for P-yesist-v2. | `webcam_capture.py:30-55` | Imported by P-yesist-v2 `main.py`. |
| `P-yesist-v2/P-yesist-v2/module/detect_plastic.py` | app code | 7,356 B | Standalone CLI detector for P-yesist-v2. | `detect_plastic.py:25-45` | Standalone script. |
| `P-yesist-v2/P-yesist-v2/module/train_waste_model.py` | app code | 9,982 B | Model training script for P-yesist-v2. | `train_waste_model.py:25-50` | Standalone script. |
| `P-yesist-v2/P-yesist-v2/module/report_generator.py` | app code | 3,505 B | Report generator for P-yesist-v2. | `report_generator.py:20-40` | Standalone script. |
| `P-yesist-v2/P-yesist-v2/module/requirements.txt` | config | 1,383 B | Dependency manifest for P-yesist-v2 (omits Flask and llama-cpp-python). | `requirements.txt:1-30` | Configuration. |
| `P-yesist-v2/P-yesist-v2/module/README.md` | docs | 2,887 B | Original documentation for P-yesist-v2 variant. | `README.md:1-40` | Standalone documentation. |
| `P-yesist-v2/P-yesist-v2/module/pico_sensors/main.py` | app code | 9,661 B | MicroPython firmware for Pico W in P-yesist-v2. | `pico_sensors/main.py:20-50` | Telemetry firmware. |
| `P-yesist-v2/P-yesist-v2/module/yolo/weights/best.pt` | model | 22,510,826 B | 5-class YOLOv8s trained model in P-yesist-v2 weights folder. | `waste_detector.py:257` | Loaded by P-yesist-v2 `WasteDetector`. |
| `P-yesist-v2/P-yesist-v2/module/yolo/weights/training 2 .pt` | model | 22,510,826 B | Identical duplicate of `best.pt` with spaces in filename. | (name only) | Unused duplicate. |
| `P-yesist-v2/P-yesist-v2/module/yolo/weights/weights.pt` | model | 20,342,981 B | Single-class `bottle` YOLOv8s model in P-yesist-v2. | `waste_detector.py:276` | Fallback model in P-yesist-v2. |
| `P-yesist-v2/P-yesist-v2/module/yolo/weights/weights.onnx` | model | 38,155,638 B | ONNX exported model for single-class `bottle` (input 640x640). | (name only) | ONNX model file. |
| `P-yesist-v2/P-yesist-v2/module/yolov8n.pt` | model | 6,549,796 B | Pretrained COCO 80-class YOLOv8n weights in P-yesist-v2 module. | `waste_detector.py:314` | General model in `WasteDetector`. |
| `P-yesist-v2/P-yesist-v2/module/models/llama-3.2-1b-instruct-q4_k_m.gguf` | model | 807,694,464 B | Hardlinked copy of Llama 3.2 1B GGUF model. | (name only) | Model file. |
| `P-yesist-v2/P-yesist-v2/module/database/waste_monitoring.db` | database | 4,481,024 B | SQLite database for P-yesist-v2 (5,701 records, 5,701 sensor logs, 73 alerts). | `database.py:35` | Read/written by P-yesist-v2. |
| `P-yesist-v2/P-yesist-v2/module/database/README.md` | docs | 231 B | Note on database schema. | `README.md:1-10` | Documentation. |
| `P-yesist-v2/P-yesist-v2/module/data/waste_monitoring.db` | database | 36,864 B | Seed/test database (30 records) in P-yesist-v2. | `config.py:139` | Seed database. |
| `P-yesist-v2/P-yesist-v2/module/tests/test_material_mapping.py` | app code | 524 B | Unit test for material normalization in P-yesist-v2. | `test_material_mapping.py:1-13` | Tests `waste_detector.py`. |
| `P-yesist-v2/P-yesist-v2/module/reports/` | generated | 47,814 B | Generated reports directory in P-yesist-v2 (9 files). | `report_generator.py` | Output directory. |
| `P-yesist-v2/P-yesist-v2/module/logs/waste_system.log` | generated | 869,047 B | Log file for P-yesist-v2. | `main.py:58` | Generated log. |
| `P-yesist-v2/P-yesist-v2/PROJECT_STATUS.md` | docs | 2,387 B | Project status summary for P-yesist-v2. | `PROJECT_STATUS.md:1-40` | Standalone documentation. |
| `P-yesist-v2/P-yesist-v2/hailo_acceleration_archive/` | app code / model | ~38.2 MB | 5 files: `HAILO_INTEGRATION_GUIDE.md` (2,762 B), `hailo_export.py` (16,713 B), `inference_backend.py` (32,832 B), `test_hailo_backend.py` (8,773 B), `weights.onnx` (38,155,638 B) evaluating HailoRT NPU acceleration. | `hailo_export.py:1-40`, `HAILO_INTEGRATION_GUIDE.md:1-50` | Isolated experimental archive. |
| `P-yesist-v2/P-yesist-v2/tests/test_config_paths.py` | app code | 662 B | Unit test for config paths. | `test_config_paths.py:1-25` | Tests `config.py`. |
| `P-yesist-v2/P-yesist-v2/waste-detection.v1i.yolov8/` | dataset | ~25 MB | Single-class `plastic` Roboflow dataset copy (330 files). | `data.yaml:1-16` | Standalone dataset folder. |
| `test C2S/garbage detection.v4i.yolov8/` | dataset | ~100 MB | 5-class Roboflow dataset (`cardboard`, `glass`, `metal`, `paper`, `plastic`) (~1,500 files). | `data.yaml:1-13` | Training dataset directory. |
| `test C2S/waste.v28i.yolov11/` | dataset | ~1.2 GB | 9-class Roboflow YOLOv11 dataset (`Electronics`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`) (~20,000 files). | `data.yaml:1-13` | Training dataset directory. |
| `test C2S/waste.v28i.yolov11.zip` | dataset | 744,366,721 B | ZIP archive of the 9-class YOLOv11 dataset. | (name only) | Raw ZIP archive. |
| Root empty directories (`database`, `logs`, `reports/json`, `evidence`, `yolo/weights`, `yolo/training_data/waste`) | junk | 0 B | Empty directory skeletons located at the root of the workspace. | (name only) | Unused directories. |

---

## D. How the App Runs Today

### Frontend
- **Framework:** Vanilla HTML5, Vanilla CSS3, Vanilla ES6 JavaScript (zero external JS framework dependencies; Chart.js is not imported; custom CSS and canvas are used).
- **Entry File:** `version one c2s/module/web/templates/index.html` (structure), `version one c2s/module/web/static/js/app.js` (logic), `version one c2s/module/web/static/css/style.css` (presentation).
- **Start Mechanism & Port:** Served directly by the Flask backend (`web_app.py:152`) on HTTP port `5000` (configurable via CLI flag `--port`).
- **Calls to the Backend:**
  1. `GET /api/webcam/status` (`app.js:176`): Checks camera connectivity on page load.
  2. `GET /api/llama/status` (`app.js:196`): Checks Llama 3.2 engine status for the UI badge.
  3. `POST /api/detect` (`app.js:295`): Sends `FormData` with image file or base64 JSON payload for AI detection.
  4. `GET /api/webcam/stream?<timestamp>` (`app.js:321`): Sets `<img id="webcamFeed">` src to stream live MJPEG video.
  5. `POST /api/webcam/capture` (`app.js:357`): Requests the server to snapshot and classify the current live webcam frame.
  6. `GET /api/stats` (`app.js:529`): Retrieves total records, waste distribution counts, and daily stats for dashboard cards.
  7. `GET /api/records?limit=50[&type=...]` (`app.js:628`): Fetches paginated/filtered historical waste records for the table.
  8. `POST /api/chat` (`app.js:691`, `app.js:738`): Sends `{ message: text, detection_context: ctx }` to the chatbot endpoint.

### Backend
- **Framework:** Flask 3.1.3 (Python WSGI).
- **Entry File:** `version one c2s/module/web_app.py`.
- **Run Command:** `python web_app.py` (or `python web_app.py --port 5000 --model yolo/weights/best.pt`).
- **Complete Route Inventory:**
  1. `GET /` (`web_app.py:152`): Serves `index.html`. Request: none. Response: `text/html`.
  2. `POST /api/detect` (`web_app.py:161`): Detects waste in image. Request: multipart `image` file OR JSON `{"image": "<base64>"}`. Response: JSON `{"success": true, "detections": [...], "annotated_image": "<base64>", "advice": {...}, "object_guess": str, "plastic_type": str, "recycling_steps": [...], "household_reuse": [...], "summary": {...}}`.
  3. `POST /api/chat` (`web_app.py:319`): AI assistant chatbot with hybrid WasteAdvisor knowledge engine + Llama 3.2 GGUF routing. Request: JSON `{"message": str, "detection_context": dict, "force_llama": bool}`. Response: JSON `{"success": true, "response": str, "type": str, "engine": str, "waste_info": dict, "inference_ms": float}`.
  4. `GET /api/llama/status` (`web_app.py:412`): Health check for Llama engine. Request: none. Response: JSON `{"success": true, "active_backend": str, "load_status": str, "model_path": str}`.
  5. `GET /api/stats` (`web_app.py:423`): Dashboard summary metrics. Request: none. Response: JSON `{"success": true, "total_records": int, "waste_distribution": [{"waste_type": str, "count": int}], "daily_stats": [...], "recent_alerts": [...]}`.
  6. `GET /api/records` (`web_app.py:456`): Recent waste classifications. Request query: `?limit=50&type=plastic`. Response: JSON `{"success": true, "count": int, "records": [...]}`.
  7. `GET /api/sensors` (`web_app.py:480`): Sensor history logs. Request query: `?limit=50`. Response: JSON `{"success": true, "count": int, "readings": [...]}`.
  8. `GET /api/alerts` (`web_app.py:497`): Hardware/safety alerts. Request query: `?limit=50`. Response: JSON `{"success": true, "count": int, "alerts": [...]}`.
  9. `GET /api/webcam/stream` (`web_app.py:578`): Live multipart video stream. Request: none. Response: `multipart/x-mixed-replace; boundary=frame` MJPEG stream.
  10. `POST /api/webcam/capture` (`web_app.py:587`): Snapshots live camera frame and classifies it. Request: none. Response: JSON matching `/api/detect`.
  11. `GET /api/webcam/status` (`web_app.py:679`): Webcam availability test. Request: none. Response: JSON `{"available": bool, "status": str}`.

### Step-by-Step Request Flow

#### 1. Image Upload Flow
1. User drops/selects an image in the web UI: `app.js:241-260` (`handleImageFile(file)`).
2. UI creates `FormData` with `'image': file` and calls `fetch('/api/detect', { method: 'POST', body: formData })`: `app.js:295-298`.
3. Backend route receives request: `web_app.py:161-177` (`detect_waste()` extracts bytes via `request.files['image']` and decodes with `cv2.imdecode`).
4. Backend ensures components are active via `ensure_components()` (`web_app.py:194`) and executes `detector.detect(image, frame_counter)` (`web_app.py:200`).
5. Inside `waste_detector.py`:
   - Runs `general_model.predict(image)` (`waste_detector.py:496`).
   - Runs `model.predict(image)` (`waste_detector.py:535`).
   - Gating: requires confidence $\ge 0.35$ for `paper_cardboard` (`waste_detector.py:547`) and $\ge 0.60$ for `book` (`waste_detector.py:508`).
   - Prioritizes candidates with plastic priority bonus $+0.20$ (`waste_detector.py:566-570`).
   - Reconciles overlapping bounding boxes (IoU $> 0.25$), prioritizing plastic over paper (`waste_detector.py:580-588`).
6. Backend generates annotated JPEG base64: `detector.draw_detections()` (`web_app.py:209`), writes record to SQLite `active_db.insert_waste_record()` (`web_app.py:228`), computes item advice via `advisor.analyze_detection()` (`web_app.py:257-299`), and returns JSON: `web_app.py:300-316`.
7. Frontend receives response in `app.js:299-315`, updates detection HUD, displays annotated image, renders recycling steps and badges, and passes context to chat: `app.js:380-450`.

#### 2. Gallery Pick Flow
- **Web App:** User clicks a sample image card (`index.html:158-185`, `app.js:275-290` `selectSampleImage(src)`). `app.js` fetches sample image blob via `fetch(src)`, converts to `File`, and feeds it into `handleImageFile()` (`app.js:285-290` $\rightarrow$ `app.js:295` `fetch('/api/detect')`), proceeding through the Image Upload Flow above.
- **Desktop App (`main.py`):**
  1. User presses `'g'` key in OpenCV window loop: `main.py:382`.
  2. Opens Windows OS image file dialog via `tkinter.filedialog.askopenfilename()`: `main.py:100-117` (`select_gallery_image()`).
  3. Passes path to `inspect_gallery_image(detector, image_path, ...)`: `main.py:119-148`.
  4. Reads image with `cv2.imread(image_path)`: `main.py:133`.
  5. Calls `detector.detect(image)`: `main.py:137`.
  6. Overlays HUD with `detector.draw_detections(image, waste_frame)` and renders on screen with `cv2.imshow()`: `main.py:141-144`.

#### 3. Live Webcam Flow
- **Web App:**
  1. User selects "Live Camera": `app.js:318-325`.
  2. Sets video image source `img.src = '/api/webcam/stream?' + Date.now()`: `app.js:321`.
  3. Route `/api/webcam/stream` (`web_app.py:578-585`) returns `Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')`.
  4. In `generate_webcam_frames()` (`web_app.py:507-576`):
     - Frame read from camera via `camera.read()` (`web_app.py:521-529`).
     - Every N frames runs AI inference `detector.detect(frame)` (`web_app.py:540-549`).
     - Draws HUD bounding boxes `detector.draw_detections(frame, waste_frame)` (`web_app.py:550`).
     - Compresses frame to JPEG `cv2.imencode('.jpg', frame)` (`web_app.py:557`).
     - Yields multipart chunk (`web_app.py:560-564`).
  5. If user clicks "Capture Frame", `app.js:357` sends `POST /api/webcam/capture`.
  6. Backend route `/api/webcam/capture` (`web_app.py:587-677`) captures current frame, runs `detector.detect()`, writes to SQLite database, analyzes disposal steps, and returns JSON identical to `/api/detect`.

---

## E. YOLO / Vision Models

### Verified Model Files

1. **`version one c2s/module/yolo/weights/best.pt`**
   - **Size:** 22,510,826 bytes (21.47 MB)
   - **Last-Modified:** 2026-10-10 13:00:08
   - **Task:** Detection (`detect`)
   - **Architecture & Training:** YOLOv8s, PyTorch/Ultralytics 8.4.175, training image size 640x640, 60 epochs.
   - **Classes (5):** `{0: 'plastic_bottle', 1: 'glass_bottle', 2: 'metal_can', 3: 'paper_cardboard', 4: 'organic_waste'}`
   - **Loaded By:** `version one c2s/module/waste_detector.py:295` (`self.model = YOLO(self.config.custom_model_path)`).
   - **Status:** **Active Primary Model** used by the web app and desktop system.

2. **`version one c2s/module/yolov8n.pt`**
   - **Size:** 6,549,796 bytes (6.25 MB)
   - **Last-Modified:** 2026-10-06 00:18:07
   - **Task:** Detection (`detect`)
   - **Architecture & Training:** Pretrained YOLOv8n COCO model (80 classes, 640x640).
   - **Classes (80):** Standard COCO classes (indices 0–79: person, bicycle, ..., bottle (39), wine glass (40), cup (41), fork (42), knife (43), spoon (44), bowl (45), banana (46), apple (47), sandwich (48), orange (49), broccoli (50), carrot (51), hot dog (52), pizza (53), donut (54), cake (55), book (73), toothbrush (79), etc.).
   - **Loaded By:** `version one c2s/module/waste_detector.py:314-323` (`self.general_model = YOLO(g_path)`).
   - **Status:** **Active Secondary Model** used for dual-model fusion to detect non-bottle plastic items (cups, cutlery, bowls) and organic food items.

3. **`version one c2s/module/yolo/weights/weights.pt`**
   - **Size:** 20,342,981 bytes (19.40 MB)
   - **Last-Modified:** 2026-09-30 13:13:05
   - **Task:** Detection (`detect`)
   - **Architecture & Training:** YOLOv8s, Ultralytics 8.4.55, 300 epochs, 640x640.
   - **Classes (1):** `{0: 'bottle'}`
   - **Loaded By:** `version one c2s/module/waste_detector.py:302` (fallback if `best.pt` is missing).
   - **Status:** **Old single-class model**. Kept for fallback compatibility.

4. **`P-yesist-v2/P-yesist-v2/module/yolo/weights/best.pt`**
   - **Size:** 22,510,826 bytes (21.47 MB)
   - **Last-Modified:** 2026-10-10 09:24:51
   - **Classes (5):** `{0: 'plastic_bottle', 1: 'glass_bottle', 2: 'metal_can', 3: 'paper_cardboard', 4: 'organic_waste'}`
   - **Loaded By:** `P-yesist-v2/P-yesist-v2/module/waste_detector.py:257`.
   - **Status:** **Active Primary Model** for `P-yesist-v2`.

5. **`P-yesist-v2/P-yesist-v2/module/yolo/weights/training 2 .pt`**
   - **Size:** 22,510,826 bytes (21.47 MB)
   - **Status:** **Exact duplicate** of `best.pt` with spaces in filename.

6. **`P-yesist-v2/P-yesist-v2/module/yolo/weights/weights.pt`**
   - **Size:** 20,342,981 bytes (19.40 MB)
   - **Classes (1):** `{0: 'bottle'}`
   - **Status:** Duplicate of the single-class model.

7. **`P-yesist-v2/P-yesist-v2/module/yolov8n.pt`**
   - **Size:** 6,549,796 bytes (6.25 MB)
   - **Classes (80):** Pretrained COCO.
   - **Status:** General model for P-yesist-v2.

8. **`P-yesist-v2/P-yesist-v2/hailo_acceleration_archive/weights.onnx`** (and `module/yolo/weights/weights.onnx`)
   - **Size:** 38,155,638 bytes (36.39 MB)
   - **Last-Modified:** 2026-10-09 23:21:18
   - **Architecture:** ONNX IR v7, exported from PyTorch 2.14.0.
   - **Inputs/Outputs:** Input `images` shape `[1, 3, 640, 640]`, Output `output0` shape `[1, 5, 8400]`.
   - **Classes (1):** `{0: 'bottle'}`
   - **Loaded By:** `P-yesist-v2/P-yesist-v2/hailo_acceleration_archive/hailo_export.py:25` and `inference_backend.py:120`.
   - **Status:** **Experimental archive** for HailoRT evaluation.

### Inference & Runtime Behavior
- **Inference Settings:**
  - `imgsz`: 640 (`config.py:128`)
  - `conf`: 0.35 default (`config.py:126`); custom model runs at `max(0.20, conf - 0.10)` (`waste_detector.py:526`)
  - `iou`: 0.45 default (`config.py:127`); candidate NMS/overlap threshold at 0.25 (`waste_detector.py:578`)
  - `device`: `"cpu"` default (`config.py:129`); `"0"` if GPU available
  - `half`: `False` on CPU (`config.py:130`); `True` only when device != `"cpu"`
- **Loading Behavior:** Model loads **once** at application initialization (`web_app.py:110-114` and `main.py:202-204`), warmed up with a dummy zero tensor (`waste_detector.py:340-353`). It does **not** reload on each request.
- **Webcam Loop Blocking:** In `main.py:250`, OpenCV `camera.read_frame()` executes synchronously inside the main UI loop, meaning inference latency directly bounds the UI frame rate. In `web_app.py:507-576`, each MJPEG client stream runs an independent generator thread reading from the camera.
- **Visualization:** `WasteDetector.draw_detections()` (`waste_detector.py:688-900`) overlays semi-transparent color bounding boxes, class labels, percentage confidence scores, centroid markers, and a top-level neon summary HUD bar directly onto the BGR frame.

---

## F. Chatbot (Llama 3.2 1B)

- **How Loaded:** Loaded via `llama-cpp-python` using `from llama_cpp import Llama` (`llama_engine.py:31`, `llama_engine.py:139-145`). Secondary stub for ExecuTorch `Program.load_from_file(path)` (`llama_engine.py:167`).
- **Model File Path & Size:**
  - Path: `version one c2s/module/models/llama-3.2-1b-instruct-q4_k_m.gguf`
  - Size: 807,694,464 bytes (770.28 MB)
- **Format & Quantization:** GGUF format, 4-bit medium quantization (`Q4_K_M`).
- **Context Length (`n_ctx`):** 2048 tokens (`config.py:166`, `llama_engine.py:141`).
- **Generation Settings:**
  - `max_tokens`: 150 tokens (`llama_engine.py:295`)
  - `temperature`: 0.3 (`llama_engine.py:299`)
  - `top_p`: 0.9 (`llama_engine.py:300`)
  - `stop`: `["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "\n\nUser:", "\n\nuser", "\nUser:", "\nuser:"]` (`llama_engine.py:301`)
  - `n_threads`: `max(1, os.cpu_count() or 4)` (`llama_engine.py:142`)
  - `n_gpu_layers`: 0 (CPU-only execution) (`llama_engine.py:143`)
- **System Prompt (First 400 Characters):**
  > `"You are an expert AI environmental and waste assistant running locally on-device. The user uploads pictures or scans waste items in this session. Computer vision models analyze every uploaded photo and provide you with the visual detection breakdown (detected objects, counts, confidence, materials, and reuse options) in the Context section.\nCrucial Vision & Interaction Rules:\n- The Context describes the"` (`llama_engine.py:57-69`).
- **Retrieval & Data Source:** Hybrid retrieval-augmented routing:
  1. The backend routes the query to `WasteAdvisor` (`web_app.py:334`). If a specific item, material, resin code (#1–#7), or disposal question matches the 2,300-line dictionary, it returns an instant (<5ms) deterministic answer (`web_app.py:338-346`).
  2. If the user asks open-ended or conversational questions, `LlamaEngine.format_prompt()` (`llama_engine.py:178-228`) injects detection context (detected item, confidence, resin code, disposal bin, upcycling ideas) into `<|start_header_id|>system<|end_header_id|>` and invokes GGUF inference (`llama_engine.py:296-303`).
- **Rough RAM Needed:** ~1.2 GB to 1.5 GB RAM on CPU during inference.

---

## G. Database

- **Engine:** SQLite 3 (embedded serverless, standard Python `sqlite3`).
- **Locations:**
  - Primary Active DB: `version one c2s/module/database/waste_monitoring.db` (7.31 MB, 7,311,360 bytes).
  - P-yesist-v2 Active DB: `P-yesist-v2/P-yesist-v2/module/database/waste_monitoring.db` (4.48 MB, 4,481,024 bytes).
  - Seed / Test DB: `version one c2s/module/data/waste_monitoring.db` (36 KB, 36,864 bytes).
- **Tables, Column Names, Types, and Row Counts (Verified via SQLite Read-Only Mode):**

  1. **Table `waste_records` (9,811 rows in primary DB):**
     - `id` INTEGER (Primary Key)
     - `timestamp` TEXT
     - `waste_type` TEXT
     - `ai_confidence` REAL
     - `ai_class` TEXT
     - `humidity` REAL
     - `temperature` REAL
     - `gas_level` INTEGER
     - `moisture_level` INTEGER
     - `bin_fill_level` REAL
     - `is_wet` INTEGER
     - `is_hazardous` INTEGER
     - `gas_alert` TEXT
     - `bin_alert` TEXT
     - `vehicle_id` TEXT
     - `created_at` TEXT

  2. **Table `sensor_log` (9,599 rows in primary DB):**
     - `id` INTEGER (Primary Key)
     - `timestamp` TEXT
     - `distance_cm` REAL
     - `bin_fill_pct` REAL
     - `temperature_c` REAL
     - `humidity_pct` REAL
     - `gas_analog` INTEGER
     - `gas_digital` INTEGER
     - `moisture_raw` INTEGER
     - `created_at` TEXT

  3. **Table `alerts` (112 rows in primary DB):**
     - `id` INTEGER (Primary Key)
     - `timestamp` TEXT
     - `alert_type` TEXT
     - `severity` TEXT
     - `message` TEXT
     - `resolved` INTEGER
     - `vehicle_id` TEXT
     - `created_at` TEXT

  4. **Table `json_reports` (35 rows in primary DB):**
     - `id` INTEGER (Primary Key)
     - `timestamp` TEXT
     - `report_type` TEXT
     - `payload` TEXT
     - `created_at` TEXT

- **What the App Stores:** Classification history (material, confidence, bounding box), environmental readings from the Pico W (bin distance, fill percentage, temperature, humidity, gas ADC, soil moisture), safety alerts (gas warning/danger, bin full, hazardous item), and hourly JSON summary reports.
- **Files that Read or Write It:**
  - Writes: `version one c2s/module/database.py:115-200`, `web_app.py:228`, `main.py:330-360`, `alert_system.py:150-180`.
  - Reads: `version one c2s/module/database.py:220-330`, `web_app.py:417-505`, `report_generator.py:40-70`.

---

## H. Dependencies and Environment

- **Dependency Manifests:**
  - `version one c2s/module/requirements.txt`: Pins `ultralytics>=8.0.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `torch>=2.0.0`, `torchvision>=0.15.0`, `pyserial>=3.5`, `reportlab>=4.0`, `matplotlib>=3.7.0`, `Pillow>=10.0.0`, `streamlit>=1.30.0`, `plotly>=5.18.0`, `pandas>=2.1.0`, `requests>=2.31.0`, `flask>=3.0.0`, `llama-cpp-python>=0.2.80`.
  - `P-yesist-v2/P-yesist-v2/module/requirements.txt`: Same as above, but omits `flask` and `llama-cpp-python`.
- **Installed Library Versions (Verified in `.venv`):**
  - `torch`: `2.14.0+cpu`
  - `torchvision`: `0.29.0+cpu`
  - `ultralytics`: `8.4.163`
  - `opencv-python` (`cv2`): `5.0.0`
  - `numpy`: `2.5.3`
  - `onnxruntime`: `1.31.0`
  - `transformers`: **NOT INSTALLED** (not needed; GGUF inference uses `llama_cpp`)
  - `llama-cpp-python` (`llama_cpp`): `0.3.36`
  - `flask`: `3.1.3`
  - `fastapi` / `django`: **NOT INSTALLED** (app uses Flask)
  - `streamlit`: `1.64.0`
  - `pandas`: `3.0.6`
  - `plotly`: `7.1.0`
  - `reportlab`: `5.0.1`
  - `pyserial` (`serial`): `3.5`
- **Conflicts & Missing Pins:**
  - `numpy 2.5.3` is installed with `torch 2.14.0` and `opencv 5.0.0`; all current C-extension bindings operate normally.
  - `requirements.txt` specifies minimum bounds (`>=`) rather than exact frozen versions (`==`).
- **Hardware Resources Available:**
  - Host OS: Windows 11 (Build 10.0.26300 AMD64)
  - CPU: AMD Family 23 Model 160 Stepping 0 (Multiple CPU cores detected)
  - CUDA / GPU: `torch.cuda.is_available() == False` (CPU-only execution on this development machine)

---

## I. Problems and Risks

1. **Accuracy Risks:**
   - **Narrow Custom Training Class Taxonomy:** The trained custom model `best.pt` (`waste_detector.py:163-169`) only recognizes 5 classes (`plastic_bottle`, `glass_bottle`, `metal_can`, `paper_cardboard`, `organic_waste`). It lacks classes for generic plastics (cups, bags, cling wrap, blister packs).
   - **COCO Class Mapping Ambiguities:** In `COCO_WASTE_MAP` (`waste_detector.py:188-220`), classes such as `book` (mapped to `paper`), `tie`/`umbrella` (mapped to `dry_waste`), or `scissors` (mapped to `metal`) rely on confidence gating ($\ge 0.60$ for `book`, `waste_detector.py:508`) to prevent misclassification of flat plastic packaging.

2. **Speed and Memory Risks for Small SoC (Raspberry Pi 5 / Jetson Nano):**
   - **Concurrent Memory Footprint:** Running PyTorch YOLOv8s (~500 MB) alongside Llama 3.2 1B GGUF (~807 MB weights + ~400 MB context/KV cache) consumes $\approx 1.5 - 1.8$ GB RAM. On a 2 GB or 4 GB RAM edge SoC, memory spikes could trigger OOM killer termination.
   - **Video Streaming Generator Resource Multiplier:** In `web_app.py:507-576`, `/api/webcam/stream` runs a synchronous loop invoking `detector.detect()`. If multiple browser tabs or clients connect to this endpoint, Flask spawns multiple threads attempting concurrent inference on the camera, which will saturate CPU resources.

3. **Security Risks:**
   - **Unbounded File Upload Buffer:** In `web_app.py:173-190`, the `/api/detect` route reads uploaded files into memory via `request.files['image'].read()` without an explicit `MAX_CONTENT_LENGTH` cap configured on the Flask app.
   - **Open Network Binding:** In `web_app.py:722`, `app.run(host="0.0.0.0", port=port, debug=False)` binds to all network interfaces without authentication tokens or CSRF validation.
   - **Credential Storage:** In `config.py:182-192`, `AlertConfig` defines `smtp_username` and `smtp_password`. While currently defaulting to empty strings `""`, storing credentials here would expose them in plaintext if configured.

4. **Unfinished or Dead Code:**
   - **ExecuTorch Stub Runtime:** In `executorch_llama_engine.py` and `llama_engine.py:154-174`, the ExecuTorch backend is a stub expecting a `.pte` file that is not compiled or present in the repository.
   - **Segregated Hailo NPU Archive:** `P-yesist-v2/P-yesist-v2/hailo_acceleration_archive/` contains exported ONNX weights and test scripts, but is completely disconnected from the active web app pipeline.
   - **Codebase Duplication:** `version one c2s/module` and `P-yesist-v2/P-yesist-v2/module` are parallel duplicate codebases with slight divergences in `config.py` and `requirements.txt`.
   - **Root Empty Directories:** `c:\Users\kumar\OneDrive\C2S\database`, `logs`, `evidence`, `yolo/weights`, `reports/json` are orphaned empty directories at the workspace root.
   - **Missing Tests:** No unit tests exist for `alert_system.py`, `report_generator.py`, or `pico_sensors/main.py`.

---

## J. UNKNOWN Items and Questions Only the User Can Answer

1. **Target Demonstration Hardware:** Which physical hardware target will be used for the final submission / demonstration (Raspberry Pi 5, Jetson Nano, Jetson Orin, or a Windows/Linux laptop)?
2. **Canonical Codebase Selection:** Which directory should be treated as the official, canonical project: `version one c2s` (full-stack Flask web application with live camera and chat) or `P-yesist-v2` (desktop OpenCV HUD)?
3. **Pico W Serial Port:** When deploying with physical hardware, what specific serial port or tty device (`COM3`, `/dev/ttyACM0`) is assigned to the Raspberry Pi Pico W, or should the app continue using simulated sensor mode?
4. **ExecuTorch Requirement:** For the Bharat AI-SoC Challenge PS2 submission, is an actual `.pte` compiled ExecuTorch binary required, or is the on-device `llama-cpp-python` GGUF engine accepted as meeting the on-device Llama requirement?
5. **Orphaned Root Directories:** Can the orphaned empty directories at the root (`/database`, `/logs`, `/evidence`, `/reports`, `/yolo`) and the abandoned `version one` venv folder be deleted or consolidated?

---

Self-check: 8 of 8 verified
