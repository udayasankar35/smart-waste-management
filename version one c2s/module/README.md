# 🗑️ AI-Powered Smart Waste Classification & Disposal Assistant

> **Bharat AI-SoC Challenge 2026–27 — Problem Statement 2 (PS2)**
>
> An AI-powered assistant that helps users identify, classify, and properly dispose of waste materials using **Llama + ExecuTorch** running on **Arm-based devices**, augmented with YOLOv8 computer vision and multi-sensor environmental monitoring.

---

## 🏆 Competition: Bharat AI-SoC Challenge 2026–27

| Field | Detail |
|-------|--------|
| **Organisers** | Arm + MeitY C2S Programme + IIT Delhi |
| **Theme** | Applied AI on SoC |
| **Problem Statement** | PS2 — AI-Powered Waste Classification & Disposal Assistant |
| **Registration Deadline** | 19 October 2026 |
| **Support** | [support@armbharatchallenge.com](mailto:support@armbharatchallenge.com) |
| **Technical Queries** | [tech-queries@armbharatchallenge.com](mailto:tech-queries@armbharatchallenge.com) |

---

## 📋 PS2 — Requirements Summary

### Objective
Develop an AI-powered assistant that helps users **identify, classify, and properly dispose of waste materials** using Llama and ExecuTorch running on Arm-based devices.

### Functional Requirements (Mandatory)
- [x] **Classify waste materials** — identify waste category (plastic, recyclable, metal, dry, wet, hazardous, e-waste)
- [ ] **Provide disposal recommendations** — actionable guidance on how to dispose of each detected item
- [ ] **Explain recycling options** — educate users about what can be recycled and how
- [ ] **Demonstrate use of Llama** — Llama 3.2 1B/3B-Instruct (or fine-tuned/distilled variant) for advisory/reasoning
- [ ] **Demonstrate use of ExecuTorch** — Llama deployed on-device using ExecuTorch as the reference runtime
- [ ] **Promote sustainable waste management practices**
- [ ] **Educate users about environmental impact**

### Required Technologies
| Technology | Purpose | Status |
|------------|---------|--------|
| **Llama** (Llama 3.2 1B/3B-Instruct) | Advisory & reasoning component — disposal guidance, recycling education, environmental impact | 🔴 Not yet integrated |
| **ExecuTorch** | On-device Llama inference runtime on Arm CPU | 🔴 Not yet integrated |

### Optional Technologies
| Technology | Purpose | Status |
|------------|---------|--------|
| **OpenCV** | Image preprocessing, camera capture, frame processing | 🟢 Already integrated |
| **Hailo SDK** | NPU acceleration for vision models via HailoRT (separate pipeline from Llama) | 🔴 Not yet integrated |
| **Speech processing frameworks** | Voice-based advisory interface | 🔴 Not yet integrated |
| **Computer vision models** (YOLOv8) | Waste detection & classification | 🟢 Already integrated |

---

## 🖥️ Possible Interfaces (Choose one or more)

The competition allows teams to design the most appropriate user experience. Below are the permitted interface options:

| Interface | Description | Feasibility for Our Project |
|-----------|-------------|---------------------------|
| **📱 Android Application** | Native/cross-platform mobile app with camera-based waste scanning | ⭐ High — can wrap existing Python pipeline in Android using ExecuTorch Mobile |
| **💬 Chat-based Assistant** | Conversational AI using Llama for waste queries & disposal advice | ⭐ High — natural fit for Llama advisory component |
| **🎤 Voice-based Assistant** | Hands-free waste disposal guidance via speech recognition + TTS | ⭐ Medium — requires speech framework integration |
| **📷 Camera-enabled Assistant** | Live camera feed with real-time waste detection + overlay UI | ⭐ High — **already built** (YOLOv8 + OpenCV pipeline) |
| **📲 WhatsApp Integration** | Users send waste photos via WhatsApp, receive disposal guidance | ⚡ Medium — requires WhatsApp Business API or Twilio |
| **🌐 Multimodal AI Experience** | Combined vision + language + voice + sensor fusion | ⭐ High — our multi-sensor architecture is a strong foundation |

### Recommended Interface Combination for Our Project
```
┌─────────────────────────────────────────────────────────────┐
│         MULTIMODAL WASTE CLASSIFICATION ASSISTANT           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  📷 Camera   │  │  💬 Chat     │  │  🎤 Voice        │  │
│  │  Interface   │  │  Interface   │  │  Interface       │  │
│  │  (YOLOv8 +   │  │  (Llama 3.2 │  │  (Speech-to-Text │  │
│  │   OpenCV)    │  │   advisory)  │  │   + TTS)         │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│         ▼                 ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              DATA FUSION ENGINE                     │   │
│  │     AI Vision + Sensor Data + Llama Reasoning       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│    Waste Class     Disposal Guide    Recycling Info        │
│    + Severity      + Nearest Center   + Env. Impact       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hardware Platforms (Competition-Approved)

Teams may use **any Arm-based platform** appropriate for their solution:

| Platform | Llama Runtime | Vision Pipeline | Sensors | Status |
|----------|--------------|-----------------|---------|--------|
| **Arm-based Android Smartphone** | ExecuTorch (Arm CPU) | YOLOv8 via TFLite/ONNX | Phone sensors (camera, GPS) | 🟡 Planned |
| **Raspberry Pi 5 (CPU-only)** | ExecuTorch (Arm Cortex-A76) | YOLOv8 via OpenCV | Pico W sensor array | 🟡 Planned — primary target |
| **Raspberry Pi 5 + AI HAT+** | ExecuTorch (Arm CPU) | YOLOv8 via HailoRT (NPU) | Pico W sensor array | 🟡 Planned — enhanced target |
| **Raspberry Pi 5 + AI HAT+ 2** | ExecuTorch (Arm CPU) | YOLOv8 via HailoRT (NPU) | Pico W sensor array | 🟡 Optional |
| **Arm-based Linux Device** | ExecuTorch (Arm CPU) | YOLOv8 via OpenCV | External sensors | 🟡 Compatible |

> **⚠️ Important (Hailo Note):** If a Hailo accelerator is used for vision workloads, vision models run through **HailoRT on the Hailo NPU**, while Llama runs through **ExecuTorch on the Arm CPU**. These are **separate pipelines** and must be clearly documented.

### Current Hardware (Version 1 — to be migrated)
| Component | Model | Purpose |
|-----------|-------|---------|
| AI Processor | NVIDIA Jetson Nano | YOLOv8 inference (**must migrate to Arm**) |
| Microcontroller | Raspberry Pi Pico W | Sensor interfacing (✅ keep) |
| Camera | USB Webcam | Waste image capture (✅ keep) |
| Distance Sensor | HC-SR04 | Bin fill level detection (✅ keep) |
| Humidity Sensor | DHT11 | Wet waste identification (✅ keep) |
| Gas Sensor | MQ Module | Toxic gas detection (✅ keep) |
| Moisture Sensor | HW-103 | Wet/dry waste classification (✅ keep) |

---

## 🏗️ System Architecture (Competition-Aligned)

```
                        ┌──────────────────────────────┐
                        │    USER INTERFACE LAYER       │
                        │  Android App / Chat / Voice   │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │      ARM-BASED DEVICE        │
                        │  (RPi 5 / Android / Linux)    │
                        │                              │
                        │  ┌────────────────────────┐  │
                        │  │   LLAMA 3.2 (1B/3B)    │  │
                        │  │   via ExecuTorch        │  │
                        │  │   (Arm CPU)             │  │
                        │  │   • Disposal advice     │  │
                        │  │   • Recycling guidance  │  │
                        │  │   • Environmental info  │  │
                        │  └────────────┬───────────┘  │
                        │               │              │
                        │  ┌────────────▼───────────┐  │
                        │  │   YOLOv8 Vision        │  │
                        │  │   (Arm CPU or Hailo NPU)│  │
                        │  │   • Waste detection     │  │
                        │  │   • Classification      │  │
                        │  │   • Severity scoring    │  │
                        │  └────────────┬───────────┘  │
                        │               │              │
USB Webcam ─────────────┤  ┌────────────▼───────────┐  │
                        │  │   DATA FUSION ENGINE   │  │
RPi Pico W ─(Serial)────┤  │   AI + Sensor Fusion   │  │
 ├─ HC-SR04 (Distance)  │  └────────────┬───────────┘  │
 ├─ DHT11 (Temp/Humid)  │               │              │
 ├─ MQ Gas Sensor       │  ┌────────────▼───────────┐  │
 └─ HW-103 (Moisture)   │  │   SQLite Database      │  │
                        │  │   + Alerts + Reports   │  │
                        │  └────────────────────────┘  │
                        └──────────────────────────────┘
```

---

## 📊 Example Application Areas (PS2)

Our system can address the following competition application areas:

| Application Area | Our Implementation |
|-----------------|-------------------|
| **♻️ Household Recycling** | Camera scans household items → Llama explains if/how to recycle |
| **📱 E-waste Management** | Detect electronics → Llama advises on certified disposal centres |
| **🏘️ Community Recycling** | Bin monitoring (sensors) → alerts when full → route optimization |
| **🛍️ Plastic Waste Reduction** | Plastic detection (YOLOv8) → Llama provides alternatives & impact data |
| **📚 Sustainability Education** | Chat/voice interface → Llama teaches waste management practices |

---

## 📦 Deliverables (Competition Requirements)

| Deliverable | Description | Status |
|-------------|-------------|--------|
| **Source Code** | Full project on GitHub — all modules, models, configs | 🟢 In progress |
| **Application or Interface** | Working demo on Arm device (Android app / camera assistant / chat) | 🟡 Pending Llama integration |
| **Demonstration Video** | End-to-end demo showing waste detection → Llama advisory → disposal guidance | 🔴 Not yet created |
| **Technical Report** | Architecture, model choices, benchmarks, power/latency analysis | 🔴 Not yet written |

---

## 🔌 Pin Connections (Raspberry Pi Pico W — Sensor Array)

| Sensor | Sensor Pin | Pico Pin | Function |
|--------|-----------|----------|----------|
| HC-SR04 | VCC | 5V (Pin 40) | Power |
| HC-SR04 | GND | GND | Ground |
| HC-SR04 | TRIG | GP3 | Trigger pulse |
| HC-SR04 | ECHO | GP2 | Echo return |
| DHT11 | VCC | 3.3V | Power |
| DHT11 | GND | GND | Ground |
| DHT11 | DATA | GP15 | Temp & humidity data |
| MQ Gas | VCC | 5V | Power |
| MQ Gas | GND | GND | Ground |
| MQ Gas | AO | GP26 (ADC0) | Analog gas level |
| MQ Gas | DO | GP16 | Digital threshold |
| HW-103 | VCC | 5V | Power |
| HW-103 | GND | GND | Ground |
| HW-103 | AO | GP27 (ADC1) | Analog moisture level |

---

## 🚀 Quick Start

### 1. Pico W Setup
Upload `pico_sensors/main.py` to the Pico W using Thonny or mpremote.

### 2. Device Setup (Raspberry Pi 5 / Arm Linux)
```bash
cd module
pip install -r requirements.txt
python main.py
```

### 3. Run Modes
```bash
python main.py                          # Normal operation (camera + sensors)
python main.py --strict                 # Production — abort if Pico not connected
python main.py --test --headless        # Self-test without hardware
python main.py --image path/to/photo    # Analyse a single waste photo
python main.py --serial /dev/ttyUSB0    # Override serial port
```

---

## 🗂️ Waste Categories Detected

| Category | Detection Method | Examples |
|----------|-----------------|----------|
| 🛢️ **Plastic** | YOLOv8 vision | Bottles, bags, containers, packaging |
| ♻️ **Recyclable** | YOLOv8 vision | Paper, cardboard, glass |
| 🔩 **Metal** | YOLOv8 vision | Cans, foil, metal objects |
| 🍂 **Dry Waste** | Vision + moisture sensor | Non-recyclable dry materials |
| 💧 **Wet Waste** | Vision + humidity/moisture sensors | Food waste, organic matter |
| ☠️ **Hazardous** | Gas sensor + vision | High gas levels, toxic waste, chemicals |
| 📱 **E-waste** | YOLOv8 vision (planned) | Electronics, batteries, cables |

---

## 🧠 Llama + ExecuTorch Integration Plan

### Model Selection
| Model | Parameters | Use Case | RAM Required |
|-------|-----------|----------|-------------|
| **Llama 3.2 1B-Instruct** | 1B | Lightweight advisory on RPi 5 | ~2 GB |
| **Llama 3.2 3B-Instruct** | 3B | Richer disposal guidance | ~6 GB |
| Fine-tuned/distilled variant | <1B | Optimized for waste domain | <2 GB |

### ExecuTorch Deployment Pipeline
```
Llama 3.2 (HuggingFace)
    │
    ▼
torch.export() → ExportedProgram
    │
    ▼
ExecuTorch lowering (XNNPACK / Arm backend)
    │
    ▼
.pte file (on-device runtime)
    │
    ▼
Arm CPU inference (RPi 5 / Android)
```

### Llama Advisory Capabilities (to be built)
- **Waste Disposal Guidance** — "This is a PET plastic bottle. Rinse and place in blue recycling bin."
- **Recycling Education** — "PET #1 plastic is widely recyclable. It can be remade into polyester fibre."
- **Environmental Impact** — "Recycling this bottle saves 0.5 kg CO₂ vs. landfill."
- **Local Regulations** — Region-specific disposal rules and facility locations
- **Sustainability Tips** — Waste reduction, alternatives, composting guidance

---

## 🔄 Migration: Jetson Nano → Arm (RPi 5)

| Component | Current (Jetson Nano) | Target (Arm RPi 5) | Action |
|-----------|----------------------|---------------------|--------|
| AI Inference | PyTorch + CUDA | ExecuTorch + XNNPACK | Re-export models |
| YOLOv8 | ultralytics (GPU) | ultralytics (CPU) or HailoRT (NPU) | Benchmark & choose |
| Camera | OpenCV (USB) | OpenCV (USB) or PiCamera2 | Minimal change |
| Sensors | Pico W → USB serial | Pico W → USB serial | No change |
| Database | SQLite | SQLite | No change |
| LLM | None | Llama 3.2 via ExecuTorch | **New component** |

---

## 🏋️ Training Custom Model

```bash
python train_waste_model.py --data yolo/training_data/waste/data.yaml --epochs 100
```

Uses TrashNet, TACO, and WasteNet datasets for fine-tuning YOLOv8n.
