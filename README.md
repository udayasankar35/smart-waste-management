# ♻️ AI-Enabled Smart Waste Segregation & Monitoring System (Zynq-7000 SoC)

[![Platform: Zynq-7000](https://img.shields.io/badge/Platform-Xilinx%20Zynq--7000%20(ZedBoard)-orange?logo=xilinx)](https://www.xilinx.com/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/udayasankar35/smart-waste-management)
[![AI Model: YOLOv8](https://img.shields.io/badge/Vision-YOLOv8%20Custom-blue?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![LLM: Llama 3.2 1B](https://img.shields.io/badge/LLM-Llama%203.2%201B%20Instruct-purple?logo=meta)](https://huggingface.co/meta-llama)
[![OS: PetaLinux / Embedded Linux](https://img.shields.io/badge/OS-PetaLinux%20%2F%20Embedded%20Linux-green?logo=linux)](https://www.xilinx.com/products/design-tools/embedded-software/petalinux-sdk.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-sensor waste classification and bin monitoring platform designed for edge deployment on the **Xilinx Zynq-7000 SoC (Avnet ZedBoard XC7Z020)**.

The system couples **YOLOv8 computer vision** with **IoT multi-sensor telemetry** (fill level, humidity, temperature, toxic gases, moisture) streamed from a Raspberry Pi Pico W, backed by an **on-device Llama 3.2 1B LLM** and an expert rule engine for instant recycling guidance.

---

## 🌟 Key Highlights

- **Dual-Domain Edge Intelligence:**
  - **Vision:** Custom-trained YOLOv8 for 5 waste classes (*plastics, glass, metals, paper/cardboard, organics*).
  - **Sensors:** Hardware sensor fusion analyzing fill percentage (HC-SR04), environment (DHT11), hazardous gas surge (MQ sensor), and wet/dry status (soil moisture).
- **Tiered Memory Architecture (BRAM vs. 8 GB External Storage):**
  - High-priority, zero-latency control and alerting loops execute in **On-Chip Memory (OCM)** and **AXI Block RAM (PL BRAM)**.
  - Heavy vision weights, SQLite database, full-stack web dashboard, and LLM engines execute from an attached **8 GB external storage** configured with a **2 GB virtual swap partition**.
- **On-Device LLM & Advisory Engine:**
  - Integrated on-device Llama 3.2 1B Instruct engine coupled with a 2,300-line domain expert knowledge base (`WasteAdvisor`) delivering sub-5ms disposal recommendations.
- **Full-Stack Web & HUD Interfaces:**
  - Embedded Flask application providing real-time MJPEG camera streaming, historical analytics, safety alerts, and a conversational chatbot interface.

---

## 🏗️ Hardware Architecture & Memory Hierarchy

```
+-----------------------------------------------------------------------------------+
|                              ZedBoard Zynq-7000 SoC                               |
|                                                                                   |
|  +---------------------------+             +-----------------------------------+  |
|  |     Processing System     |             |         Programmable Logic        |  |
|  |  Dual ARM Cortex-A9 CPU   |             |  140 BRAMs (RAMB36) = 630 KB      |  |
|  |                           |  AXI-GP0    |                                   |  |
|  |  [PS OCM: 256 KB]         |<----------->|  [AXI BRAM Controller: 256 KB]    |  |
|  |  Zero-latency control     |             |  Address: 0x40000000              |  |
|  |  Critical loop execution  |             |  Zero-copy telemetry buffer       |  |
|  +---------------------------+             +-----------------------------------+  |
|                |                                             |                    |
|                | Hard Memory Controller                      |                    |
|                v                                             v                    |
|  +---------------------------+             +-----------------------------------+  |
|  |  Onboard DDR3 RAM: 512 MB |             |  AXI GPIO: LEDs (LD0-LD7)         |  |
|  |  Active OS & Frame Buffer |             |            Switches (SW0-SW7)     |  |
|  +---------------------------+             +-----------------------------------+  |
+-----------------------------------------------------------------------------------+
                                 |
                                 | SD Card (SDIO) / USB OTG
                                 v
+-----------------------------------------------------------------------------------+
|                        8 GB External Memory Storage                               |
|  - 2 GB Virtual Swapfile (/mnt/ext_storage/swapfile -> enables Llama & PyTorch)   |
|  - YOLOv8 Custom Weights (best.pt ~22.5 MB, yolov8n.pt ~6.5 MB)                   |
|  - Llama 3.2 1B Instruct GGUF Model (~807 MB)                                     |
|  - SQLite Persistent Database (waste_monitoring.db)                               |
|  - Full-Stack Web Dashboard (HTML5, CSS, JS, Media)                               |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
├── zynq-7000 soc/
│   ├── bram/                           # Low-footprint, time-critical modules (OCM / BRAM)
│   │   ├── config_zynq.py              # System configuration for ZedBoard paths & peripherals
│   │   ├── main_zynq.py                # Autonomous headless monitoring loop with LED signals
│   │   ├── bram_direct_io.py           # Direct mmap interface to physical 256KB PL BRAM
│   │   ├── data_fusion.py              # Correlates AI vision with Pico W sensor inputs
│   │   ├── alert_system.py             # Rule-based safety & overflow threshold evaluator
│   │   ├── database.py                 # SQLite database storage & query interface
│   │   ├── webcam_capture.py           # USB UVC camera grabber & reconnect logic
│   │   ├── report_generator.py         # Automated audit reporting (JSON/CSV)
│   │   ├── detect_plastic.py           # Standalone batch image CLI detector
│   │   └── requirements.txt            # Python dependencies for target ARM Linux
│   │
│   ├── external/                       # Heavy models, web assets, and runtime storage
│   │   ├── waste_detector.py           # Dual-model YOLO inference coordinator
│   │   ├── waste_advisor.py            # 2,300-line waste domain knowledge engine
│   │   ├── llama_engine.py             # Llama 3.2 GGUF & ExecuTorch inference engine
│   │   ├── web_app.py                  # Full-stack Flask REST API & Web Dashboard
│   │   ├── models/
│   │   │   ├── best.pt                 # 5-class YOLOv8s trained waste weights (22.5 MB)
│   │   │   └── yolov8n.pt              # General YOLOv8n weights (6.5 MB)
│   │   ├── pico_sensors/main.py        # MicroPython firmware for Raspberry Pi Pico W
│   │   └── web/                        # Web UI (HTML templates, CSS, JS, static media)
│   │
│   └── vivado/                         # Vivado automation & constraints
│       ├── zedboard_block_design.tcl   # TCL script: builds PS7 + 256KB BRAM + AXI-GPIO
│       ├── zedboard_constraints.xdc    # Master XDC file for ZedBoard (100MHz clock, LEDs, SW)
│       ├── deploy_to_zedboard.sh       # Automated SSH deploy & 2GB swap creator script
│       └── c2s-waste-zedboard.service  # systemd service unit for startup on boot
```

---

## 🔌 Hardware Connections (ZedBoard)

| Connector | Label on Board | Interface | Connected Device |
|:---|:---|:---|:---|
| **J13** | `USB OTG` | USB Host | USB UVC Webcam (`/dev/video0`) via OTG adapter |
| **J14** | `USB UART` | Serial Console | Micro-USB to PC for terminal (115200 baud) |
| **J7** | `USB Host` / Hub | Serial / USB | Raspberry Pi Pico W telemetry (`/dev/ttyACM0`) |
| **J2** | `ETH` | RJ-45 | Gigabit Ethernet to LAN (Web Dashboard @ port 5000) |
| **J8** | `SD` | SDIO | 8 GB Boot + External Storage SD Card |
| **LD0–LD7** | `LD0 .. LD7` | PL User LEDs | Heartbeat, detection indicators, and critical alarms |
| **SW0–SW7** | `SW0 .. SW7` | PL Switches | Runtime configuration switches (live/sim, logging) |

---

## 🚀 Quick Start Guide

### 1. Build Vivado Hardware (Bitstream & XSA)
Open **Vivado** (2020.2 or later), open the Tcl Console, and run:
```tcl
cd "zynq-7000 soc/vivado"
source zedboard_block_design.tcl
```
1. Click **Generate Bitstream**.
2. Export hardware with bitstream: `File -> Export -> Export Hardware...` as `c2s_zedboard_hw.xsa`.

### 2. Format 8 GB External Storage / SD Card
- **Partition 1 (BOOT, FAT32, ~500 MB):** Place `BOOT.BIN`, `boot.scr`, and `image.ub`.
- **Partition 2 (ROOTFS, ext4, ~7.5 GB):** Extract PetaLinux rootfs.

### 3. Automated One-Command Deployment
Ensure your ZedBoard is connected to your local network, then run from your workstation:
```bash
cd "zynq-7000 soc/vivado"
chmod +x deploy_to_zedboard.sh
./deploy_to_zedboard.sh <ZEDBOARD_IP>
```
*This automatically creates a 2 GB swapfile on your 8 GB storage, mounts directories, copies all BRAM and External files, and configures systemd.*

### 4. Running the System
SSH into the ZedBoard:
```bash
ssh root@<ZEDBOARD_IP>
cd /opt/c2s

# Run autonomous detection loop:
python3 main_zynq.py

# Or start the Web Dashboard:
python3 /mnt/ext_storage/web_app.py
```
Access the dashboard at `http://<ZEDBOARD_IP>:5000`.

---

## ☁️ Deploy to Vercel (1-Click Cloud Web App)

The repository includes a ready-to-deploy serverless configuration (`vercel.json` + `api/index.py`) for hosting the interactive web dashboard on Vercel:

1. Go to [Vercel Dashboard](https://vercel.com/new).
2. Click **Add New...** -> **Project**.
3. Import your repository: `udayasankar35/smart-waste-management`.
4. Leave all settings at default (Root Directory: `./`, Framework Preset: **Other**).
5. Click **Deploy**.

Vercel will automatically build the `@vercel/python` serverless function and assign a live production URL (e.g., `https://smart-waste-management.vercel.app`).

---

## 📊 Performance on ARM Cortex-A9 (667 MHz)

| Pipeline Component | Metric | Performance |
|:---|:---|:---|
| **YOLOv8n Inference (416×416)** | Latency | ~800–1100 ms / frame |
| **YOLOv8s best.pt (416×416)** | Latency | ~1400–1800 ms / frame |
| **Sensor Fusion & DB Commit** | Execution Time | < 5 ms |
| **WasteAdvisor Rule Engine** | Response Time | < 2 ms |
| **Llama 3.2 1B (Q4 GGUF)** | Generation Speed | ~2–4 tokens / sec |
| **Live MJPEG Video Feed** | Stream Frame Rate | ~10–15 FPS |

---

## 📜 License & Acknowledgments

This project is released under the [MIT License](LICENSE).  
Developed for the **Bharat AI-SoC Challenge (Problem Statement 2: AI-Powered Waste Classification & Disposal Assistant)**, supported by Arm, MeitY C2S Programme, and IIT Delhi.
