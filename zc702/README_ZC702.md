# ZC702 Deployment — C2S Smart Waste AI System

## Overview

This folder contains everything needed to deploy the **AI-Enabled Smart Waste Segregation & Monitoring System** on the **Xilinx ZC702 Evaluation Board (Zynq-7000 XC7Z020-CLG484-1)** using **Vivado 2023.x / PetaLinux**.

The ZC702 runs the AI pipeline on its **dual ARM Cortex-A9** Processing System (PS), with the **Programmable Logic (PL)** fabric providing AXI GPIO and optional BRAM staging. YOLOv8 inference, LLM responses, and the Flask web dashboard all run in software on the PS Linux.

---

## Folder Structure

```
zc702/
├── bram/                      ← Files mapped to BRAM / On-Chip Memory (OCM)
│   ├── config_zc702.py        ← ZC702-adapted system configuration
│   ├── config.py              ← Original config (reference copy)
│   ├── main_zc702.py          ← ZC702 entry point (headless, /dev/ttyPS0)
│   ├── main.py                ← Original main (reference copy)
│   ├── alert_system.py        ← Alert evaluator (small, fits in OCM)
│   ├── data_fusion.py         ← Sensor fusion engine
│   ├── database.py            ← SQLite abstraction layer
│   ├── webcam_capture.py      ← OpenCV camera wrapper
│   ├── report_generator.py    ← JSON/CSV report generator
│   ├── detect_plastic.py      ← CLI detection utility
│   └── requirements.txt       ← Python dependencies
│
├── external/                  ← Files stored in external DDR3 (1 GB)
│   ├── waste_detector.py      ← YOLO detection engine (63 KB Python)
│   ├── waste_advisor.py       ← 2300-line knowledge base (135 KB)
│   ├── llama_engine.py        ← Llama 3.2 1B inference (27 KB)
│   ├── executorch_llama_engine.py  ← ExecuTorch Arm SoC runtime
│   ├── train_waste_model.py   ← Model training script
│   ├── download_model.py      ← Llama model downloader
│   ├── models/
│   │   ├── best.pt            ← YOLOv8s 5-class waste model (22 MB)
│   │   └── yolov8n.pt         ← YOLOv8n general model (6 MB)
│   ├── pico_sensors/          ← MicroPython firmware for Pico W
│   │   └── main.py
│   └── web/                   ← Flask web dashboard assets
│       ├── templates/
│       │   └── index.html
│       └── static/
│           ├── css/style.css
│           ├── js/app.js
│           └── images/
│
└── vivado/                    ← Vivado project files
    ├── zc702_block_design.tcl ← TCL script to build the block design
    ├── zc702_constraints.xdc  ← Pin/timing constraints for PL
    ├── deploy_to_zc702.sh     ← SSH deployment script
    ├── petalinux_recipe.bb    ← Yocto/PetaLinux recipe
    └── c2s-waste.service      ← systemd auto-start service
```

---

## Memory Allocation Strategy

| Region | Size | What lives here |
|--------|------|-----------------|
| **PS OCM (On-Chip RAM)** | 256 KB | Python interpreter boot, config, alert, fusion, database, main |
| **PL BRAM** | 630 KB (140×RAMB36) | AXI-GPIO registers, DMA descriptors, frame staging |
| **DDR3 (external)** | 1 GB | Python runtime, YOLO models, LLM, SQLite DB, web assets, logs |

> **ZC702 BRAM total (PL):** 140 × RAMB36E1 = 630 KB  
> **PS OCM:** 4 × 64 KB banks = 256 KB  
> **Files in `bram/`** total ≈ 96 KB — fits comfortably in PS OCM  
> **Files in `external/`** total ≈ 29 MB — stored in DDR3 on boot

---

## Step-by-Step: Build & Deploy with Vivado

### Step 1 — Build the Vivado Block Design

1. Open **Vivado 2023.x**
2. Open the TCL console: `Window → Tcl Console`
3. Run the block design script:
   ```tcl
   cd C:/Users/kumar/OneDrive/C2S/zc702/vivado
   source zc702_block_design.tcl
   ```
4. Wait for synthesis + implementation to finish (≈20–40 minutes)
5. The script auto-generates the bitstream and exports `c2s_zc702.xsa`

### Step 2 — Build PetaLinux Image

```bash
# On Linux host with PetaLinux 2023.x installed
petalinux-create -t project --template zynq -n c2s_zc702
cd c2s_zc702
petalinux-config --get-hw-description=/path/to/c2s_zc702.xsa

# Enable Python3, OpenCV, SSH, USB camera support
petalinux-config -c rootfs
# → Filesystem Packages → misc → python3
# → Filesystem Packages → misc → uvc-gadget (USB camera)
# → user packages → c2s-waste-system

petalinux-build
petalinux-package --boot --fsbl --fpga --u-boot --force
```

### Step 3 — Prepare SD Card

| SD Card Partition | Format | Contents |
|-------------------|--------|----------|
| Partition 1 (FAT32, 512MB) | FAT32 | `BOOT.BIN`, `boot.scr`, `image.ub` |
| Partition 2 (ext4, rest) | ext4 | `rootfs.tar.gz` extracted here |

```bash
# Extract rootfs (on Linux host)
sudo tar xf images/linux/rootfs.tar.gz -C /mnt/rootfs/
```

### Step 4 — Deploy Application Files

```bash
# Boot ZC702 with SD card, find its IP address, then:
chmod +x deploy_to_zc702.sh
./deploy_to_zc702.sh 192.168.1.100
```

This script copies:
- `bram/*.py` → `/opt/c2s/` (PS OCM-staged code)
- `external/*.py` → `/opt/c2s/` (DDR3 application code)
- `external/models/` → `/mnt/models/` (DDR3 model storage)
- `external/web/` → `/opt/c2s/web/` (DDR3 web assets)

### Step 5 — Run the System

```bash
ssh root@<zc702-ip>
cd /opt/c2s

# Start the AI pipeline (headless, production mode)
python3 main_zc702.py --headless

# Or enable web dashboard (Flask on port 5000):
python3 web_app.py
```

Open browser: `http://<zc702-ip>:5000`

### Step 6 — Enable Auto-Start (systemd)

```bash
# On ZC702:
cp /opt/c2s/c2s-waste.service /etc/systemd/system/
systemctl enable c2s-waste.service
systemctl start  c2s-waste.service
```

---

## Hardware Connections

| ZC702 Connector | Signal | Peripheral |
|-----------------|--------|------------|
| J7 (USB Type-A) | USB OTG | USB UVC Webcam (`/dev/video0`) |
| J58 MIO49 | UART1 RX | Pico W TX (115200 baud) |
| J58 GND | GND | Pico W GND |
| J2 (Ethernet) | ETH0 | Network / web dashboard |
| J7 SD card slot | SD0 | SD card (boot + rootfs) |
| DS15–DS18 (LEDs) | GPIO | Status indicators |
| SW10 (DIP) | GPIO | Mode select (headless / strict) |

---

## Performance on ZC702 (Cortex-A9 @ 667 MHz)

| Task | Expected Performance |
|------|---------------------|
| YOLOv8n inference (416×416) | ~800–1200 ms/frame |
| YOLOv8s best.pt (416×416) | ~1500–2000 ms/frame |
| Sensor fusion + DB write | ~5 ms |
| Llama 3.2 1B (GGUF Q4) | ~2–5 tokens/sec |
| Flask web dashboard | ~10 FPS MJPEG stream |

> **Tip:** Set `input_size = 320` in `config_zc702.py` for faster inference at the cost of accuracy.

---

## Files That Need Manual Transfer to ZC702 (Not in This Repo)

These large files must be downloaded separately and placed in `/mnt/models/` on the ZC702:

| File | Size | Source |
|------|------|--------|
| `llama-3.2-1b-instruct-q4_k_m.gguf` | ~807 MB | Run `python3 download_model.py` on a PC |
| `llama3_2_1b.pte` | ~600 MB | ExecuTorch build (optional) |
| `tokenizer.model` | ~500 KB | Bundled with GGUF download |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Check `ls /dev/video*`, ensure UVC module loaded: `modprobe uvcvideo` |
| Pico W serial not found | Check `ls /dev/ttyPS*`, ensure UART1 enabled in PS7 config |
| YOLO model load fail | Confirm `/mnt/models/best.pt` exists and is readable |
| Out of memory | Use `yolov8n.pt` (6MB) instead of `best.pt` (22MB), set `input_size=320` |
| Flask not accessible | Check firewall: `iptables -I INPUT -p tcp --dport 5000 -j ACCEPT` |
| systemd service fails | Check logs: `journalctl -u c2s-waste.service -f` |

