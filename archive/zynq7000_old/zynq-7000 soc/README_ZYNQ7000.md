# C2S Smart Waste AI System — Zynq-7000 SoC (ZedBoard)

Target Hardware: **Avnet / Digilent ZedBoard (Xilinx Zynq-7000 XC7Z020-CLG484-1)**  
Memory Configuration: **512 MB Physical DDR3 + 8 GB Attached External Memory**

---

## 1. System & Memory Hierarchy

The Zynq-7000 on the ZedBoard combines a Dual-Core ARM Cortex-A9 Processing System (PS) with 85K Logic Cells of Programmable Logic (PL). With an **8 GB external storage/memory** attached (via SD card ext4 or USB OTG storage), your system solves the embedded memory bottleneck.

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
|  |  Critical loop execution  |             |  Zero-copy sensor/telemetry buff  |  |
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
|  - YOLOv8 Custom Weights (best.pt ~22MB, yolov8n.pt ~6.5MB)                       |
|  - Llama 3.2 1B Instruct GGUF Model (~807 MB)                                     |
|  - SQLite Persistent Database (waste_monitoring.db)                               |
|  - Full-Stack Web Dashboard (HTML5, Vanilla CSS, JS, static media)               |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Layout: `bram` vs `external`

```
zynq-7000 soc/
├── bram/                             <-- FAST / LOW-FOOTPRINT CONTROL LOGIC (OCM/BRAM)
│   ├── config_zynq.py                <-- ZedBoard configuration (I/O, paths, 8GB mounts)
│   ├── main_zynq.py                  <-- Autonomous embedded entry point with LED monitor
│   ├── bram_direct_io.py             <-- Direct mmap interface to 256KB PL BRAM (0x40000000)
│   ├── alert_system.py               <-- Gas, fill level, hazardous alert rules
│   ├── data_fusion.py                <-- Multi-sensor fusion engine (Pico W UART + AI)
│   ├── database.py                   <-- SQLite abstraction layer
│   ├── webcam_capture.py             <-- UVC camera capture & buffer manager
│   ├── report_generator.py           <-- Periodic JSON/CSV audit reports
│   ├── detect_plastic.py             <-- Batch CLI detection utility
│   └── requirements.txt              <-- Embedded dependency list
│
├── external/                         <-- STORED IN 8 GB EXTERNAL MEMORY (/mnt/ext_storage)
│   ├── waste_detector.py             <-- YOLO inference coordinator
│   ├── waste_advisor.py              <-- 2,300-line waste domain knowledge base
│   ├── llama_engine.py               <-- Llama 3.2 GGUF inference engine
│   ├── executorch_llama_engine.py    <-- ExecuTorch ARM SoC runner
│   ├── web_app.py                    <-- Flask REST API & telemetry web server
│   ├── train_waste_model.py          <-- Model fine-tuning script
│   ├── download_model.py             <-- Model download tool
│   ├── models/
│   │   ├── best.pt                   <-- 5-class YOLOv8s custom weights (22.5 MB)
│   │   └── yolov8n.pt                <-- Generic YOLOv8n weights (6.5 MB)
│   ├── database/                     <-- Persistent SQLite database location
│   ├── pico_sensors/                 <-- MicroPython firmware for Pico W sensors
│   │   └── main.py
│   └── web/                          <-- Web UI templates, CSS, JS, images
│
└── vivado/                           <-- HARDWARE DESIGN & BOARD AUTOMATION
    ├── zedboard_block_design.tcl     <-- Vivado TCL to build PS7 + BRAM + GPIO bitstream
    ├── zedboard_constraints.xdc      <-- Pin constraints for ZedBoard (100MHz, LEDs, SW)
    ├── deploy_to_zedboard.sh         <-- Automated SSH deploy & 2GB swap creator
    └── c2s-waste-zedboard.service    <-- Systemd service for auto-start on boot
```

---

## 3. Vivado Hardware Creation (Step-by-Step)

### Step 1: Open Vivado and Run the Block Design Script
1. Open **Xilinx Vivado (2020.2, 2022.x, or 2023.x)**.
2. In the Vivado Tcl Console, change directory to the `vivado` folder:
   ```tcl
   cd "c:/Users/kumar/OneDrive/C2S/zynq-7000 soc/vivado"
   source zedboard_block_design.tcl
   ```
3. The script automatically:
   - Configures the Zynq-7000 XC7Z020-CLG484-1 device
   - Sets up the ZedBoard Processing System (DDR3, UART0/1, USB0, SD0, ENET0)
   - Instantiates an **AXI BRAM Controller** and **256 KB Block Memory Generator** mapped at `0x40000000`
   - Instantiates **AXI GPIO** connected to the 8 User LEDs (`LD0-LD7`) and 8 Switches (`SW0-SW7`)
   - Links the `zedboard_constraints.xdc` file

### Step 2: Generate Bitstream & Export Hardware Platform
1. In Vivado GUI, click **Generate Bitstream** (or in TCL: `launch_runs impl_1 -to_step write_bitstream -jobs 4`).
2. After bitstream generation completes, export the hardware:
   `File -> Export -> Export Hardware...`
3. Select **Include Bitstream** and export `c2s_zedboard_hw.xsa`.

---

## 4. Setting up PetaLinux / SD Card with 8GB External Storage

### Partitioning the 8GB External Storage (SD Card or USB)
Partition the 8GB drive into two partitions:
1. **Partition 1 (BOOT, FAT32, ~500 MB):**
   - Holds `BOOT.BIN`, `boot.scr`, and `image.ub`
2. **Partition 2 (ROOTFS, ext4, ~7.5 GB):**
   - Extracted PetaLinux root filesystem
   - Houses `/mnt/ext_storage` for your models, database, and swapfile

```bash
# On your Linux host or PetaLinux workstation:
sudo tar -xzf rootfs.tar.gz -C /media/$USER/ROOTFS/
```

---

## 5. Automated Deployment to ZedBoard

Once the ZedBoard boots and acquires an IP address on your network (via Ethernet J2):

```bash
# From your PC (Git Bash, WSL, or Linux shell):
cd "c:/Users/kumar/OneDrive/C2S/zynq-7000 soc/vivado"
chmod +x deploy_to_zedboard.sh
./deploy_to_zedboard.sh 192.168.1.120
```

### What `deploy_to_zedboard.sh` Does:
1. Creates directory `/opt/c2s` and `/mnt/ext_storage`.
2. **Creates a 2 GB swapfile** on your 8 GB storage (`/mnt/ext_storage/swapfile`). This is critical: it prevents Out-Of-Memory errors when running YOLO and Llama on the 512MB RAM!
3. Copies `bram/` modules to `/opt/c2s/`.
4. Copies `external/` modules, models (`best.pt`, `yolov8n.pt`), and web assets to `/mnt/ext_storage/`.
5. Installs the systemd auto-start service.

---

## 6. Running the System on ZedBoard

### Option A: Autonomous AI Detection Loop
```bash
ssh root@<ZEDBOARD_IP>
cd /opt/c2s
python3 main_zynq.py
```
- Heartbeat LED (`LD0`) pulses to indicate the A9 processing loop is active.
- Detects waste objects, ingests Pico W telemetry, executes fusion, and stores records in SQLite on external memory.

### Option B: Interactive Web Dashboard (Flask)
```bash
cd /mnt/ext_storage
python3 web_app.py
```
Open your browser and navigate to:
```
http://<ZEDBOARD_IP>:5000
```
- Real-time webcam streaming
- Detection analytics & material breakdown
- Waste disposal AI chatbot assistant
- Historical database records and alerts

---

## 7. ZedBoard Physical Hardware Connections

| Connector | Label on Board | Function | Physical Peripheral |
|---|---|---|---|
| **J13** | `USB OTG` | USB Host / Camera | USB UVC Webcam (`/dev/video0`) |
| **J14** | `USB UART` | Console (115200) | Micro-USB cable to PC for Linux shell |
| **J7** | `USB Host` or Hub | Pico Sensor UART | Raspberry Pi Pico W (`/dev/ttyACM0` or `/dev/ttyUSB0`) |
| **J2** | `ETH` | Gigabit Ethernet | RJ-45 cable to LAN/Router for Web UI |
| **J8** | `SD` | SDIO Card Slot | 8 GB Boot + External Storage SD Card |
| **LD0-LD7** | `LD0 .. LD7` | PL User LEDs | Heartbeat, Detection Status, Alert Flags |
| **SW0-SW7** | `SW0 .. SW7` | PL User Switches | Runtime mode selection |

