#!/bin/bash
# ============================================================================
# ZC702 PetaLinux Application Deployment Script
# AI-Enabled Smart Waste Segregation & Monitoring System
# Target: Xilinx ZC702 / Zynq-7000 XC7Z020 running PetaLinux
# ============================================================================
# USAGE (run on host machine after PetaLinux build):
#   chmod +x deploy_to_zc702.sh
#   ./deploy_to_zc702.sh <zc702-ip-address>
#
# PREREQUISITES:
#   - ZC702 booted with PetaLinux image
#   - SSH enabled on ZC702 (root@<ip>)
#   - SD card formatted with ext4 rootfs partition
# ============================================================================

set -e
ZC702_IP="${1:-192.168.1.100}"
ZC702_USER="root"
APP_DIR="$(dirname "$0")/.."

echo "========================================"
echo "  ZC702 C2S Deployment Script"
echo "  Target: ${ZC702_USER}@${ZC702_IP}"
echo "========================================"

# ── 1. Create target directories on ZC702 ────────────────────────────────────
echo "[1/6] Creating directories on ZC702..."
ssh ${ZC702_USER}@${ZC702_IP} "
    mkdir -p /opt/c2s
    mkdir -p /mnt/models
    mkdir -p /mnt/data/database
    mkdir -p /mnt/data/logs
    mkdir -p /mnt/data/reports
"

# ── 2. Upload BRAM files (lightweight Python modules) ─────────────────────────
echo "[2/6] Uploading BRAM files (lightweight modules) → /opt/c2s/ ..."
scp ${APP_DIR}/bram/*.py ${ZC702_USER}@${ZC702_IP}:/opt/c2s/

# ── 3. Upload External DDR3 files (large Python modules) ─────────────────────
echo "[3/6] Uploading External files (large modules) → /opt/c2s/ ..."
scp ${APP_DIR}/external/*.py ${ZC702_USER}@${ZC702_IP}:/opt/c2s/

# ── 4. Upload web assets → /opt/c2s/web/ ─────────────────────────────────────
echo "[4/6] Uploading web assets → /opt/c2s/web/ ..."
ssh ${ZC702_USER}@${ZC702_IP} "mkdir -p /opt/c2s/web"
scp -r ${APP_DIR}/external/web/* ${ZC702_USER}@${ZC702_IP}:/opt/c2s/web/

# ── 5. Upload YOLO models → /mnt/models/ (DDR3 side) ─────────────────────────
echo "[5/6] Uploading YOLO models → /mnt/models/ ..."
echo "    NOTE: best.pt is 22MB, yolov8n.pt is 6MB. This may take a minute."
scp ${APP_DIR}/external/models/yolov8n.pt ${ZC702_USER}@${ZC702_IP}:/mnt/models/
scp ${APP_DIR}/external/models/best.pt    ${ZC702_USER}@${ZC702_IP}:/mnt/models/

# ── 6. Upload Pico W firmware ─────────────────────────────────────────────────
echo "[6/6] Uploading Pico W firmware → /opt/c2s/pico_sensors/ ..."
ssh ${ZC702_USER}@${ZC702_IP} "mkdir -p /opt/c2s/pico_sensors"
scp -r ${APP_DIR}/external/pico_sensors/* ${ZC702_USER}@${ZC702_IP}:/opt/c2s/pico_sensors/

# ── Install Python dependencies on ZC702 ─────────────────────────────────────
echo "[+] Installing Python packages on ZC702 (requires internet on target)..."
ssh ${ZC702_USER}@${ZC702_IP} "
    pip3 install --upgrade pip
    pip3 install opencv-python-headless==4.8.1.78 \
                 ultralytics==8.3.0 \
                 flask==3.0.0 \
                 pyserial==3.5 \
                 numpy==1.26.4 \
                 pillow==10.3.0 \
                 llama-cpp-python==0.2.90 \
                 --extra-index-url https://www.piwheels.org/simple
" || echo "WARNING: pip install failed — install manually on target."

echo ""
echo "========================================"
echo "  DEPLOYMENT COMPLETE!"
echo "========================================"
echo ""
echo "  To run the system on ZC702:"
echo "    ssh root@${ZC702_IP}"
echo "    cd /opt/c2s"
echo "    python3 main.py --headless --config config_zc702.py"
echo ""
echo "  Web dashboard (after starting web_app.py):"
echo "    http://${ZC702_IP}:5000"
echo ""

