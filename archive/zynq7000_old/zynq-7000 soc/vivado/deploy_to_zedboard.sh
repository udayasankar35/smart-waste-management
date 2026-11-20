#!/bin/bash
# ==============================================================================
# ZedBoard (Zynq-7000) Automated Deployment & Setup Script
# AI-Enabled Smart Waste Segregation & Monitoring System
#
# Target: Avnet / Digilent ZedBoard (XC7Z020)
# Memory: 512 MB Physical DDR3 + 8 GB External Storage (/mnt/ext_storage)
# ==============================================================================
# Usage:
#   chmod +x deploy_to_zedboard.sh
#   ./deploy_to_zedboard.sh <ZEDBOARD_IP>
# Example:
#   ./deploy_to_zedboard.sh 192.168.1.120
# ==============================================================================

set -e

ZED_IP="${1:-192.168.1.120}"
ZED_USER="root"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."

echo "=================================================================="
echo "  Deploying C2S Waste System to ZedBoard (${ZED_IP})"
echo "  Attached Storage: 8 GB External Memory"
echo "=================================================================="

# ── 1. Create Directories on ZedBoard ─────────────────────────────────────────
echo "[1/6] Preparing filesystems on ZedBoard..."
ssh ${ZED_USER}@${ZED_IP} "
    # Primary fast runtime directory (fits in OCM / rootfs)
    mkdir -p /opt/c2s

    # 8 GB External Memory Mount (SD card ext4 or USB)
    mkdir -p /mnt/ext_storage/models
    mkdir -p /mnt/ext_storage/database
    mkdir -p /mnt/ext_storage/logs
    mkdir -p /mnt/ext_storage/reports
    mkdir -p /mnt/ext_storage/web

    # Ensure link from /opt/c2s/external points to external storage
    ln -sfn /mnt/ext_storage /opt/c2s/external
"

# ── 2. Configure 2 GB Swap Space on the 8 GB External Memory ─────────────────
echo "[2/6] Configuring 2 GB virtual swap on 8 GB external memory..."
ssh ${ZED_USER}@${ZED_IP} "
    if [ ! -f /mnt/ext_storage/swapfile ]; then
        echo 'Creating 2GB swapfile on 8GB external memory...'
        dd if=/dev/zero of=/mnt/ext_storage/swapfile bs=1M count=2048 status=progress
        chmod 600 /mnt/ext_storage/swapfile
        mkswap /mnt/ext_storage/swapfile
    fi
    swapon /mnt/ext_storage/swapfile || true
    free -h
"

# ── 3. Upload BRAM / Low-Footprint Control Files ──────────────────────────────
echo "[3/6] Deploying BRAM / OCM modules to /opt/c2s/ ..."
scp ${BASE_DIR}/bram/*.py ${ZED_USER}@${ZED_IP}:/opt/c2s/
scp ${BASE_DIR}/bram/requirements.txt ${ZED_USER}@${ZED_IP}:/opt/c2s/

# ── 4. Upload External Memory Files (Large Modules & Web) ────────────────────
echo "[4/6] Deploying application modules to 8 GB External Storage..."
scp ${BASE_DIR}/external/*.py ${ZED_USER}@${ZED_IP}:/mnt/ext_storage/
scp -r ${BASE_DIR}/external/web/* ${ZED_USER}@${ZED_IP}:/mnt/ext_storage/web/

if [ -d "${BASE_DIR}/external/pico_sensors" ]; then
    scp -r ${BASE_DIR}/external/pico_sensors ${ZED_USER}@${ZED_IP}:/mnt/ext_storage/
fi

# ── 5. Upload AI Model Weights to External Memory ────────────────────────────
echo "[5/6] Uploading YOLO models to 8 GB External Storage (/mnt/ext_storage/models)..."
if [ -f "${BASE_DIR}/external/models/yolov8n.pt" ]; then
    scp ${BASE_DIR}/external/models/yolov8n.pt ${ZED_USER}@${ZED_IP}:/mnt/ext_storage/models/
fi
if [ -f "${BASE_DIR}/external/models/best.pt" ]; then
    scp ${BASE_DIR}/external/models/best.pt ${ZED_USER}@${ZED_IP}:/mnt/ext_storage/models/
fi

# ── 6. Setup Systemd Service ──────────────────────────────────────────────────
echo "[6/6] Installing auto-start systemd service..."
scp ${SCRIPT_DIR}/c2s-waste-zedboard.service ${ZED_USER}@${ZED_IP}:/etc/systemd/system/c2s-waste.service
ssh ${ZED_USER}@${ZED_IP} "
    systemctl daemon-reload
    systemctl enable c2s-waste.service || true
"

echo "=================================================================="
echo "  DEPLOYMENT TO ZEDBOARD COMPLETE!"
echo "=================================================================="
echo "To start the system interactively:"
echo "  ssh root@${ZED_IP}"
echo "  cd /opt/c2s"
echo "  python3 main_zynq.py"
echo ""
echo "To start the Web Dashboard:"
echo "  cd /mnt/ext_storage && python3 -m http.server 5000"
echo "  or via Flask web_app.py"
echo "=================================================================="

