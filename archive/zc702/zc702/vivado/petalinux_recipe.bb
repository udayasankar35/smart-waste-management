################################################################################
# ZC702 PetaLinux Configuration Recipe
# AI-Enabled Smart Waste Segregation & Monitoring System
#
# This file is the PetaLinux user layer recipe to build a rootfs image
# containing Python 3, OpenCV, and auto-start for the C2S application.
#
# LOCATION (in PetaLinux project):
#   project-spec/meta-user/recipes-apps/c2s-waste-system/c2s-waste-system.bb
#
# HOW TO USE:
#   1. Create PetaLinux project: petalinux-create -t project --template zynq -n c2s_zc702
#   2. Configure with XSA:       petalinux-config --get-hw-description=path/to/c2s_zc702.xsa
#   3. Add this recipe to meta-user layer
#   4. Enable in rootfs:         petalinux-config -c rootfs → user packages → c2s-waste-system
#   5. Build:                    petalinux-build
#   6. Package:                  petalinux-package --boot --fsbl --fpga --u-boot --force
#   7. Copy to SD card:          BOOT.BIN, boot.scr, image.ub → FAT32 partition
#                                rootfs.tar.gz extracted → ext4 partition
################################################################################

SUMMARY = "C2S Smart Waste AI System for ZC702"
LICENSE  = "MIT"

SRC_URI = "file://c2s_app.tar.gz"
S = "${WORKDIR}/c2s_app"

RDEPENDS_${PN} += "python3 python3-pip python3-opencv"

do_install() {
    install -d ${D}/opt/c2s
    install -m 0755 ${S}/*.py ${D}/opt/c2s/

    # Systemd service
    install -d ${D}${systemd_unitdir}/system
    install -m 0644 ${S}/c2s-waste.service ${D}${systemd_unitdir}/system/
}

SYSTEMD_SERVICE_${PN} = "c2s-waste.service"

