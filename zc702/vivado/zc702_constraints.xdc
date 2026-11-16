################################################################################
# ZC702 Vivado Constraints File
# AI-Enabled Smart Waste Segregation & Monitoring System
# Target: Xilinx ZC702 Evaluation Board (Zynq-7000 XC7Z020-CLG484-1)
#
# PURPOSE: This XDC file assigns physical pin locations and I/O standards
# for PL-side peripherals. The PS-side pins (MIO: UART, Ethernet, USB)
# are configured automatically by the Zynq PS7 IP block in Vivado and do
# NOT need XDC entries.
#
# HOW TO USE:
#   1. Create a Vivado project targeting xc7z020clg484-1
#   2. Add this file: Constraints → Add Sources → Select this .xdc
#   3. Build your block design (PS7 + AXI-GPIO + AXI-UART16550 if needed)
#   4. Run Synthesis → Implementation → Generate Bitstream
#   5. Export Hardware (.xsa) → Launch PetaLinux or bare-metal SDK
################################################################################

# ── CLOCK ──────────────────────────────────────────────────────────────────────
# ZC702 has a 200 MHz differential oscillator on U43 (bank 13)
set_property PACKAGE_PIN Y9    [get_ports "SYSCLK_P"]
set_property PACKAGE_PIN Y8    [get_ports "SYSCLK_N"]
set_property IOSTANDARD  DIFF_SSTL15 [get_ports "SYSCLK_P"]
set_property IOSTANDARD  DIFF_SSTL15 [get_ports "SYSCLK_N"]
create_clock -period 5.000 -name sys_clk_p [get_ports SYSCLK_P]

# ── USER LEDs (DS15–DS18 on ZC702, Bank 7 / MIO — controlled via PS GPIO) ─────
# PL-side LED DS23 (GPIO_LED_0 on J55 header) ← EMIO routed via AXI-GPIO
set_property PACKAGE_PIN Y21   [get_ports "GPIO_LED_0"]
set_property IOSTANDARD  LVCMOS33 [get_ports "GPIO_LED_0"]

# ── USER DIP SWITCHES (SW10 on Bank 34) ───────────────────────────────────────
# Use to select run-mode: SW10[0]=headless, SW10[1]=strict-sensor
set_property PACKAGE_PIN F22   [get_ports "GPIO_DIP_SW0"]
set_property PACKAGE_PIN G22   [get_ports "GPIO_DIP_SW1"]
set_property IOSTANDARD  LVCMOS25 [get_ports "GPIO_DIP_SW0"]
set_property IOSTANDARD  LVCMOS25 [get_ports "GPIO_DIP_SW1"]

# ── USER PUSH BUTTONS (Bank 34) ───────────────────────────────────────────────
# CPU_RESET button (SW4 / FMCCC) → mapped to system reset
set_property PACKAGE_PIN J15   [get_ports "CPU_RESET"]
set_property IOSTANDARD  LVCMOS25 [get_ports "CPU_RESET"]

# ── PMOD (JA1–JA4 on Bank 13 — optional sensor PMOD if Pico not used) ─────────
# set_property PACKAGE_PIN Y11   [get_ports "PMOD_JA1"]
# set_property IOSTANDARD  LVCMOS33 [get_ports "PMOD_JA1"]

# ── IIC / I2C (Bank 34, used for FMC EEPROM) ──────────────────────────────────
# ZC702 IIC_SCL / IIC_SDA pass through PS MIO — no PL XDC needed

# ── SPI FLASH (PS MIO, controlled by PS7 IP) ──────────────────────────────────
# No XDC required — PS7 handles QSPI MIO pins automatically

# ── USB OTG (PS MIO J7 — USB camera /dev/video0) ──────────────────────────────
# PS7 handles USB ULPI MIO34-53 — no XDC required

# ── ETHERNET (PS MIO, GigE) ───────────────────────────────────────────────────
# Web dashboard served over Ethernet. PS7 handles ETH0 MIO16-27 automatically.

# ── UART1 (Pico W serial, PS MIO48/49) ────────────────────────────────────────
# Configured inside PS7 IP block → no XDC required.
# Physical connection: Pico W TX → ZC702 J58 MIO49, GND → GND

# ── DDR3 Memory (512MB on board, PS-controlled via PS7) ───────────────────────
# No XDC required — DDR3 is directly connected to Zynq PS hard memory controller

# ── TIMING / FALSE PATHS ──────────────────────────────────────────────────────
set_false_path -from [get_ports "GPIO_DIP_SW*"]
set_false_path -from [get_ports "CPU_RESET"]
set_false_path -to   [get_ports "GPIO_LED_0"]

################################################################################
# BRAM placement guidance (reference — Vivado handles placement automatically)
#
# If using AXI BRAM Controller with custom BRAM init for firmware data:
#   - Target RAMB36E1 primitives in SLR0 columns X0Y0–X0Y9
#   - Each RAMB36E1 = 36Kb = 4.5KB
#   - ZC702 (XC7Z020) has 140 RAMB36 = 630KB of BRAM in PL fabric
#   - PS On-Chip RAM (OCM) = 256KB (4x64KB banks, PS side)
#
# Recommended BRAM allocation for this application:
#   Bank 0 (64KB OCM) : Interrupt vectors, OS boot stub
#   Bank 1 (64KB OCM) : Python interpreter startup code
#   Bank 2 (64KB OCM) : config_zc702.py + alert_system.py + database.py
#   Bank 3 (64KB OCM) : main.py + data_fusion.py + webcam_capture.py
#   PL BRAM (630KB)   : AXI-GPIO register map, DMA descriptors, frame buffers
#   DDR3 (1GB)        : Python runtime, YOLO models, LLM, database, web assets
################################################################################

