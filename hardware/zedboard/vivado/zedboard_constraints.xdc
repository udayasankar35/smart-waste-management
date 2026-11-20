################################################################################
# ZedBoard Master Constraints File (.xdc)
# Target Board: Avnet / Digilent ZedBoard (Zynq-7000 XC7Z020-CLG484-1)
# Project: C2S Smart Waste AI System
################################################################################

# ── 100 MHz System Clock (Oscillator U23 on ZedBoard) ─────────────────────────
set_property PACKAGE_PIN Y9 [get_ports "GCLK"]
set_property IOSTANDARD LVCMOS33 [get_ports "GCLK"]
create_clock -period 10.000 -name sys_clk_pin -waveform {0 5} [get_ports GCLK]

# ── 8 User LEDs (LD0 .. LD7) ──────────────────────────────────────────────────
# Mapped to AXI GPIO Channel 1 (8-bit output)
set_property PACKAGE_PIN T22 [get_ports "LEDS_8BITS_tri_o[0]"] ;# LD0 (Heartbeat)
set_property PACKAGE_PIN T21 [get_ports "LEDS_8BITS_tri_o[1]"] ;# LD1 (Camera Active)
set_property PACKAGE_PIN U22 [get_ports "LEDS_8BITS_tri_o[2]"] ;# LD2 (AI Detections)
set_property PACKAGE_PIN U21 [get_ports "LEDS_8BITS_tri_o[3]"] ;# LD3 (Pico Serial Rx)
set_property PACKAGE_PIN V22 [get_ports "LEDS_8BITS_tri_o[4]"] ;# LD4 (Wet Waste Flag)
set_property PACKAGE_PIN W22 [get_ports "LEDS_8BITS_tri_o[5]"] ;# LD5 (Gas Warning)
set_property PACKAGE_PIN U19 [get_ports "LEDS_8BITS_tri_o[6]"] ;# LD6 (Bin Full Alert)
set_property PACKAGE_PIN U14 [get_ports "LEDS_8BITS_tri_o[7]"] ;# LD7 (System Error)

set_property IOSTANDARD LVCMOS33 [get_ports -filter {NAME =~ LEDS_8BITS_tri_o*}]

# ── 8 User Slide Switches (SW0 .. SW7) ────────────────────────────────────────
# Mapped to AXI GPIO Channel 2 (8-bit input)
set_property PACKAGE_PIN F22 [get_ports "SWITCHES_8BITS_tri_i[0]"] ;# SW0 (Mode: Live/Sim)
set_property PACKAGE_PIN G22 [get_ports "SWITCHES_8BITS_tri_i[1]"] ;# SW1 (Strict Sensors)
set_property PACKAGE_PIN H22 [get_ports "SWITCHES_8BITS_tri_i[2]"] ;# SW2 (Logging Verbose)
set_property PACKAGE_PIN F21 [get_ports "SWITCHES_8BITS_tri_i[3]"] ;# SW3 (Enable LLM)
set_property PACKAGE_PIN H19 [get_ports "SWITCHES_8BITS_tri_i[4]"] ;# SW4 (Reserved)
set_property PACKAGE_PIN H18 [get_ports "SWITCHES_8BITS_tri_i[5]"] ;# SW5 (Reserved)
set_property PACKAGE_PIN H17 [get_ports "SWITCHES_8BITS_tri_i[6]"] ;# SW6 (Reserved)
set_property PACKAGE_PIN M15 [get_ports "SWITCHES_8BITS_tri_i[7]"] ;# SW7 (Factory Reset)

set_property IOSTANDARD LVCMOS25 [get_ports -filter {NAME =~ SWITCHES_8BITS_tri_i*}]

# ── 5 User Push Buttons (Center, Down, Left, Right, Up) ───────────────────────
set_property PACKAGE_PIN P16 [get_ports "BTNC"] ;# Center Button
set_property PACKAGE_PIN R16 [get_ports "BTND"] ;# Down Button
set_property PACKAGE_PIN N15 [get_ports "BTNL"] ;# Left Button
set_property PACKAGE_PIN R18 [get_ports "BTNR"] ;# Right Button
set_property PACKAGE_PIN T18 [get_ports "BTNU"] ;# Up Button

set_property IOSTANDARD LVCMOS25 [get_ports -filter {NAME =~ BTN*}]

# ── Pmod JA (Optional Direct Sensor Header) ───────────────────────────────────
# JA1..JA4 (Top Row), JA7..JA10 (Bottom Row)
set_property PACKAGE_PIN Y11  [get_ports "PMOD_JA[0]"] ;# JA1
set_property PACKAGE_PIN AA11 [get_ports "PMOD_JA[1]"] ;# JA2
set_property PACKAGE_PIN Y10  [get_ports "PMOD_JA[2]"] ;# JA3
set_property PACKAGE_PIN AA9  [get_ports "PMOD_JA[3]"] ;# JA4
set_property PACKAGE_PIN AB11 [get_ports "PMOD_JA[4]"] ;# JA7
set_property PACKAGE_PIN AB10 [get_ports "PMOD_JA[5]"] ;# JA8
set_property PACKAGE_PIN AB9  [get_ports "PMOD_JA[6]"] ;# JA9
set_property PACKAGE_PIN AA8  [get_ports "PMOD_JA[7]"] ;# JA10

set_property IOSTANDARD LVCMOS33 [get_ports -filter {NAME =~ PMOD_JA*}]

# ── Timing Constraints & False Paths ──────────────────────────────────────────
set_false_path -from [get_ports -filter {NAME =~ SWITCHES_8BITS_tri_i*}]
set_false_path -from [get_ports -filter {NAME =~ BTN*}]
set_false_path -to   [get_ports -filter {NAME =~ LEDS_8BITS_tri_o*}]

################################################################################
# BRAM & Memory Allocation Summary for ZedBoard (Zynq-7000 XC7Z020)
#
# Total PL Block RAMs:  140 RAMB36 primitives (630 KB total)
# Total PS OCM:         256 KB fast SRAM (Bank 0-3)
# External Physical RAM: 512 MB DDR3 (Micron MT41J128M16)
# Attached Storage:     8 GB External Memory (Ext4 rootfs, models, swap)
#
# Hardware AXI BRAM Controller Address: 0x40000000 - 0x4003FFFF (256 KB)
# Used as high-speed hardware telemetry buffer between PL sensors and PS Cortex-A9.
################################################################################

