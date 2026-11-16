################################################################################
# Vivado Block Design Automation TCL Script
# Target: Avnet / Digilent ZedBoard (Zynq-7000 XC7Z020-CLG484-1)
# Project: C2S Smart Waste AI System
#
# Execution:
#   Open Vivado -> Tools -> Run Tcl Script -> Select this file
#   OR from Vivado Tcl Console:
#     cd [get_property DIRECTORY [current_project]]
#     source zedboard_block_design.tcl
################################################################################

set proj_name "c2s_zedboard_hw"
set proj_dir  "./vivado_zedboard_project"
set part_num  "xc7z020clg484-1"

puts "=========================================================="
puts "  CREATING ZEDBOARD HARDWARE DESIGN (XC7Z020)"
puts "=========================================================="

# ── 1. Create Vivado Project ──────────────────────────────────────────────────
create_project $proj_name $proj_dir -part $part_num -force

# Attempt to set ZedBoard board part if installed in Vivado board repository
catch {
    set_property BOARD_PART "em.avnet.com:zed:part0:1.4" [current_project]
    puts "Board part set to: em.avnet.com:zed:part0:1.4"
}

# ── 2. Create Block Design ────────────────────────────────────────────────────
create_bd_design "c2s_zedboard"

# ── 3. Add Zynq Processing System 7 ──────────────────────────────────────────
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]

# Apply board presets if available, otherwise configure MIO directly
catch {
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
        $ps7
}

# Explicit ZedBoard hardware configuration parameters
set_property -dict [list \
    CONFIG.PCW_DDR_RAM_HIGHADDR {0x1FFFFFFF} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.0} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART0_UART0_IO {MIO 14 .. 15} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_USB0_IO {MIO 28 .. 39} \
    CONFIG.PCW_USB0_RESET_ENABLE {1} \
    CONFIG.PCW_USB0_RESET_IO {MIO 7} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_SD0_IO {MIO 40 .. 45} \
    CONFIG.PCW_SD0_GRP_CD_ENABLE {1} \
    CONFIG.PCW_SD0_GRP_CD_IO {MIO 0} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_ENET0_IO {MIO 16 .. 27} \
    CONFIG.PCW_ENET0_GRP_MDIO_ENABLE {1} \
    CONFIG.PCW_ENET0_GRP_MDIO_IO {MIO 52 .. 53} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {1} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_QSPI_QSPI_IO {MIO 1 .. 6} \
] $ps7

# ── 4. Add Reset System ───────────────────────────────────────────────────────
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# ── 5. Add AXI Interconnect ───────────────────────────────────────────────────
set axi_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $axi_ic

# ── 6. Add AXI GPIO for ZedBoard LEDs & Switches ──────────────────────────────
set axi_gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0]
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH {8} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_GPIO2_WIDTH {8} \
    CONFIG.C_ALL_INPUTS_2 {1} \
] $axi_gpio

# ── 7. Add AXI BRAM Controller (256 KB On-Chip Block RAM) ────────────────────
set bram_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.0 axi_bram_ctrl_0]
set_property -dict [list \
    CONFIG.DATA_WIDTH {32} \
    CONFIG.SINGLE_PORT_BRAM {1} \
] $bram_ctrl

# Add Block Memory Generator
set blk_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_0]
set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Write_Width_A {32} \
    CONFIG.Read_Width_A {32} \
    CONFIG.Write_Depth_A {65536} \
    CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
] $blk_mem

# Connect BRAM Controller to BRAM Generator
connect_bd_intf_net [get_bd_intf_pins $bram_ctrl/BRAM_PORTA] [get_bd_intf_pins $blk_mem/BRAM_PORTA]

# ── 8. Connect AXI Buses ──────────────────────────────────────────────────────
connect_bd_intf_net [get_bd_intf_pins $ps7/M_AXI_GP0] [get_bd_intf_pins $axi_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $axi_ic/M00_AXI] [get_bd_intf_pins $axi_gpio/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $axi_ic/M01_AXI] [get_bd_intf_pins $bram_ctrl/S_AXI]

# ── 9. Connect Clocks & Resets ────────────────────────────────────────────────
connect_bd_net [get_bd_pins $ps7/FCLK_CLK0] \
               [get_bd_pins $axi_ic/ACLK] \
               [get_bd_pins $axi_ic/S00_ACLK] \
               [get_bd_pins $axi_ic/M00_ACLK] \
               [get_bd_pins $axi_ic/M01_ACLK] \
               [get_bd_pins $axi_gpio/s_axi_aclk] \
               [get_bd_pins $bram_ctrl/s_axi_aclk] \
               [get_bd_pins $rst/slowest_sync_clk]

connect_bd_net [get_bd_pins $ps7/FCLK_RESET0_N] [get_bd_pins $rst/ext_reset_in]
connect_bd_net [get_bd_pins $rst/interconnect_aresetn] [get_bd_pins $axi_ic/ARESETN]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] \
               [get_bd_pins $axi_ic/S00_ARESETN] \
               [get_bd_pins $axi_ic/M00_ARESETN] \
               [get_bd_pins $axi_ic/M01_ARESETN] \
               [get_bd_pins $axi_gpio/s_axi_aresetn] \
               [get_bd_pins $bram_ctrl/s_axi_aresetn]

# ── 10. Make External Ports ───────────────────────────────────────────────────
# Export LED and Switch pins
set leds_port [create_bd_port -dir O -from 7 -to 0 LEDS_8BITS_tri_o]
connect_bd_net [get_bd_pins $axi_gpio/gpio_io_o] [get_bd_ports $leds_port]

set sw_port [create_bd_port -dir I -from 7 -to 0 SWITCHES_8BITS_tri_i]
connect_bd_net [get_bd_pins $axi_gpio/gpio2_io_i] [get_bd_ports $sw_port]

# Export PS7 DDR and Fixed IO if not already external
catch { make_bd_intf_pins_external [get_bd_intf_pins $ps7/DDR] }
catch { make_bd_intf_pins_external [get_bd_intf_pins $ps7/FIXED_IO] }

# ── 11. Address Assignment ────────────────────────────────────────────────────
assign_bd_address

# Map AXI GPIO to 0x41200000
set_property offset 0x41200000 [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_gpio_0_Reg}]
set_property range  64K        [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_gpio_0_Reg}]

# Map AXI BRAM to 0x40000000 (256 KB)
set_property offset 0x40000000 [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_bram_ctrl_0_Mem0}]
set_property range  256K       [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_bram_ctrl_0_Mem0}]

# ── 12. Validate & Generate Wrapper ───────────────────────────────────────────
validate_bd_design
save_bd_design

make_wrapper -files [get_files c2s_zedboard.bd] -top
set wrapper_file [glob -nocomplain ./vivado_zedboard_project/c2s_zedboard_hw.gen/sources_1/bd/c2s_zedboard/hdl/c2s_zedboard_wrapper.v]
if {[file exists $wrapper_file]} {
    add_files -norecurse $wrapper_file
    set_property top c2s_zedboard_wrapper [current_fileset]
}

# Add ZedBoard Constraints
add_files -fileset constrs_1 -norecurse "../zedboard_constraints.xdc"

puts "=========================================================="
puts "  BLOCK DESIGN CREATED SUCCESSFULLY!"
puts "  Run Synthesis and Implementation to generate Bitstream:"
puts "    launch_runs impl_1 -to_step write_bitstream -jobs 4"
puts "=========================================================="

