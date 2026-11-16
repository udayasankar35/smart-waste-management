################################################################################
# ZC702 Vivado Block Design TCL Script
# AI-Enabled Smart Waste Segregation & Monitoring System
# Target: Xilinx ZC702 / Zynq-7000 XC7Z020-CLG484-1
#
# Run this script inside Vivado TCL console:
#   source zc702_block_design.tcl
#
# What this creates:
#   - Zynq PS7 (ARM Cortex-A9 dual-core, DDR3 controller, UART, USB, ETH)
#   - AXI GPIO  (LED + DIP switch control from PS Linux)
#   - AXI BRAM Controller (256KB PL-side BRAM for firmware / data staging)
#   - AXI Interconnect
#   - Processor System Reset
################################################################################

# ── Create Project ─────────────────────────────────────────────────────────────
create_project c2s_zc702 ./vivado_project -part xc7z020clg484-1 -force
set_property BOARD_PART xilinx.com:zc702:part0:1.4 [current_project]

# ── Create Block Design ────────────────────────────────────────────────────────
create_bd_design "c2s_system"

# ── Add Zynq PS7 ───────────────────────────────────────────────────────────────
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]

# Apply ZC702 preset (loads MIO config, DDR3, Ethernet, USB, UART)
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    $ps7

# Enable additional PS interfaces
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
    CONFIG.PCW_DDR_RAM_HIGHADDR {0x3FFFFFFF} \
] $ps7

# ── Add Processor System Reset ─────────────────────────────────────────────────
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# ── Add AXI Interconnect ───────────────────────────────────────────────────────
set axi_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $axi_ic

# ── Add AXI GPIO (LED + DIP Switch) ───────────────────────────────────────────
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0]
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH {4} \
    CONFIG.C_GPIO2_WIDTH {4} \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_ALL_INPUTS_2 {1} \
] $gpio

# ── Add AXI BRAM Controller (256KB BRAM for lightweight code staging) ──────────
set bram_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.0 axi_bram_ctrl_0]
set_property -dict [list \
    CONFIG.DATA_WIDTH {32} \
    CONFIG.SINGLE_PORT_BRAM {1} \
] $bram_ctrl

# ── Add Block Memory Generator ─────────────────────────────────────────────────
set bram_gen [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 blk_mem_gen_0]
set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Write_Width_A {32} \
    CONFIG.Read_Width_A {32} \
    CONFIG.Write_Depth_A {65536} \
    CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
] $bram_gen

# ── Connect BRAM Controller to Memory Generator ────────────────────────────────
connect_bd_intf_net [get_bd_intf_pins $bram_ctrl/BRAM_PORTA] \
                    [get_bd_intf_pins $bram_gen/BRAM_PORTA]

# ── Connect PS to AXI Interconnect ────────────────────────────────────────────
connect_bd_intf_net [get_bd_intf_pins $ps7/M_AXI_GP0] \
                    [get_bd_intf_pins $axi_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $axi_ic/M00_AXI] \
                    [get_bd_intf_pins $gpio/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $axi_ic/M01_AXI] \
                    [get_bd_intf_pins $bram_ctrl/S_AXI]

# ── Clocks & Resets ────────────────────────────────────────────────────────────
connect_bd_net [get_bd_pins $ps7/FCLK_CLK0] \
               [get_bd_pins $axi_ic/ACLK] \
               [get_bd_pins $gpio/s_axi_aclk] \
               [get_bd_pins $bram_ctrl/s_axi_aclk] \
               [get_bd_pins $rst/slowest_sync_clk]

connect_bd_net [get_bd_pins $ps7/FCLK_RESET0_N] \
               [get_bd_pins $rst/ext_reset_in]

connect_bd_net [get_bd_pins $rst/interconnect_aresetn] \
               [get_bd_pins $axi_ic/ARESETN]

connect_bd_net [get_bd_pins $rst/peripheral_aresetn] \
               [get_bd_pins $gpio/s_axi_aresetn] \
               [get_bd_pins $bram_ctrl/s_axi_aresetn]

# ── Make GPIO External ─────────────────────────────────────────────────────────
make_bd_intf_pins_external [get_bd_intf_pins $gpio/GPIO]
make_bd_intf_pins_external [get_bd_intf_pins $gpio/GPIO2]

# ── Assign Addresses ───────────────────────────────────────────────────────────
assign_bd_address

# Configure AXI GPIO address
set_property offset 0x41200000 [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_gpio_0_Reg}]
set_property range 64K         [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_gpio_0_Reg}]

# Configure AXI BRAM address (256KB at 0x40000000)
set_property offset 0x40000000 [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_bram_ctrl_0_Mem0}]
set_property range 256K        [get_bd_addr_segs {processing_system7_0/Data/SEG_axi_bram_ctrl_0_Mem0}]

# ── Validate & Save ────────────────────────────────────────────────────────────
validate_bd_design
save_bd_design

# ── Generate HDL Wrapper ───────────────────────────────────────────────────────
make_wrapper -files [get_files c2s_system.bd] -top
add_files -norecurse ./vivado_project/c2s_zc702.gen/sources_1/bd/c2s_system/hdl/c2s_system_wrapper.v
set_property top c2s_system_wrapper [current_fileset]

# ── Add Constraints ────────────────────────────────────────────────────────────
add_files -fileset constrs_1 ../zc702_constraints.xdc

# ── Run Implementation ─────────────────────────────────────────────────────────
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# ── Export Hardware (for PetaLinux / SDK) ──────────────────────────────────────
write_hw_platform -fixed -include_bit -force -file ./c2s_zc702.xsa
puts "SUCCESS: Bitstream generated and hardware platform exported to c2s_zc702.xsa"
puts "Next: Launch PetaLinux with this XSA and deploy the application."

