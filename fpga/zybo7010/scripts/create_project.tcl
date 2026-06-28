set repo_root [file normalize [lindex $argv 0]]
set board_repo [file normalize [lindex $argv 1]]
set out_dir [file join $repo_root build vivado_zybo7010]

set_param board.repoPaths [list $board_repo]

create_project tinyml_npu_zybo7010 $out_dir -part xc7z010clg400-1 -force
set_property board_part digilentinc.com:zybo:part0:2.0 [current_project]
cd $out_dir

add_files [list \
  [file join $repo_root build rtl VenusCoreTop.v] \
  [file join $repo_root fpga zybo7010 rtl axi_lite_to_apb3.v] \
  [file join $repo_root fpga zybo7010 rtl ahb_lite_to_bram_port.v] \
  [file join $repo_root fpga zybo7010 rtl venuscore_zybo_wrapper.v]]
update_compile_order -fileset sources_1

create_bd_design system

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {apply_board_preset "1" make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} \
  [get_bd_cells ps7]
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_EN_CLK0_PORT {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50.000000} \
  CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
  CONFIG.PCW_IRQ_F2P_INTR {1}] [get_bd_cells ps7]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells axi_ic]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 bram_ctrl
set_property -dict [list CONFIG.DATA_WIDTH {32} CONFIG.SINGLE_PORT_BRAM {1}] [get_bd_cells bram_ctrl]

create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 shared_bram
set_property -dict [list \
  CONFIG.Memory_Type {True_Dual_Port_RAM} \
  CONFIG.Use_Byte_Write_Enable {true} \
  CONFIG.Byte_Size {8} \
  CONFIG.Write_Width_A {32} \
  CONFIG.Read_Width_A {32} \
  CONFIG.Write_Depth_A {32768} \
  CONFIG.Write_Width_B {32} \
  CONFIG.Read_Width_B {32} \
  CONFIG.Enable_A {Use_ENA_Pin} \
  CONFIG.Enable_B {Use_ENB_Pin} \
  CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
  CONFIG.Register_PortB_Output_of_Memory_Primitives {false} \
  CONFIG.Use_RSTA_Pin {true} \
  CONFIG.Use_RSTB_Pin {true}] [get_bd_cells shared_bram]

create_bd_cell -type module -reference venuscore_zybo_wrapper npu

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ic/M00_AXI] [get_bd_intf_pins npu/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ic/M01_AXI] [get_bd_intf_pins bram_ctrl/S_AXI]
connect_bd_intf_net [get_bd_intf_pins bram_ctrl/BRAM_PORTA] [get_bd_intf_pins shared_bram/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins npu/BRAM_PORT] [get_bd_intf_pins shared_bram/BRAM_PORTB]

connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
  [get_bd_pins ps7/M_AXI_GP0_ACLK] \
  [get_bd_pins axi_ic/ACLK] \
  [get_bd_pins axi_ic/S00_ACLK] \
  [get_bd_pins axi_ic/M00_ACLK] \
  [get_bd_pins axi_ic/M01_ACLK] \
  [get_bd_pins bram_ctrl/s_axi_aclk] \
  [get_bd_pins npu/s_axi_aclk] \
  [get_bd_pins rst_ps7/slowest_sync_clk]

connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst_ps7/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7/interconnect_aresetn] [get_bd_pins axi_ic/ARESETN]
connect_bd_net [get_bd_pins rst_ps7/peripheral_aresetn] \
  [get_bd_pins axi_ic/S00_ARESETN] \
  [get_bd_pins axi_ic/M00_ARESETN] \
  [get_bd_pins axi_ic/M01_ARESETN] \
  [get_bd_pins bram_ctrl/s_axi_aresetn] \
  [get_bd_pins npu/s_axi_aresetn]

connect_bd_net [get_bd_pins npu/venus_irq] [get_bd_pins ps7/IRQ_F2P]

assign_bd_address -offset 0x43C00000 -range 0x00001000 \
  -target_address_space [get_bd_addr_spaces ps7/Data] \
  [get_bd_addr_segs npu/S_AXI/reg0]
assign_bd_address -offset 0x40000000 -range 0x00020000 \
  -target_address_space [get_bd_addr_spaces ps7/Data] \
  [get_bd_addr_segs bram_ctrl/S_AXI/Mem0]

validate_bd_design
save_bd_design

make_wrapper -files [get_files system.bd] -top
add_files -norecurse [file join $out_dir tinyml_npu_zybo7010.gen sources_1 bd system hdl system_wrapper.v]
set_property top system_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1

generate_target all [get_files system.bd]
launch_runs synth_1 -jobs 8
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
if {![string match "*Complete*" $synth_status]} {
  error "synth_1 failed: $synth_status"
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
if {![string match "*Complete*" $impl_status]} {
  error "impl_1 failed: $impl_status"
}

open_run impl_1
report_utilization -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing_summary.rpt]
set check_timing_file [file join $out_dir check_timing.rpt]
check_timing -verbose -file $check_timing_file

set timing_handle [open $check_timing_file r]
set timing_checks [read $timing_handle]
close $timing_handle
foreach required_check {no_clock unconstrained_internal_endpoints} {
  if {![regexp "checking ${required_check} \\(0\\)" $timing_checks]} {
    error "timing check failed: $required_check is non-zero"
  }
}

set unrouted_count [llength [get_nets -quiet -hierarchical -filter {ROUTE_STATUS == UNROUTED}]]
set partial_count [llength [get_nets -quiet -hierarchical -filter {ROUTE_STATUS == PARTIALLY_ROUTED}]]
puts "TINYML_NPU_UNROUTED_NETS=$unrouted_count"
puts "TINYML_NPU_PARTIAL_NETS=$partial_count"
if {$unrouted_count != 0 || $partial_count != 0} {
  error "routing is incomplete: unrouted=$unrouted_count partial=$partial_count"
}

set worst_path [get_timing_paths -setup -max_paths 1]
set wns [get_property SLACK $worst_path]
puts "TINYML_NPU_WNS=$wns"
if {$wns < 0.0} {
  error "timing constraints are not met: WNS=$wns ns"
}
write_hw_platform -fixed -include_bit -force [file join $out_dir tinyml_npu_zybo7010.xsa]

set bit_file [file join $out_dir tinyml_npu_zybo7010.runs impl_1 system_wrapper.bit]
if {![file exists $bit_file]} {
  error "bitstream missing: $bit_file"
}
set release_bit [file join $out_dir tinyml_npu_zybo7010.bit]
file copy -force $bit_file $release_bit
puts "TINYML_NPU_BIT=$release_bit"
puts "TINYML_NPU_XSA=[file join $out_dir tinyml_npu_zybo7010.xsa]"
puts "TINYML_NPU_VIVADO_PASS"
exit
