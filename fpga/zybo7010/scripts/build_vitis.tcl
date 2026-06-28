set repo_root [file normalize [lindex $argv 0]]
set xsa_file [file join $repo_root build vivado_zybo7010 tinyml_npu_zybo7010.xsa]
set out_dir [file join $repo_root build vitis_zybo7010]

if {![file exists $xsa_file]} {
  error "hardware platform is missing: $xsa_file"
}

file delete -force $out_dir
file mkdir $out_dir

hsi::open_hw_design $xsa_file
hsi::create_sw_design tinyml_npu_bsp \
  -proc ps7_cortexa9_0 \
  -os standalone \
  -app empty_application
hsi::generate_bsp -dir [file join $out_dir bsp] -compile
hsi::generate_app -dir [file join $out_dir app_template] -app empty_application

puts "TINYML_NPU_BSP=[file join $out_dir bsp ps7_cortexa9_0]"
puts "TINYML_NPU_HSI_PASS"
exit
