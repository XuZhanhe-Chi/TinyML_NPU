set repo_root [file normalize [lindex $argv 0]]
set bit_file [file join $repo_root build vivado_zybo7010 tinyml_npu_zybo7010.runs impl_1 system_wrapper.bit]
set elf_file [file join $repo_root build vitis_zybo7010 kws_test.elf]
set ps7_init_file [file join $repo_root build vivado_zybo7010 ps7_init.tcl]

foreach required [list $bit_file $elf_file $ps7_init_file] {
  if {![file exists $required]} {
    error "required board artifact is missing: $required"
  }
}

set result_addr 0x4001FFC0
set result_magic 0x544E5055
set result_words 9
set expected_version 0x00050000
set expected_top1 0
set max_allowed_error 5

connect -url tcp:127.0.0.1:3121
puts "TINYML_NPU_TARGETS:\n[targets]"

targets -set -nocase -filter {name =~ "APU"} -timeout 10
rst -system
after 1000

fpga -file $bit_file
puts "TINYML_NPU_FPGA_PROGRAMMED"

targets -set -nocase -filter {name =~ "*Cortex-A9*#0"} -timeout 10
rst -processor
source $ps7_init_file
ps7_init
ps7_post_config
configparams force-mem-access 1

set version [mrd -value 0x43C0000C]
puts [format "TINYML_NPU_VERSION=0x%08X" $version]
if {$version != $expected_version} {
  disconnect
  error [format "unexpected NPU version: got 0x%08X expected 0x%08X" \
    $version $expected_version]
}

mwr $result_addr 0
dow $elf_file
con

set complete 0
for {set attempt 0} {$attempt < 600} {incr attempt} {
  after 100
  set magic [mrd -value $result_addr]
  if {$magic == $result_magic} {
    set complete 1
    break
  }
}

if {!$complete} {
  targets -set -nocase -filter {name =~ "*Cortex-A9*#0"} -timeout 10
  stop
  set words [mrd -value $result_addr $result_words]
  disconnect
  error "board result timeout: $words"
}

set words [mrd -value $result_addr $result_words]
set code [lindex $words 1]
set hw_status [lindex $words 2]
set top1 [lindex $words 3]
set mismatches [lindex $words 4]
set max_abs_error [lindex $words 5]
set status [lindex $words 6]
set debug0 [lindex $words 7]
set debug1 [lindex $words 8]

puts [format "TINYML_NPU_RESULT code=%u hw_status=0x%08X top1=%u mismatches=%u max_abs_error=%u" \
  $code $hw_status $top1 $mismatches $max_abs_error]
puts [format "TINYML_NPU_DEBUG status=0x%08X debug0=0x%08X debug1=0x%08X" \
  $status $debug0 $debug1]

disconnect
if {$code != 0} {
  error "KWS board test failed with result code $code"
}
if {$top1 != $expected_top1 || $max_abs_error > $max_allowed_error} {
  error "KWS board result violates the v0.1 acceptance contract"
}

puts "TINYML_NPU_BOARD_PASS"
exit
