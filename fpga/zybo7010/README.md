# ZYBO7010 Demo

This directory contains the files needed for the first public board demo.

## RTL Files

- `rtl/axi_lite_to_apb3.v`: PS AXI4-Lite control to NPU APB3 registers.
- `rtl/ahb_lite_to_bram_port.v`: NPU AHB-Lite DMA to shared BRAM native port.
- `rtl/venuscore_zybo_wrapper.v`: wrapper around the bridges and generated `VenusCoreTop`.

Generate the NPU RTL before creating the Vivado project:

```bash
bash scripts/gen_rtl.sh --top VenusCoreTop
```

Add these Verilog files plus `build/rtl/VenusCoreTop.v` to Vivado.

For the reproducible batch flow, provide the Digilent `board_files` directory:

```bash
DIGILENT_BOARD_REPO=/path/to/vivado-boards/new/board_files \
  bash scripts/build_zybo7010.sh
```

The script builds `build/vivado_zybo7010/tinyml_npu_zybo7010.xsa` and the matching bitstream.
The PL clock is 50 MHz, and the build fails if post-route setup timing is not met.

## Address Map

- Wrapper AXI4-Lite control: `0x43C0_0000`
- Shared BRAM: `0x4000_0000..0x4001_FFFF`
- IRQ: connect `venus_irq` to PS `IRQ_F2P[0]`

The BRAM bridge uses byte addresses on `bram_addr`. If your BRAM IP expects word addresses, insert a small address-shift wrapper or adjust the port mapping in the block design.

## Vitis App

Use `app/src` as the bare-metal application source directory. The app is testvector-only:

- No microphone input.
- No audio frontend.
- No runtime model loading.

Build the standalone application with Vitis 2021.1:

```bash
bash scripts/build_vitis_zybo7010.sh
```

The resulting ELF is `build/vitis_zybo7010/kws_test.elf`. The script uses Vitis
HSI to generate and compile the standalone BSP, then links the app with the
Vitis ARM cross-compiler; it does not require the Eclipse UI.

With the board connected over JTAG, program and run the demo with:

```bash
bash scripts/run_zybo7010.sh
```

Besides the UART log, the firmware publishes its result at `0x4001_FFC0`, so
the script can report PASS/failure and the NPU debug registers through XSDB.
PASS requires the expected top1 and a maximum int8 logit error of at most 5;
the exact byte mismatch count is still reported for visibility.

Expected UART output:

```text
TinyML_NPU ZYBO7010 KWS testvector demo
NPU version: ...
uOP count  : 44
top1=0 expected=0
TEST PASSED
```
