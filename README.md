# TinyML_NPU

TinyML_NPU is a small open-source NPU project for FPGA-based TinyML experiments. The first public release targets a closed-loop keyword spotting demo on Digilent ZYBO7010:

`bundle.h/testvector -> Zynq PS writes shared BRAM -> VenusCore PL NPU runs -> PS checks output/top1 -> UART prints TEST PASSED`

TinyML_NPU 是一个面向 FPGA TinyML 实验的小型 NPU 开源工程。首版目标是在 Digilent ZYBO7010 上跑通关键词识别测试向量闭环：

`预置 bundle.h/testvector -> Zynq PS 写共享 BRAM -> VenusCore PL NPU 执行 -> PS 比对输出/top1 -> 串口打印 TEST PASSED`

## Scope

Included:

- SpinalHDL source for the VenusCore NPU APB3 control plane and AHB-Lite DMA data plane.
- ZYBO7010 glue RTL: AXI4-Lite-to-APB3, AHB-Lite-to-BRAM, and a wrapper.
- A compact Python compiler subset for hand-written smoke tests and ONNX QDQ KWS bundle generation.
- A testvector-only Vitis bare-metal app with `bundle.h` and `kws_testvector_fpga.h`.

Not included:

- Model authoring assets, large data files, generated RTL, simulation caches, board-specific experiments outside ZYBO7010, or physical-implementation backend materials.

## Repository Layout

```text
hw/spinal/                  SpinalHDL NPU source
fpga/zybo7010/rtl/          Board glue RTL
fpga/zybo7010/app/src/      Testvector-only PS application
sw/compiler/                Python compiler subset
scripts/gen_rtl.sh          SpinalHDL -> Verilog generation
docs/architecture.md        Interface and dataflow notes
```

## Quick Start

Generate RTL:

```bash
cd TinyML_NPU
bash scripts/gen_rtl.sh --top VenusCoreTop
```

Run compiler smoke tests:

```bash
cd sw/compiler
python -m pip install -e .[dev]
pytest -q
python -m examples.hand_written.conv3x3_single_tile --output-dir out/examples/conv3x3_single_tile
```

Run the full local verification bundle:

```bash
cd TinyML_NPU
bash scripts/check_local.sh
```

Probe the Xilinx JTAG connection:

```bash
bash scripts/probe_xilinx_hw.sh /home/tools/Xilinx/Vivado/2021.1/settings64.sh
```

Build the ZYBO7010 bitstream and XSA with Digilent board files:

```bash
DIGILENT_BOARD_REPO=/path/to/vivado-boards/new/board_files \
  bash scripts/build_zybo7010.sh
bash scripts/build_vitis_zybo7010.sh
bash scripts/run_zybo7010.sh
```

Compile a KWS ONNX QDQ model if you have one locally:

```bash
python -m examples.onnx.compile_kws_qdq \
  --model /path/to/kws_qdq_int8.onnx \
  --output-dir out/examples/onnx_kws_qdq \
  --address-mode offset \
  --post-check-act-base 0x40000000 \
  --post-check-param-base 0x40000000
```

## ZYBO7010 Demo

Address map:

- NPU control AXI4-Lite window: `0x43C0_0000`
- Shared BRAM window: `0x4000_0000..0x4001_FFFF`
- Interrupt: `venus_irq` to `IRQ_F2P[0]`

Vivado block design:

1. Add Zynq-7000 PS.
2. Add `venuscore_zybo_wrapper.v`, `axi_lite_to_apb3.v`, `ahb_lite_to_bram_port.v`, and generated `build/rtl/VenusCoreTop.v`.
3. Connect PS `M_AXI_GP0` to wrapper AXI4-Lite control and assign `0x43C0_0000`.
4. Connect wrapper BRAM native port to a 128KB shared BRAM block at `0x4000_0000`.
5. Connect `venus_irq` to `IRQ_F2P[0]`.

The Vitis app in `fpga/zybo7010/app/src` uses the prebuilt KWS bundle and testvector. Expected UART output includes NPU version, uOP count, top1, and `TEST PASSED`.

## License

Apache-2.0. See [LICENSE](LICENSE).
