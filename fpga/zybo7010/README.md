# ZYBO7010 KWS Testvector Demo

> **English summary:** This directory contains the original Digilent Zybo XC7Z010 integration, bare-metal testvector firmware, and non-interactive Vivado/Vitis/XSDB scripts. It does not target Zybo Z7.

## 闭环

```text
bundle.h + kws_testvector_fpga.h
        -> PS copies and relocates data in shared BRAM
        -> VenusCore executes 44 uOPs for each KWS sample
        -> PS checks logits/top1 and label accuracy
        -> UART TEST PASSED + JTAG result block
```

演示没有麦克风、音频前端、文件系统或动态模型加载。固定 bundle 和 120 个平衡 KWS testvector 让 FPGA bring-up 不依赖训练环境。

## 文件

- `rtl/axi_lite_to_apb3.v`：PS AXI4-Lite control 到 NPU APB3。
- `rtl/ahb_lite_to_bram_port.v`：NPU AHB-Lite DMA 到 shared BRAM port B。
- `rtl/venuscore_zybo_wrapper.v`：bridge 和生成的 `VenusCoreTop` wrapper。
- `app/src/`：bare-metal driver、固定 bundle 和 KWS testvector。
- `scripts/create_project.tcl`：完整 block design、implementation 和 XSA export。
- `scripts/build_vitis.tcl`：standalone BSP 和 app linker template。
- `scripts/run_board.tcl`：program、download 和 result ABI 读取。

## 地址与时钟

| 接口 | 配置 |
|---|---|
| Device | `xc7z010clg400-1` |
| Board part | `digilentinc.com:zybo:part0:2.0` |
| PL clock | 50 MHz |
| NPU control | `0x43C0_0000..0x43C0_0FFF` |
| Shared BRAM | `0x4000_0000..0x4001_FFFF` |
| Result block | `0x4001_FFC0..0x4001_FFFF` |
| IRQ | `IRQ_F2P[0]` |

BRAM bridge 的 `bram_addr` 是 byte address，与 Vivado BRAM interface metadata 匹配。固件只在共享窗口内布置 uOP、参数、activation 和 result。

## 构建

从仓库根目录执行：

```bash
make env BOARD=1
make zybo-bitstream
make zybo-app
```

输出：

```text
build/vivado_zybo7010/tinyml_npu_zybo7010.bit
build/vivado_zybo7010/tinyml_npu_zybo7010.xsa
build/vitis_zybo7010/kws_test.elf
```

默认脚本会取得固定版本的 Digilent board files。离线构建可以设置：

```bash
export DIGILENT_BOARD_REPO=/path/to/vivado-boards/new/board_files
export XILINX_VIVADO_SETTINGS=/path/to/Vivado/2021.1/settings64.sh
export XILINX_VITIS_SETTINGS=/path/to/Vitis/2021.1/settings64.sh
```

Vivado flow 在 post-route WNS 为负、存在 unrouted net 或关键 timing endpoint 未约束时失败。

## 连接与运行

板卡上电并连接 JTAG 后：

```bash
make zybo-probe
make zybo-run
```

`run_board.tcl` 依次执行 system reset、FPGA program、PS7 init、ELF download，并最多等待 60 秒 result magic。成功时至少包含：

```text
TINYML_NPU_VERSION=0x00050000
TINYML_NPU_RESULT code=0 hw_status=0x00000020 samples=120 label_correct=117 ref_top1_match=120 max_abs_error=0 total_mismatches=0 total_cycles=67582560
TINYML_NPU_FIRST_FAILURE sample=4294967295 top1=4294967295 expected_top1=4294967295 label=4294967295
TINYML_NPU_DEBUG status=0x00000020 debug0=0x000897F4 debug1=0x00200000
TINYML_NPU_BOARD_PASS
```

UART 同时打印：

```text
TinyML_NPU ZYBO7010 KWS multivector demo
NPU version: 0x00050000
uOP count  : 44
label_accuracy=117/120 (97.50%)
ref_top1_match=120/120 (100.00%)
output max_abs_error=0 tolerance=5 total_mismatches=0
TEST PASSED
```

板级 PASS 要求样本数为 120、参考 top1 全部一致、label accuracy 不低于 117/120，且 signed INT8 max absolute error 不超过 5。逐 byte mismatch 数仍会输出，但不是单独的失败条件。

## 故障排查

1. `make zybo-probe` 无设备：检查板卡电源、JTAG 模式、USB 权限和 `hw_server`。
2. VERSION 不正确：确认下载的是本次 XSA 对应 bitstream，并检查 control address assignment。
3. result timeout：停止 Cortex-A9 后读取 `0x4001_FFC0` 的 16 个 word；检查固件是否启动。
4. runtime failure：查看 `hw_status`、`STATUS`、`DEBUG0/1` 和固件打印的首条 uOP W0-W7。
5. output failure：先确认 bundle/testvector 没有混用，再比较 `ref_top1_match`、`label_correct`、max error 和 logits。

result 字段与寄存器解释见 [ISA 与运行时文档](../../docs/isa-and-runtime.md)，测量口径见 [验证文档](../../docs/verification.md)。
