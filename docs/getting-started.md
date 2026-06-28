# 入门与工具链

> **English summary:** Open-source checks need Python 3.10+, Java 8, sbt, GCC, and Icarus Verilog. The board flow additionally needs Vivado/Vitis 2021.1 and a connected original Digilent Zybo. Use the root Makefile as the stable entrypoint.

## 支持环境

v0.1.0 的参考环境：

| 工具 | 已验证版本 | 用途 |
|---|---|---|
| Python | 3.11.7 | 编译器与测试 |
| Java | Corretto 8 | sbt / SpinalHDL |
| sbt | launcher 1.10.2，project 1.11.7 | Scala 构建 |
| Scala | 2.13.14 | SpinalHDL source |
| SpinalHDL | 1.12.3 | RTL 生成 |
| Icarus Verilog | 11.0 | glue RTL 仿真 |
| Vivado / Vitis | 2021.1 | ZYBO7010 构建与下载 |

Linux 是当前唯一验证的主机环境。开源检查不需要安装 Xilinx 工具。

## 一次性准备

```bash
git clone https://github.com/XuZhanhe-Chi/TinyML_NPU.git
cd TinyML_NPU
make env
make setup
```

`make env` 只检测环境。`make setup` 创建 `.venv/` 并以 editable 模式安装 `sw/compiler` 的开发依赖，不会写入系统 Python。

如果希望使用已有虚拟环境：

```bash
python3 -m pip install -e './sw/compiler[dev]'
make check PYTHON=python3
```

## 日常检查

```bash
make check
```

该命令执行：

- Python 单元测试和 compiler smoke bundle。
- bundle relocation、目标配置和公开工件完整性检查。
- sbt compile 与两个公开顶层的 RTL 生成。
- glue RTL 行为仿真及全量 Verilog 语法检查。
- 固件 host syntax check。
- 文档链接和公开仓库卫生检查。

检查会写入被忽略的 `build/`、`hw/spinal/target/` 和 `sw/compiler/out/`，但不会修改已跟踪源码。

## Xilinx 工具发现

脚本按以下顺序选择 settings 文件：

1. `XILINX_VIVADO_SETTINGS` 或 `XILINX_VITIS_SETTINGS`。
2. 兼容变量 `XILINX_SETTINGS`。
3. `/home/tools/Xilinx/<tool>/2021.1/settings64.sh`。
4. `/opt/Xilinx/<tool>/2021.1/settings64.sh`。

例如：

```bash
export XILINX_VIVADO_SETTINGS=/tools/Xilinx/Vivado/2021.1/settings64.sh
export XILINX_VITIS_SETTINGS=/tools/Xilinx/Vitis/2021.1/settings64.sh
```

## Digilent board files

`make zybo-bitstream` 默认获取 `Digilent/vivado-boards` 的固定提交
`36f34ab687b7fa9c778b779d027f3bce63b3ace9`，并使用其中的
`new/board_files`。下载内容位于 `build/deps/`，不进入 Git。

离线环境可以显式指定现有目录：

```bash
make zybo-bitstream DIGILENT_BOARD_REPO=/path/to/vivado-boards/new/board_files
```

目标 board part 是 `digilentinc.com:zybo:part0:2.0`，器件是
`xc7z010clg400-1`。不要把该配置用于 Zybo Z7。

## 板级运行

```bash
make zybo-bitstream
make zybo-app
make zybo-run
```

如果只想检查连接：

```bash
make zybo-probe
```

板级脚本需要本机 `hw_server` 能够访问 JTAG。`make zybo-run` 会复位 PS、下载 bitstream、执行 `ps7_init`、下载 ELF，并轮询共享 BRAM 顶部的 result block。

## 常见故障

| 现象 | 检查项 |
|---|---|
| 找不到 board part | 检查固定 board-files 是否完整，路径应指向 `new/board_files` |
| 找不到 `vivado`/`xsct` | 先运行 `make env BOARD=1`，确认 settings 文件 |
| 没有 JTAG target | 检查 USB 权限、线缆、电源和 `hw_server` |
| result timeout | 查看 XSDB 输出的 9 个 result words，并读取 `STATUS/DEBUG0/DEBUG1` |
| top1 正确但字节不同 | 以 max absolute INT8 error 是否超过 5 判定，mismatch 数只用于诊断 |

更具体的 Vivado block design 和固件说明见 [ZYBO7010 README](../fpga/zybo7010/README.md)。
