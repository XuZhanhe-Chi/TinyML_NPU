# 验证结果与复现边界

> **English summary:** v0.1.0 is accepted through open-source unit/RTL checks, a clean Vivado/Vitis 2021.1 build, and a physical Zybo run. Reported NPU latency is derived from an internal active-cycle counter and excludes host overhead.

## 验证层次

| 层次 | 检查内容 | 自动化入口 |
|---|---|---|
| Compiler | ISA、layout、tiler、backend、behavioral sim、relocation | `make test-python` |
| SpinalHDL | Scala compile、公开顶层生成 | `make rtl` |
| Glue RTL | AXI/APB、AHB/BRAM 行为与语法 | `make test-rtl` |
| Firmware | host C syntax、ABI constants | `make test-firmware` |
| Repository | 大文件、范围、链接、生成物 | `make hygiene` |
| FPGA | synthesis、implementation、timing、bit/XSA | `make zybo-bitstream` |
| Board | program、ELF download、result ABI | `make zybo-run` |

当前公开仓库不启用 GitHub-hosted Actions CI。开源工具链检查通过本地 `make check` 复现；Vivado/Vitis 需要专有工具，实体板还需要本地 JTAG，因此两者由发布维护者手工验收并记录到 release manifest。

## 参考工具环境

- RHEL 8.10。
- Python 3.11.7。
- Java 8 Corretto。
- sbt launcher 1.10.2 / project 1.11.7。
- Scala 2.13.14 / SpinalHDL 1.12.3。
- Icarus Verilog 11.0。
- Vivado/Vitis 2021.1。
- Digilent board-files commit `36f34ab687b7fa9c778b779d027f3bce63b3ace9`。

## 板级基线

`v0.1.0` 更新验收在原版 Digilent Zybo XC7Z010 上得到：

```text
NPU version       0x00050000
uOP count         44
samples           120, balanced 10 per class
label accuracy    117 / 120 = 97.50%
reference top1    120 / 120 = 100.00%
byte mismatches   0
max abs error     0 (tolerance 5)
STATUS            0x00000020
DEBUG0            0x000897F4 = 563188 cycles per sample
DEBUG1            0x00200000
total cycles      67,582,560
result            PASS
```

样本集来自预生成 `testvectors_kws_mixed`，只公开量化后的 feature/testvector header，不公开训练数据、ONNX 模型或训练脚本。类别为：

```text
yes, no, up, down, left, right, on, off, stop, go, noise, silence
```

每类 10 个样本，共 120 个。参考来自 VenusCore behavioral backend 的 INT8 logits，因此当前板测逐 byte 一致；验收仍保留 `max_abs_error <= 5` 的 fixed-point tolerance，用于容纳后续 RTL/工具小幅舍入差异。`ref_top1_match` 必须全对，`label_accuracy` 记录模型对真实 label 的准确度。

## 实现结果

Vivado 2021.1 post-route，50 MHz：

| 资源/时序 | 使用量 | XC7Z010 占比 |
|---|---:|---:|
| LUT | 7,328 | 41.64% |
| Registers | 6,738 | 19.14% |
| BRAM tiles | 42 | 70.00% |
| DSP | 7 | 8.75% |
| WNS | +3.025 ns | timing met |

验收还要求 timing summary 中不存在未约束内部 endpoint，implementation 不存在 unrouted net。

## 延迟口径

`DEBUG0` 从 NPU busy 拉起开始，在 busy 期间每个 PL clock 加一。当前每个样本的 NPU active cycles 均为：

```text
563188 cycles / 50,000,000 Hz = 0.01126376 s
```

120 个样本的 active-cycle 求和为：

```text
67582560 cycles / 50,000,000 Hz = 1.3516512 s
```

因此只能报告 **11.26376 ms/sample** 和 **1351.6512 ms/120 samples** 的 NPU active-cycle equivalent。它不包含：

- PS 把 bundle 和 input 复制到共享 BRAM。
- cache flush/invalidate。
- bare-metal 程序启动和校验输出。
- UART 打印、JTAG 下载和 XSDB 轮询。

没有额外 PS timer 或 logic analyzer 测量时，不报告端到端 latency、实时系数或每秒推理数。

## 发布验收

发布 `v0.1.0` 前必须从干净工作树执行：

```bash
make check
make zybo-bitstream
make zybo-app
make zybo-run
make release-package VERSION=v0.1.0
```

发布 manifest 记录提交、工具版本、SHA-256、timing 和板级 PASS 摘要。二进制资产不进入 Git 历史。
