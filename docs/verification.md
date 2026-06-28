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

GitHub Actions 只执行开源工具链部分。Vivado/Vitis 需要专有工具，实体板还需要本地 JTAG，因此两者由发布维护者手工验收。

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

首次公开提交 `c33553c` 在原版 Digilent Zybo XC7Z010 上得到：

```text
NPU version       0x00050000
uOP count         44
top1              0 (expected 0)
byte mismatches   10 / 12
max abs error     5 (tolerance 5)
STATUS            0x00000020
DEBUG0            0x000897F4 = 563188 cycles
DEBUG1            0x00200000
result            PASS
```

参考 logits 为：

```text
expected: [74, -26, -21, -19, -31, -18, -18, -26, -24, -23, -20, -21]
board:    [69, -22, -20, -20, -31, -18, -16, -22, -22, -20, -19, -20]
```

测试使用 fixed-point tolerance 而不是要求逐 byte 相等，因为 behavioral reference 与硬件截断/舍入边界可能产生小的 INT8 差异。top1 必须严格相同，最大绝对误差不得超过 5。

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

`DEBUG0` 从 NPU busy 拉起开始，在 busy 期间每个 PL clock 加一。当前结果：

```text
563188 cycles / 50,000,000 Hz = 0.01126376 s
```

因此只能报告 **11.26376 ms NPU active-cycle equivalent**。它不包含：

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
