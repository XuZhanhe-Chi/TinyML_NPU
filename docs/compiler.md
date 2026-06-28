# 编译器、算子能力与内存布局

> **English summary:** The public compiler lowers a constrained ONNX QDQ subset or hand-written graphs into VenusCore uOPs, parameter blocks, metadata, and a C header. It is a transparent research compiler, not a general ONNX runtime.

## 编译阶段

```text
ONNX QDQ / manual IR
        -> normalize and validate
        -> quant parameter extraction
        -> NCHWc4 layout lowering
        -> hardware-aware tiling
        -> activation memory planning
        -> parameter packing
        -> uOP encoding
        -> bundle + host sanity checks
```

每个阶段保留 Python 数据结构，便于在 `--dump-ir` 或 `--dump-uop` 输出中追踪。编译器不会调用硬件工具，也不会把模型嵌入 RTL。

## 公开输出

| 工件 | 内容 |
|---|---|
| `uops.bin` | 连续的 32-byte little-endian uOP |
| `params.bin` | 按 tile/lane 打包的权重和量化参数 |
| `metadata.json` | 地址模式、张量范围、tile-uOP 映射和 plan 摘要 |
| `bundle.h` | 64-byte aligned C arrays、宏和可选执行 plan |

板级路径使用 `address_mode=offset`。W3/W4/W5 分别保存 parameter、input 和 output offset，固件加载时加上实际 staging/activation 基址。

## 算子能力矩阵

| 算子 | IR / compiler | VenusCore v0.1 RTL | 板级 KWS 覆盖 |
|---|---|---|---|
| Conv 3x3 | 支持 | 支持 | 是 |
| Pointwise Conv 1x1 | 支持 | 支持 | 是 |
| Depthwise Conv 3x3, multiplier=1 | 支持 | 支持 | 是 |
| AveragePool 2x2/3x3 | 支持 | 支持 | 是 |
| MaxPool | 编码与约束支持 | 支持 | 否 |
| Fully connected / MatMul | 可规范化为 1x1 路径 | ISA 保留 opcode | 否 |
| Add / Concat | plan 可表示 CPU fallback | NPU 不执行 | 否 |

“支持”只代表当前约束内能够生成并执行，不代表覆盖任意 ONNX 形状、广播或量化配置。KWS 之外的能力主要由单元测试和 behavioral simulator 验证，尚未全部在 ZYBO7010 上形成网络级回归。

## 量化和布局约束

- 输入、权重和输出为 signed INT8，内部累加为 INT32。
- 当前硬件只启用 `QMode.INT8`。
- 激活布局为 NCHWc4，通道补齐到 4。
- Conv stride 支持 1 或 2，二维 stride 必须相等。
- padding 位域每边只能表示 0 或 1。
- tile 高和宽位域为 8 bit；channel group 和 `y_index` 为 10 bit。
- 单个 IBUF 行由 compiler 限制为 3,840 bytes。
- 单个 tile 的 parameter payload 必须适配 8 KiB WBUF 总容量。

违反约束时 compiler 应抛出带 layer/tile 上下文的 `ValueError`，而不是静默回退。

## 内存规划

线性网络默认使用 ping-pong activation placement；多输入、fan-out 或 CPU plan step 使用 arena allocation。`activation_peak_bytes` 是整个 arena 的峰值，不含 uOP 和 parameter staging。

板级 KWS 当前布局：

| 字段 | 大小/偏移 |
|---|---:|
| uOP stream | 1,408 bytes / 44 uOPs |
| parameter block | 58,112 bytes |
| activation peak | 64,000 bytes |
| input | offset 0，8,000 bytes |
| output | offset 32,000，12 bytes |

## 使用方式

推荐用专用 example 入口，而不是通用 console wrapper：

```bash
python -m examples.hand_written.conv3x3_single_tile --output-dir out/smoke
python -m examples.onnx.compile_kws_qdq \
  --model /path/to/model.onnx \
  --output-dir out/kws \
  --address-mode offset
python -m scripts.check_bundle_relocation \
  --out-dir out/kws \
  --act-base 0x40000000 \
  --param-base 0x40000000
```

通用 `tinyml-npu-compiler` 只接受存在的 `.onnx` 输入或显式的 `--manual-smoke`，不会再把拼写错误的文件名静默解释为手写网络。
