# TinyML_NPU Compiler

> **English summary:** This package is the transparent compiler subset used by the ZYBO7010 demo. It lowers constrained ONNX QDQ or hand-written graphs into VenusCore uOPs, parameter blocks, metadata, and a C bundle.

## 范围

公开编译器保留：

- hand-written operator smoke examples。
- ONNX QDQ KWS frontend 和受约束 lowering。
- normalization、NCHWc4 layout、tiling、memory planning 和 parameter packing。
- 32-byte uOP encoder/decoder。
- bundle parser、relocation/sanity checks 和 behavioral simulator。

它不是通用 ONNX Runtime，不包含训练、数据集、模型下载或 CPU inference framework。完整能力矩阵见 [compiler 文档](../../docs/compiler.md)。

## 安装和测试

推荐从仓库根目录执行：

```bash
make setup
make test-python
```

手动安装：

```bash
python3 -m venv ../../.venv
../../.venv/bin/python -m pip install -e '.[dev]'
../../.venv/bin/python -m pytest -q
```

## 生成 smoke bundle

```bash
../../.venv/bin/python -m examples.hand_written.conv3x3_single_tile \
  --output-dir out/examples/conv3x3_single_tile

../../.venv/bin/python -m scripts.check_bundle_relocation \
  --out-dir out/examples/conv3x3_single_tile \
  --act-base 0x40000000 \
  --param-base 0x40000000
```

## 编译本地 KWS QDQ 模型

```bash
../../.venv/bin/python -m examples.onnx.compile_kws_qdq \
  --model /path/to/kws_qdq_int8.onnx \
  --output-dir out/examples/onnx_kws_qdq \
  --address-mode offset \
  --post-check-act-base 0x40000000 \
  --post-check-param-base 0x40000000
```

输出目录包含 `uops.bin`、`params.bin`、`metadata.json` 和 `bundle.h`。模型文件不会被复制到输出目录。

## 通用 CLI

```bash
tinyml-npu-compiler --input model.onnx --output-dir out/model
tinyml-npu-compiler --manual-smoke --output-dir out/manual
```

输入文件必须存在且扩展名为 `.onnx`；手写 smoke 必须显式选择，避免错误文件名被静默接受。

## 包结构

```text
frontend/       ONNX loader and manual IR builders
midend/         normalization, quantization, layout and tiling
backend/        memory/parameter layout and uOP generation
isa/            enum and 32-byte bitfield source of truth
runtime/        artifacts, bundle parser and host post-checks
plan/           NPU/CPU/alias execution-plan representation
sim/            behavioral reference and runners
tests/          focused compiler regressions
```
