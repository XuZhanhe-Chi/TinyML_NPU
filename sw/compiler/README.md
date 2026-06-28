# TinyML_NPU Compiler

This is the public compiler subset used by the ZYBO7010 KWS demo.

Retained paths:

- Hand-written operator smoke examples.
- ONNX QDQ KWS frontend and lowering path.
- uOP encoder, layout, tiler, parameter block generation, runtime bundle parser, and host sanity checks.
- Behavioral simulator used by unit tests.

Install:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

Generate a hand-written smoke bundle:

```bash
python -m examples.hand_written.conv3x3_single_tile --output-dir out/examples/conv3x3_single_tile
```

Compile a local KWS QDQ ONNX model:

```bash
python -m examples.onnx.compile_kws_qdq \
  --model /path/to/kws_qdq_int8.onnx \
  --output-dir out/examples/onnx_kws_qdq \
  --address-mode offset
```

Check relocation against the ZYBO7010 shared BRAM base:

```bash
python -m scripts.check_bundle_relocation \
  --out-dir out/examples/onnx_kws_qdq \
  --act-base 0x40000000 \
  --param-base 0x40000000
```
