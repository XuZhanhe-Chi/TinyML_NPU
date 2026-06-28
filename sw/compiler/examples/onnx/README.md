# ONNX Example

`compile_kws_qdq.py` expects a local quantized KWS ONNX model:

```bash
python -m examples.onnx.compile_kws_qdq \
  --model /path/to/kws_qdq_int8.onnx \
  --output-dir out/examples/onnx_kws_qdq \
  --address-mode offset
```

The model file is not included in this repository.
