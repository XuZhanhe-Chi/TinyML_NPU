# Examples

Public examples are intentionally small:

- `hand_written/conv3x3_single_tile.py`: no model dependency, useful as a compiler smoke test.
- `onnx/compile_kws_qdq.py`: compiles a local KWS QDQ ONNX model into `uops.bin`, `params.bin`, `metadata.json`, and `bundle.h`.
