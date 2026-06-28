# Frontend

The public release keeps:

- `manual_builder.py`: hand-written IR builder for small smoke examples.
- `onnx_loader.py`: ONNX QDQ INT8 import path used by the KWS compiler example.
- `graph_utils.py`: graph traversal and topology helpers.

Unsupported model formats should be converted to ONNX QDQ before using this compiler subset.
