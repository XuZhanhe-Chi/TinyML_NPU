# Host Check Scripts

- `dump_uop_debug.py`: decode `uops.bin` and print each uOP.
- `check_bundle_relocation.py`: parse `bundle.h`, compare binary blobs, and validate W3/W4/W5 relocation in offset mode.
- `check_plan_sanity.py`: validate plan tensor/step bounds, alignment, and optional address ranges.

Examples:

```bash
python -m scripts.dump_uop_debug --uops-bin out/examples/onnx_kws_qdq/uops.bin
python -m scripts.check_bundle_relocation --out-dir out/examples/onnx_kws_qdq --act-base 0x40000000 --param-base 0x40000000
python -m scripts.check_plan_sanity --out-dir out/examples/onnx_kws_qdq --act-base 0x40000000 --param-base 0x40000000
```
