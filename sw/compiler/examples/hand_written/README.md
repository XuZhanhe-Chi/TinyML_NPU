# 手写 IR 示例（examples/hand_written）

本目录的脚本通过“手写 IR builder”直接构建小网络，用于在不依赖 ONNX 的情况下验证：

- IR 表达与 lowering（例如 FC 自动 lower 为 1×1 PW）
- tiling 与约束检查
- 后端 `uops.bin / params.bin / bundle.h` 生成

## 常用脚本

- `conv3x3_single_tile.py`：单 tile 的 3×3 卷积
- `pw1x1_single_tile.py`：单 tile 的 1×1 点卷积
- `dw3x3_single_tile.py`：单 tile 的 3×3 深度卷积
- `avgpool_2x2_single_tile.py`：2×2 平均池化
- `fc_single_tile.py`：全连接（Midend 中会 lower 为 1×1 点卷积）
- `conv3x3_multi_tile.py`：多 tile 示例（主要覆盖 H 方向切分）
- `multi_layer_chain.py`：多层串联端到端示例（便于观察中间张量与内存规划）

运行示例：

```bash
python -m examples.hand_written.multi_layer_chain
```
