# uOP 行为仿真（sim/behavioral）

本目录包含 uOP 级功能模型，用于解释执行编译器生成的 32B uOP 指令流并在内存模型上读写 IFM/OFM/Param Block。

## 文件说明

- `venuscore_sim.py`
  - uOP 解释器与内存模型（非 cycle-accurate）
  - 支持主要 opcode（Conv/PW/DW/AvgPool 等）与 SFU/量化系数应用
  - 当前 AvgPool 为 2×2、stride 固定为 2（与硬件实现口径一致）

> 说明：Plan 级端到端执行（含 CPU fallback）请使用 `sim/scripts/run_plan_sim.py`。
