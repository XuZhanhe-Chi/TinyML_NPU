# 仿真（sim）

`sim/` 提供两条验证路径：

- uOP 级行为仿真：直接解释执行 `uops.bin/params.bin`，用于对齐单层/线性链路的数值
- Plan 级端到端仿真：按 `bundle.h` 的 Plan 逐步执行，支持 CPU fallback（用于残差/分支网络）

## 目录说明

- `behavioral/`：uOP 级功能模型（非 cycle-accurate）
- `scripts/`：仿真脚本入口（行为仿真、Plan 仿真）
- `testvectors/`：测试向量（输入与参考输出）
- `network/`：测试网络的训练/导出脚本（用于复现）

