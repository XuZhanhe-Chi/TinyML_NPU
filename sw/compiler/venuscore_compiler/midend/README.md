# Midend（下沉、切分与约束）

Midend 的职责是把逻辑 IR 下沉为“面向硬件的 tile 级执行视图”，并在不破坏后端既有 uOP/Param Block 生成逻辑的前提下，新增 Plan 与 CPU fallback 所需的信息（步骤切分、张量表、alias 语义等）。

## 文件说明

- `normalize.py`
  - 图规范化与算子形态整理（把前端导入的多样形式归一到编译器内部口径）
- `layout_lowering.py`
  - 布局下沉与 layout 约束整理（当前运行时张量物理布局默认固定为 `NCHWc4 + int8`）
- `quantize.py`
  - 量化摘要与系数表整理（对称 INT8 口径）
- `npu_constraints.py`
  - NPU 约束检查（tile 尺寸、对齐、tiling 方向限制等）
- `tiler.py`
  - tiling：将整层算子拆分为若干 tile（当前只做 H 方向与 Cout 方向切分；不支持 W/Cin 方向切分）
- `partition_fallback.py`
  - **Partition/Fallback 切分**：按拓扑顺序生成执行步骤：
    - `NPU_STEP`：连续可下沉的子图合并成一步
    - `CPU_STEP`：不支持的算子单独形成一步（当前最小集合包含 `ADD`、`CONCAT_C`）
    - `ALIAS_STEP`：`RESHAPE/IDENTITY` 等零拷贝视图变化
- `param_data.py`
  - Param Block 所需的量化系数、权重数据整理与打包前准备
- `debug_export.py`
  - 导出调试 JSON（IR 与 uOP 相关），用于定位编译链路问题
- `types.py`
  - Midend 内部类型定义（tile 计划、张量描述等）

