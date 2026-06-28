# Backend（内存规划、Param Block 与 uOP 生成）

Backend 的职责是把 Midend 的输出（tile 计划、量化系数、权重数据、步骤切分等）转换为硬件/运行时可消费的工件：

- `params.bin`：按 tile 拼接的 Param Block blob
- `uops.bin`：32B 定长 uOP 指令流（8×u32，小端）
- `bundle.h`：包含 `uops_words[] / params_words[]` 以及 Plan（`VC_PLAN`）的 C 头文件

## 文件说明

- `memory_planner.py`
  - Activation arena 内存规划：为运行时张量分配 `offset_bytes`（含 16B 对齐）
  - 对 alias 张量（reshape/identity 输出）复用输入张量的 offset，不分配新空间
- `layout_ifm_ofm.py`
  - IFM/OFM 的物理布局与寻址辅助（NCHWc4/int8）
- `layout_param_block.py` / `param_block.py`
  - Param Block 布局与打包（需与 `sw/compiler/doc/VenusCore_MemortMap.md` 对齐）
- `codegen_uop.py`
  - uOP 生成：按 tile 输出 uOP，并在 Plan 模式下记录每个 NPU step 对应的 uOP/Param 段范围（word 为单位）
- `uop_builder.py`
  - uOP 字段级构建（根据 tile/量化/地址规划填充 W0~W5）
- `uop_encoder.py`
  - uOP 编码/序列化（写入 `uops.bin` 或 `bundle.h` 的 `uops_words[]`）
- `types.py`
  - 后端内部类型定义（内存计划、参数块描述等）
