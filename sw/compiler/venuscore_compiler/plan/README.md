# Plan（执行计划）

Plan 是编译器新增的“运行时调度层”，用于把整个网络切分为若干串行步骤（step），从而支持：

- 仅支持部分算子在 NPU 执行时，网络仍可端到端跑通（NPU step + CPU step 串行）
- `RESHAPE/IDENTITY` 等视图算子以“共享存储 offset”的方式零拷贝处理（**默认不输出 `ALIAS_STEP`**，Plan 允许 `ALIAS_STEP==0`）

## 关键概念

- **TensorDesc**：运行时张量描述（`offset_bytes`、shape、dtype、layout、`quant_index` 等）
- **StepDesc**：步骤描述（类型、输入/输出张量 id 列表）
  - `NPU_STEP`：包含本 step 对应的 `uop_off_words/uop_words`、`param_off_words/param_words`
  - `CPU_STEP`：包含算子类型（如 `ADD`、`CONCAT_C`）与张量引用
  - `ALIAS_STEP`：零拷贝视图变化（可选；默认省略，runtime 侧通常为 NOP）

## 文件说明

- `types.py`
  - Plan 数据结构（张量表、步骤表、枚举等）
- `builder.py`
  - 将 Midend/Backend 的信息汇总为最终 Plan 的构建逻辑
- `binary_format.py`
  - Plan 的二进制表示（可选输出 `plan.bin` 的格式定义与序列化/反序列化）
