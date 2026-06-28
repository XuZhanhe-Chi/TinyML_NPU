# VenusCore Compiler（Python 包结构说明）

`venuscore_compiler/` 是编译器主体实现，按典型三层编译器分为 Frontend / Midend / Backend，并在其上新增 Plan（执行计划）与 CPU fallback，用于支持“部分算子上 NPU、其余算子在 CPU 串行执行”的端到端路径。

## 总体流程（从模型到工件）

1. **Frontend**：解析输入（手写 builder / ONNX），构建逻辑 IR：`VcProgram`
2. **Midend**：对 IR 做规范化、量化信息整理、tiling、约束校验，并将图切分为 `NPU_STEP / CPU_STEP / ALIAS_STEP`
3. **Backend**：做物理内存规划（Activation arena）、生成 `params.bin`（Param Block）、生成 `uops.bin`（32B uOP 流），并固化 `Plan` 到 `bundle.h`
4. **Runtime 导出**：输出面向固件/仿真的封装（`bundle.h`、可选二进制 plan 等）

## 子目录说明

- `frontend/`：输入解析与 IR 构建
- `ir/`：逻辑 IR 数据结构（张量、算子、程序）
- `midend/`：IR pass（lowering/tiler/partition/约束等）
- `backend/`：内存规划、Param Block 打包、uOP 生成与编码
- `plan/`：Plan 数据结构与二进制格式（张量表/步骤表）
- `runtime/`：导出器（`bundle.h`、测试平台导出）
- `isa/`：uOP 编解码与字段定义（需与仓库 `docs/isa-and-runtime.md` 对齐）
- `config/`：默认配置与硬件约束参数
- `common/`、`utils/`：通用工具
