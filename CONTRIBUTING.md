# Contributing to TinyML_NPU

感谢你改进 TinyML_NPU。这个项目优先保证可理解、可复现和 ZYBO7010 基线不退化，而不是快速扩大功能范围。

## 开始之前

- 阅读 [项目范围](README.md#范围与许可) 和 [已知限制](docs/limitations.md)。
- 对新增板卡、模型、算子或硬件配置，先创建 issue 说明目标、验证方法和维护成本。
- 不要提交模型、数据集或第三方源码，除非权利来源和许可证已经明确记录。
- 不要提交 `build/`、生成 Verilog、Vivado/Vitis 工程、waveform 或 Python/Scala cache。

## 开发环境

```bash
make env
make setup
make check
```

`make check` 是合入前的最低要求。修改 RTL bridge 时应同时更新对应 Icarus testbench；修改 SpinalHDL 时必须重新运行 `make rtl`。修改 uOP、寄存器、bundle 或 result ABI 时，必须同步文档、固件和一致性测试。

涉及板级行为的更改还需要：

```bash
make zybo-bitstream
make zybo-app
make zybo-run
```

请在提交或 PR 中记录 Vivado WNS、资源变化和结构化 board result。

## 提交原则

- 一个提交只处理一个可解释的主题。
- 使用简短的 imperative commit subject，例如 `Add bridge protocol regressions`。
- 保持公开 API 的兼容性；需要破坏接口时先在 issue 中说明迁移方式。
- Python 公共 API 使用类型标注，错误应抛出有上下文的 `ValueError`。
- Scala/SpinalHDL 遵循仓库已有命名和中文模块说明。
- 新代码和测试使用 Apache-2.0 SPDX header。

## Pull Request 检查项

- 说明改变了什么、为什么改变以及用户影响。
- 列出实际运行的测试，不要只写“应该通过”。
- 区分 compiler simulation、RTL simulation 和 physical-board evidence。
- 不把 active-cycle 估算描述为端到端性能。
- 确认 `git status` 不包含工具生成文件。

## English Summary

Keep changes focused, preserve the validated ZYBO7010 KWS baseline, run `make check`, and attach physical-board evidence for changes that affect board behavior. Do not commit generated tool output, external models, datasets, or third-party code without documented redistribution rights.
