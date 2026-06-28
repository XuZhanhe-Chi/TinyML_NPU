# 已知限制

> **English summary:** v0.1.0 is a deliberately narrow research release: one Zybo board, one fixed KWS artifact, INT8 execution, shared BRAM, and a bare-metal PS application. Unsupported behavior should be treated as unverified, not implied support.

## v0.1.0 边界

- 只验证原版 Digilent Zybo XC7Z010，不支持 Zybo Z7 或其他开发板。
- PL 固定使用 50 MHz 单时钟域，没有异步接口配置。
- 演示只读取编译进 ELF 的 KWS bundle 和 testvector，没有实时麦克风或音频前端。
- 仓库不提供 KWS ONNX 模型、训练代码或数据集，因而不能从零重训并重生成同一个 demo bundle。
- 编译器只支持受约束的 ONNX QDQ 子集，不是通用 ONNX Runtime。
- 当前量化执行只承诺 signed INT8；INT4/INT2 编码是保留值。
- 板级 PASS 要求 120 个样本全部完成、reference top1 全部一致、label accuracy 不低于 117/120，并使用最大 INT8 误差容差；逐 byte mismatch 只作为诊断信息。
- 共享 BRAM 只有 128 KiB，模型必须同时容纳 uOP、参数、activation arena 和 result block。
- PS 驱动采用 polling 完成路径；IRQ 已连接，但 demo 不实现完整中断服务程序。
- AXI-Lite/APB bridge 只支持 32-bit full-word register write 契约。
- 发布二进制由 Vivado/Vitis 2021.1 生成，不能由纯开源本地检查重建。

## 非目标

本版本不尝试提供生产级错误恢复、安全启动、多租户隔离、动态模型调度、操作系统驱动、完整性能分析或独立形式验证。将代码用于产品前，需要自行完成这些工程工作和合规评估。

## 解释测试结果

单元测试通过不等于所有网络都能在硬件运行；硬件算子支持也不等于任意 ONNX 节点组合都能被 frontend 接受。能力矩阵区分 compiler、RTL 和 board-demo 三种验证层级。

## 后续方向

路线图见 [ROADMAP.md](../ROADMAP.md)。新增功能必须先保留 v0.1.0 的 KWS 回归，并提供明确的资源、时序和板级验证记录。
