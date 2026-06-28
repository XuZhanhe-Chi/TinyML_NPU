# Roadmap

TinyML_NPU 的路线图以可验证性为门槛，不以功能数量为目标。以下项目不构成发布时间承诺。

## v0.1.x: Reproducibility

- 保持 ZYBO7010 KWS testvector 的 compiler、RTL 和 board 回归。
- 改善错误诊断、文档和跨 Linux 发行版的工具发现。
- 补充更多公开算子的 board-level micro tests，但不改变稳定 ABI。

## v0.2: Live Input Exploration

- 评估实时音频采集和特征前端。
- 用 PS timer 明确定义并测量端到端 latency。
- 在不破坏 testvector 模式的前提下增加 interrupt-driven runtime。

## Later Research

- 评估更大的共享内存或 DDR 数据路径。
- 增加更多量化模型和板卡需要各自独立的许可证、资源、时序和实体板证据。
- 探索性能计数器、编译器 cost model 和更系统的 differential testing。

## 不进入路线图的隐含承诺

路线图不承诺生产级驱动、完整 ONNX 覆盖、商业 IP 支持或固定发布时间。任何新方向都必须先说明维护者、测试资产和硬件可访问性。

## English Summary

Near-term work prioritizes reproducibility and diagnostics. Live audio, broader hardware targets, and performance research are possible later, but only after the current board baseline remains automated and independently reviewable.
