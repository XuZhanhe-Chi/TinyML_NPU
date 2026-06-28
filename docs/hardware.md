# 硬件微架构与配置

> **English summary:** VenusCore is a single-cluster, four-lane, SIMD4 INT8 accelerator. It uses a three-line IBUF, per-lane WBUF banks, an output buffer, APB3 control, and an AHB-Lite DMA engine. The public preset is intentionally fixed for the XC7Z010 demo.

## 默认配置

| 参数 | v0.1.0 |
|---|---:|
| Cluster | 1 |
| Lanes per cluster | 4 |
| SIMD per lane | 4 |
| Activation / weight | signed INT8 |
| Accumulator | INT32 |
| AHB / APB data width | 32 bit |
| WBUF | 2,048 B/lane × 4 = 8 KiB |
| IBUF | 12 KiB，3 lines |
| IBUF physical line | 4,096 B |
| Compiler line limit | 3,840 B |
| OBUF | 256 × 32-bit words |
| Shared BRAM | 128 KiB |

编译器采用 3,840-byte 行限制，给边界和后续参数调整保留余量；RTL 的物理容量仍是 4,096 bytes/line。两者不是配置错误，自动测试会固定这一保守关系。

## 数据通路

1. `CtrlUopFetch` 通过 DMA 读取连续 32-byte uOP。
2. `CtrlScheduler` 将解码后的 tile 命令发送给 cluster。
3. activation DMA 把 NCHWc4 输入行加载到三行 IBUF。
4. weight DMA 把权重和每输出通道量化参数放入四个 WBUF bank。
5. 四个 PE lane 并行执行 SIMD4 点积，结果进入 INT32 累加路径。
6. SFU 加 bias、缩放、移位、激活并饱和到 INT8。
7. OBUF 暂存结果，由 DMA 按行写回共享 BRAM。

## 片上存储

### IBUF

IBUF 按 NCHWc4 的 32-bit spatial word 组织。三行 rolling buffer 支持 3x3 卷积和 depthwise 的窗口复用；pointwise 和 pooling 复用同一接口。

### WBUF

每个 lane 有独立 2 KiB bank。参数块按 lane 交织，使同一个输出通道组的权重可以并行供给四个 lane。bias/scale/shift 与权重一起由 weight DMA 装载。

### OBUF

OBUF 是计算和输出 DMA 之间的弹性缓冲。当前深度为 256 words，并保留 almost-full margin，避免 DMA 背压覆盖正在完成的计算结果。

## 控制与 DMA

控制面在 `0x43C0_0000` 暴露 4 KiB AXI window，其中核心实际寄存器位于低 256 bytes。APB 写没有 byte strobe；板级 bridge 接收 AXI `WSTRB`，但固件契约要求所有寄存器都使用 32-bit full-word 写入。

AHB-Lite DMA bridge 支持 byte、halfword 和 word strobe，地址是 byte address。访问共享 BRAM 范围外的地址返回错误响应。BRAM 是同步读，因此 bridge 插入可配置等待周期。

## 时钟与复位

ZYBO7010 实现的 PS、AXI interconnect、BRAM、glue 和 VenusCore 都使用 50 MHz `FCLK_CLK0`。`proc_sys_reset` 生成同步 peripheral reset。v0.1.0 不支持异步时钟域配置。

## 配置来源

- RTL preset：`VenusCoreConfig.zybo7010`。
- compiler preset：`default_hw_config("zybo7010")`。
- Vivado 地址与时钟：ZYBO7010 `create_project.tcl`。

修改 lane、buffer 或共享内存容量会同时影响 RTL、compiler tiler、固件布局和板级资源，不属于 v0.1.0 稳定接口。任何修改都必须重新运行 `make check` 和实体板验证。
