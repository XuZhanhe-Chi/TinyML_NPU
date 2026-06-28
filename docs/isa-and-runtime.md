# uOP ISA、寄存器与运行时 ABI

> **English summary:** VenusCore executes fixed 32-byte little-endian uOPs. The PS controls execution through a small register file and publishes board-test results through a versioned 64-byte shared-memory record.

## 32-byte uOP

每条 uOP 由 8 个 32-bit little-endian word 组成：

| Word | 主要字段 |
|---|---|
| W0 | opcode、activation、first/last、stride、padding、tile H/W、sync |
| W1 | C4 input/output、Y index、quant mode |
| W2 | input/output row stride |
| W3 | parameter byte address 或 offset |
| W4 | input feature-map byte address 或 offset |
| W5 | output feature-map byte address 或 offset |
| W6 | original IFM width/height |
| W7 | activation/output DMA line word count |

关键位域：

| 字段 | 位宽 | 约束 |
|---|---:|---|
| opcode | 4 | `0x0..0x6` 已定义 |
| activation | 3 | none、ReLU、ReLU6 |
| stride | 2 | 编码 1 或 2 |
| each padding | 1 | 0 或 1 |
| tile H/W | 8 each | `0..255` |
| C4 input/output | 10 each | 通道数除以 4 后编码 |
| address W3-W5 | 32 each | byte address |

Opcode：`0x0 NOP`、`0x1 Conv2D`、`0x2 Pointwise`、`0x3 Depthwise`、`0x4 AveragePool`、`0x5 MaxPool`、`0x6 MatMul/FC`。

Python 的 `layout_spec.py` 是 bitfield 编码来源；RTL decoder、固件结构和本文档由一致性测试约束。

## 控制寄存器

所有偏移相对 `VENUS_REG_BASE`，ZYBO7010 上为 `0x43C0_0000`。

| Offset | Name | Access | 说明 |
|---|---|---|---|
| `0x00` | UOP_BASE | RW | uOP byte address |
| `0x04` | CTRL | RW/W1P | bit0 start，bit1 abort，bit2 soft reset |
| `0x08` | STATUS | RO | busy、error、opcode、error code |
| `0x0C` | VERSION | RO | 当前板级值 `0x00050000` |
| `0x10` | UOP_COUNT | RW | 有效 uOP 数 |
| `0x20` | INT_ENABLE | RW | done/error enable |
| `0x24` | INT_STATUS | RW1C | done/error status |
| `0x80` | DEBUG0 | RO | busy 期间 NPU cycle count |
| `0x84` | DEBUG1 | RO | error/opcode/scheduler/fetch/flags snapshot |
| `0x88..0xA0` | DEBUG2..8 | RO | cluster、stall、DMA byte 和执行计数 |
| `0xA4` | DEBUG_CTRL | W1P | bit0 清零 debug counters |

寄存器写必须是 32-bit full-word transaction。当前 APB3 interface 没有 `PSTRB`，AXI-Lite bridge 不实现 partial register writes。

## STATUS 与 DEBUG1

`STATUS`：

- bit 0：busy。
- bit 1：error。
- bits 7:4：current opcode。
- bits 15:8：error code。

`DEBUG1`：

- bits 31:24：error code。
- bits 23:20：current opcode。
- bits 19:16：scheduler state。
- bits 15:12：uOP fetch state。
- bits 11:0：内部 ready/busy/IRQ flags。

## Board result ABI v1

固件在共享 BRAM 最后 64 bytes，即 `0x4001_FFC0`，发布 16 个 word。写入所有字段并 flush cache 后，最后写 magic，XSDB 因而不会读取到半完成记录。

| Word | Offset | 字段 | 说明 |
|---:|---:|---|---|
| 0 | `0x00` | magic | `0x544E5055`，记录完成 |
| 1 | `0x04` | code | 0 pass，1 hardware/runtime failure，2 output failure |
| 2 | `0x08` | hw_status | `venus_run_bundle` 返回的 STATUS |
| 3 | `0x0C` | top1 | UINT32_MAX 表示没有有效输出 |
| 4 | `0x10` | mismatches | 与参考 logits 不同的 byte 数 |
| 5 | `0x14` | max_abs_error | signed INT8 最大绝对误差 |
| 6 | `0x18` | status | 最终寄存器 STATUS snapshot |
| 7 | `0x1C` | debug0 | NPU active cycles |
| 8 | `0x20` | debug1 | 状态机 snapshot |
| 9..15 | `0x24..0x3C` | reserved | 必须写 0 |

PASS 条件是 `code == 0`、top1 等于固定参考值并且 `max_abs_error <= 5`。mismatch 可以非零。

## Cache 与重定位契约

- PS 写 uOP、参数或输入后必须 flush 对应 cache range。
- PS 读取 NPU 输出前必须 invalidate output range。
- offset bundle 的 W3 加 parameter staging base，W4/W5 加 activation arena base。
- 重定位后的地址必须位于 `0x4000_0000..0x4001_FFFF`。
- uOP 基址和数据数组按至少 4 bytes 对齐，生成的 C bundle 使用 64-byte alignment。
