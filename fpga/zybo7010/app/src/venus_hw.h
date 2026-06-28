/*
 * venus_hw.h
 * VenusCore Hardware Abstraction Layer
 * Definitions of Registers, Memory Map, and ISA structures.
 */

#ifndef VENUS_HW_H
#define VENUS_HW_H

#include <stdint.h>
#include "xparameters.h"
// ==========================================================
// 1. SoC 地址映射配置 (根据实际 SoC 总线矩阵修改)
// ==========================================================

// NPU 寄存器访问基地址 (APB Slave)
#if defined(XPAR_VENUSCORETOP_0_BASEADDR)
#define VENUS_REG_BASE XPAR_VENUSCORETOP_0_BASEADDR
#elif defined(XPAR_NPU_0_BASEADDR)
#define VENUS_REG_BASE XPAR_NPU_0_BASEADDR
#elif defined(XPAR_NPU_BASEADDR)
#define VENUS_REG_BASE XPAR_NPU_BASEADDR
#else
#error "NPU base address is missing from xparameters.h"
#endif

// NPU DMA 可访问的共享内存范围 (SRAM/DDR)
// 用于驱动层的安全检查（本驱动不强依赖该范围；若你需要可自行加检查）
#define SHARED_MEM_BASE 0x40000000
// 128 KByte: [0x4000_0000, 0x4001_FFFF]
#define SHARED_MEM_HIGH 0x4001FFFF

// ==========================================================
// 2. 寄存器偏移定义 (Register Map)
// ==========================================================

#define VENUS_REG_UOP_BASE 0x000   // RW: uOP 链表物理基地址
#define VENUS_REG_CTRL 0x004       // RW: 控制寄存器
#define VENUS_REG_STATUS 0x008     // RO: 状态寄存器
#define VENUS_REG_VERSION 0x00C    // RO: 版本号
#define VENUS_REG_UOP_COUNT 0x010  // RW: uOP 数量
#define VENUS_REG_INT_EN 0x020     // RW: 中断使能
#define VENUS_REG_INT_STATUS 0x024 // RW1C: 中断状态 (写1清除)
// 调试寄存器（若硬件实现了，详见 sw/compiler/doc/VenusCore_RegFile.md）
#define VENUS_REG_DEBUG0 0x080 // RO: 调试计数器/状态
#define VENUS_REG_DEBUG1 0x084 // RO: 调试状态快照（死锁定位）

// ==========================================================
// 3. 寄存器位域定义 (Bit Fields)
//    注：与 VenusCore_RegFile.md 对齐。
// ==========================================================

// CTRL Register
#define CTRL_START_MASK (1 << 0)
#define CTRL_ABORT_MASK (1 << 1)
#define CTRL_RESET_MASK (1 << 2)

// STATUS Register
// Bit0 : BUSY
// Bit1 : ERROR
// Bit7:4  : CURR_OPCODE
// Bit15:8 : ERROR_CODE
#define STATUS_BUSY_MASK (1u << 0)
#define STATUS_ERROR_MASK (1u << 1)

#define STATUS_CURR_OPCODE_SHIFT 4u
#define STATUS_CURR_OPCODE_MASK (0xFu << STATUS_CURR_OPCODE_SHIFT)

#define STATUS_ERROR_CODE_SHIFT 8u
#define STATUS_ERROR_CODE_MASK (0xFFu << STATUS_ERROR_CODE_SHIFT)

// 便捷解析宏
#define VENUS_STATUS_GET_CURR_OPCODE(status)                                   \
  (((status) & STATUS_CURR_OPCODE_MASK) >> STATUS_CURR_OPCODE_SHIFT)
#define VENUS_STATUS_GET_ERROR_CODE(status)                                    \
  (((status) & STATUS_ERROR_CODE_MASK) >> STATUS_ERROR_CODE_SHIFT)

// INT Register
#define INT_DONE_MASK (1 << 0)
#define INT_ERROR_MASK (1 << 1)

// ==========================================================
// 3.1 Debug1 bitfields (CtrlTop 打包格式)
//   [31:24] error_code（来自 NPU_STATUS）
//   [23:20] curr_opcode[3:0]（ClusterUopOpcode）
//   [19:16] scheduler_state（CtrlScheduler.SchedState）
//   [15:12] uop_fetch_state（CtrlUopFetch.State）
//   [11:0]  flags
// ==========================================================

#define DEBUG1_ERROR_CODE_SHIFT 24u
#define DEBUG1_ERROR_CODE_MASK (0xFFu << DEBUG1_ERROR_CODE_SHIFT)
#define DEBUG1_CURR_OPCODE_SHIFT 20u
#define DEBUG1_CURR_OPCODE_MASK (0xFu << DEBUG1_CURR_OPCODE_SHIFT)
#define DEBUG1_SCHED_STATE_SHIFT 16u
#define DEBUG1_SCHED_STATE_MASK (0xFu << DEBUG1_SCHED_STATE_SHIFT)
#define DEBUG1_FETCH_STATE_SHIFT 12u
#define DEBUG1_FETCH_STATE_MASK (0xFu << DEBUG1_FETCH_STATE_SHIFT)
#define DEBUG1_FLAGS_SHIFT 0u
#define DEBUG1_FLAGS_MASK (0xFFFu << DEBUG1_FLAGS_SHIFT)

#define VENUS_DEBUG1_GET_ERROR_CODE(v)                                         \
  (((v) & DEBUG1_ERROR_CODE_MASK) >> DEBUG1_ERROR_CODE_SHIFT)
#define VENUS_DEBUG1_GET_CURR_OPCODE(v)                                       \
  (((v) & DEBUG1_CURR_OPCODE_MASK) >> DEBUG1_CURR_OPCODE_SHIFT)
#define VENUS_DEBUG1_GET_SCHED_STATE(v)                                       \
  (((v) & DEBUG1_SCHED_STATE_MASK) >> DEBUG1_SCHED_STATE_SHIFT)
#define VENUS_DEBUG1_GET_FETCH_STATE(v)                                       \
  (((v) & DEBUG1_FETCH_STATE_MASK) >> DEBUG1_FETCH_STATE_SHIFT)
#define VENUS_DEBUG1_GET_FLAGS(v)                                             \
  (((v) & DEBUG1_FLAGS_MASK) >> DEBUG1_FLAGS_SHIFT)

// DEBUG1 flags[11:0] (CtrlTop.scala)
#define DEBUG1_FLAG_SCHED_BUSY (1u << 0)
#define DEBUG1_FLAG_SCHED_ERROR (1u << 1)
#define DEBUG1_FLAG_SYNC_HOLD (1u << 2)
#define DEBUG1_FLAG_FETCH_BUSY (1u << 3)
#define DEBUG1_FLAG_FETCH_UOP_VALID (1u << 4)
#define DEBUG1_FLAG_FETCH_UOP_READY (1u << 5)
#define DEBUG1_FLAG_ANY_CLUSTER_BUSY (1u << 6)
#define DEBUG1_FLAG_ANY_CLUSTER_DONE (1u << 7)
#define DEBUG1_FLAG_ANY_CLUSTER_ERROR (1u << 8)
#define DEBUG1_FLAG_IRQ_OUT (1u << 9)
#define DEBUG1_FLAG_INT_STATUS_DONE (1u << 10)
#define DEBUG1_FLAG_INT_STATUS_ERROR (1u << 11)

// ==========================================================
// 4. ISA 常量定义
// ==========================================================

// 算子 Opcode
typedef enum {
  OP_NOP = 0x0,
  OP_CONV2D = 0x1,
  OP_PWCONV = 0x2,
  OP_DWCONV = 0x3,
  OP_AVGPOOL = 0x4,
  OP_MATMUL = 0x5
} venus_opcode_t;

// 激活函数 Act
typedef enum { ACT_NONE = 0, ACT_RELU = 1, ACT_RELU6 = 2 } venus_act_t;

// ==========================================================
// 5. uOP 数据结构 (32 Bytes)
// ==========================================================
// 使用 packed 属性防止编译器插入 padding，确保与硬件 DMA 对齐
typedef struct {
  uint32_t w0; // Control 0: Op, Act, Pad, Stride, Size
  uint32_t w1; // Control 1: Channels
  uint32_t w2; // Control 2: Coordinates, Fi_Stride
  uint32_t w3; // Param Addr (Weights + Quant)
  uint32_t w4; // IFM Addr
  uint32_t w5; // OFM Addr
  uint32_t w6; // IFM_W/IFM_H
  uint32_t w7; // DMA_PRECALC
} __attribute__((packed, aligned(4))) venus_uop_t;

// uOP 固定大小常量（编译器/驱动都应使用该值）
#define VENUS_UOP_WORDS 8u
#define VENUS_UOP_SIZE_BYTES (VENUS_UOP_WORDS * 4u)

#endif // VENUS_HW_H
