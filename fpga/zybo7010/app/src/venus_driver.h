/*
 * venus_driver.h
 * -----------------------------------------------------------------------------
 * VenusCore NPU 驱动（C 语言，面向 bare-metal / RTOS / SoC 固件）。
 *
 * 设计目标：
 * 1) “以 bundle.h 为主入口”：驱动不依赖具体模型，但能够直接消费编译器生成的
 * bundle.h 导出的 uops_words/params_words 及宏信息； 2) 支持地址重定位：
 *    - OFFSET 模式：按 act_base / param_base 做 uOP patch（推荐固件集成）；
 *    - ABSOLUTE 模式：uOP 内地址视为“物理地址”，驱动仅做必要校验； 3)
 * 支持超时与可诊断错误码：
 *    - 轮询等待 NPU 完成时可设定超时；
 *    - 出错时返回软件错误码，并可读取硬件 STATUS.ERROR_CODE。
 *
 * 使用方式（典型）：
 *   #include "bundle.h"
 *   #include "venus_driver.h"
 *
 *   venus_bundle_t bundle = VENUS_BUNDLE_FROM_BUNDLE_H();
 *   venus_mem_t mem; ... 初始化 staging/arena ...
 *   venus_init();
 *   venus_run_bundle(&bundle, &mem, timeout_cycles);
 */

#ifndef VENUS_DRIVER_H
#define VENUS_DRIVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bundle.h"
#include "venus_hw.h"

#ifndef VC_HAS_PLAN
#define VC_HAS_PLAN 0u
#endif

#if (VC_HAS_PLAN == 0u)
/* Old bundles may not define vc_plan_t; keep it opaque when unused. */
typedef struct vc_plan_t vc_plan_t;
#endif

#if (VC_HAS_PLAN != 0u)
#define VENUS_BUNDLE_PLAN_PTR ((const vc_plan_t *)&VC_PLAN)
#else
#define VENUS_BUNDLE_PLAN_PTR ((const vc_plan_t *)0)
#endif

// =============================================================================
// 1) 错误码定义（负值表示失败，0 表示成功）
// =============================================================================

typedef enum {
  VENUS_OK = 0,

  // --- 参数/配置错误（软件侧） ---
  VENUS_ERR_INVALID_ARG = -1, // 传入参数为空/越界/不合法
  VENUS_ERR_NO_MEM = -2,      // staging buffer 容量不足
  VENUS_ERR_BUNDLE_FORMAT = -3, // bundle 描述与 uOP 格式不匹配（长度/对齐等）
  VENUS_ERR_RELOC_UNSUPPORTED = -4, // 不支持的重定位模式/缺少必要信息
  VENUS_ERR_ADDR_RANGE = -5, // 地址不在可访问范围（例如不在共享内存窗口内）
  VENUS_ERR_ADDR_TRUNC = -6, // 物理地址无法塞进 32bit（常见于 64 位 host 指针）
  VENUS_ERR_UNSUPPORTED = -7, // 运行时遇到不支持的 plan step / CPU kernel

  // --- 运行时错误（硬件/超时） ---
  VENUS_ERR_TIMEOUT = -10,  // 等待 NPU 完成超时
  VENUS_ERR_HARDWARE = -11, // 硬件 STATUS.ERROR 置位
} venus_status_t;

// 将错误码转成可读字符串（便于打印日志）。
const char *venus_strerror(venus_status_t st);

// =============================================================================
// 2) bundle 描述：从 bundle.h 提取并封装
// =============================================================================

typedef struct {
  // --- 工件数据（来自 bundle.h） ---
  const uint32_t *uops_words; // uOP 以 uint32 word 序列导出（小端）
  size_t uops_len_words;      // word 数
  const uint32_t *params_words; // Param Block 以 uint32 word 序列导出（小端）
  size_t params_len_words; // word 数

  // --- Plan（可选，VC_HAS_PLAN!=0 时存在） ---
  const vc_plan_t *plan; // NULL 表示仅使用 legacy “run full uOP stream”

  // --- 元信息（来自 bundle.h 宏） ---
  uint32_t address_mode_offset; // 1=offset 模式，0=absolute 模式

  // 编译器给出的基址信息：
  // - offset 模式：这里是“偏移值”（byte offset）
  // - absolute 模式：这里是“物理地址”（byte address）
  uint32_t input_base;
  uint32_t input_size;
  uint32_t output_base;
  uint32_t output_size;

  uint32_t param_base;
  uint32_t param_block_size;
  uint32_t activation_peak_bytes;
} venus_bundle_t;

/*
 * 便捷宏：在“main.c 里包含了 bundle.h 之后”使用。
 *
 * 说明：
 * - bundle.h 中的 uops_words/params_words 通常是 static const（Flash/ROM
 * 常量）。
 * - 驱动运行前会把 uops 复制到可写 staging buffer 并按需要 patch。
 */
#define VENUS_BUNDLE_FROM_BUNDLE_H()                                           \
  ((venus_bundle_t){                                                           \
      .uops_words = uops_words,                                                \
      .uops_len_words = uops_words_len_words,                                  \
      .params_words = params_words,                                            \
      .params_len_words = params_words_len_words,                              \
      .plan = VENUS_BUNDLE_PLAN_PTR,                                           \
      .address_mode_offset = (uint32_t)ADDRESS_MODE_OFFSET,                    \
      .input_base = (uint32_t)INPUT_BASE,                                      \
      .input_size = (uint32_t)INPUT_SIZE,                                      \
      .output_base = (uint32_t)OUTPUT_BASE,                                    \
      .output_size = (uint32_t)OUTPUT_SIZE,                                    \
      .param_base = (uint32_t)PARAM_BASE,                                      \
      .param_block_size = (uint32_t)PARAM_BLOCK_SIZE,                          \
      .activation_peak_bytes = (uint32_t)ACTIVATION_PEAK_BYTES,                \
  })

// =============================================================================
// 3) 运行时内存描述（纯物理/共享内存，不涉及虚拟地址）
// =============================================================================

typedef struct {
  // staging buffer：驱动会把 uops/params 复制到这里（必须 NPU DMA 可见）
  void *uops_buf;         // CPU 可写指针
  uintptr_t uops_buf_pa;  // NPU DMA 视角物理地址（写入 NPU_UOP_BASE）
  uint32_t uops_buf_bytes;

  void *params_buf;        // CPU 可写指针
  uintptr_t params_buf_pa; // NPU DMA 视角物理地址（uOP 的 PARAM_ADDR 会指向这里）
  uint32_t params_buf_bytes;

  // activation arena：offset 模式下用于 FI/FO；Plan CPU step 也会在此读写
  void *act_buf;           // CPU 指针
  uintptr_t act_base_pa;   // NPU DMA 视角物理基址
  uint32_t act_buf_bytes;  // arena 大小（用于 CPU step / 可选范围校验）

  // 可选：NPU 可访问的共享内存窗口（用于范围校验；0 表示不校验）
  uintptr_t npu_visible_base_pa;
  uintptr_t npu_visible_high_pa; // inclusive
} venus_mem_t;

// 初始化 runtime memory（纯物理，不做 VA/映射）。
void venus_mem_init(venus_mem_t *mem, void *uops_buf, uintptr_t uops_buf_pa,
                    uint32_t uops_buf_bytes, void *params_buf,
                    uintptr_t params_buf_pa, uint32_t params_buf_bytes,
                    void *act_buf, uintptr_t act_base_pa,
                    uint32_t act_buf_bytes);

// 设置 NPU 可访问地址范围（可选）。若设置，驱动会在运行前检查：
// - uops_buf_pa / params_buf_pa / act_base_pa 以及 uOP 内 W3/W4/W5 落在范围内。
void venus_mem_set_npu_visible_range(venus_mem_t *mem, uintptr_t base_pa,
                                    uintptr_t high_pa);

// =============================================================================
// 4) Cache 一致性 hooks（可选）
// =============================================================================

void venus_cache_flush(const void *addr, size_t size);
void venus_cache_invalidate(void *addr, size_t size);

// =============================================================================
// 5) NPU 控制 API
// =============================================================================

void venus_init(void);
void venus_soft_reset(void);
void venus_abort(void);

venus_status_t venus_submit_and_start(uintptr_t uop_base_pa,
                                      uint32_t uop_count);
venus_status_t venus_wait_idle(uint32_t timeout_cycles, uint32_t *out_status);

// =============================================================================
// 5.1) Debug registers snapshot (for timeout diagnosis)
// =============================================================================

typedef struct {
  // Raw register reads
  uint32_t uop_base;
  uint32_t uop_count;
  uint32_t ctrl;
  uint32_t status;
  uint32_t int_en;
  uint32_t int_status;
  uint32_t debug0;
  uint32_t debug1;

  // Decoded DEBUG1 fields
  uint8_t dbg_error_code;
  uint8_t dbg_curr_opcode;
  uint8_t dbg_sched_state;
  uint8_t dbg_fetch_state;
  uint16_t dbg_flags; // 12-bit valid
} venus_debug_snapshot_t;

// Read a consistent snapshot of key regs (STATUS/INT/DEBUG) for diagnosis.
void venus_read_debug_snapshot(venus_debug_snapshot_t *out);

// Returns the last snapshot captured by venus_wait_idle() on timeout/error.
// (Returns NULL if no snapshot has been captured yet.)
const venus_debug_snapshot_t *venus_last_debug_snapshot(void);

// Optional string helpers for printing (no allocation).
const char *venus_dbg_fetch_state_str(uint8_t fetch_state);
const char *venus_dbg_sched_state_str(uint8_t sched_state);
const char *venus_dbg_opcode_str(uint8_t opcode);

// =============================================================================
// 6) 一键运行：准备工件 -> 绑定寄存器 -> 启动 -> 等待
// =============================================================================

venus_status_t venus_run_bundle(const venus_bundle_t *bundle,
                                const venus_mem_t *mem,
                                uint32_t timeout_cycles,
                                uint32_t *out_hw_status);

#endif // VENUS_DRIVER_H
