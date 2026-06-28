/*
 * venus_driver.c
 * -----------------------------------------------------------------------------
 * VenusCore NPU 驱动实现。
 *
 * 核心职责（从“固件/驱动”视角）：
 * 1) 把编译器生成的 bundle 工件（uops_words/params_words）复制到 DMA 可见内存；
 * 2) 根据地址模式对 uOP 做“重定位/翻译”（relocation）：
 *    - OFFSET 模式：给 W3/W4/W5 加上运行时基址；
 *    - ABSOLUTE 模式：uOP 内地址视为“物理地址”，驱动仅做必要校验；
 * 3) 配置寄存器并启动 NPU；
 * 4) 等待完成/超时，并返回明确错误码；
 * 5) （可选）做 cache flush/invalidate，保证 DMA 一致性。
 */

#include "venus_driver.h"

#include <math.h>
#include <string.h>

#include "xil_cache.h"
// =============================================================================
// 0) MMIO 读写
// =============================================================================

#define VENUS_MMIO32(addr) (*((volatile uint32_t *)(uintptr_t)(addr)))

static inline void venus_reg_write(uint32_t reg_off, uint32_t v) {
  VENUS_MMIO32(VENUS_REG_BASE + reg_off) = v;
}

static inline uint32_t venus_reg_read(uint32_t reg_off) {
  return VENUS_MMIO32(VENUS_REG_BASE + reg_off);
}

// =============================================================================
// 0.1) Debug snapshot helpers
// =============================================================================

static venus_debug_snapshot_t g_last_dbg_snapshot;
static uint32_t g_last_dbg_valid = 0u;

void venus_read_debug_snapshot(venus_debug_snapshot_t *out) {
  if (!out) {
    return;
  }

  out->uop_base = venus_reg_read(VENUS_REG_UOP_BASE);
  out->ctrl = venus_reg_read(VENUS_REG_CTRL);
  out->status = venus_reg_read(VENUS_REG_STATUS);
  out->uop_count = venus_reg_read(VENUS_REG_UOP_COUNT);
  out->int_en = venus_reg_read(VENUS_REG_INT_EN);
  out->int_status = venus_reg_read(VENUS_REG_INT_STATUS);
  out->debug0 = venus_reg_read(VENUS_REG_DEBUG0);
  out->debug1 = venus_reg_read(VENUS_REG_DEBUG1);

  const uint32_t d1 = out->debug1;
  out->dbg_error_code = (uint8_t)VENUS_DEBUG1_GET_ERROR_CODE(d1);
  out->dbg_curr_opcode = (uint8_t)VENUS_DEBUG1_GET_CURR_OPCODE(d1);
  out->dbg_sched_state = (uint8_t)VENUS_DEBUG1_GET_SCHED_STATE(d1);
  out->dbg_fetch_state = (uint8_t)VENUS_DEBUG1_GET_FETCH_STATE(d1);
  out->dbg_flags = (uint16_t)VENUS_DEBUG1_GET_FLAGS(d1);
}

const venus_debug_snapshot_t *venus_last_debug_snapshot(void) {
  return g_last_dbg_valid ? &g_last_dbg_snapshot
                          : (const venus_debug_snapshot_t *)0;
}

const char *venus_dbg_fetch_state_str(uint8_t fetch_state) {
  switch (fetch_state & 0xFu) {
  case 0:
    return "IDLE";
  case 1:
    return "REQ_DMA";
  case 2:
    return "WAIT_DATA";
  case 3:
    return "GEOM";
  case 4:
    return "PUSH_FIFO";
  default:
    return "UNKNOWN";
  }
}

const char *venus_dbg_sched_state_str(uint8_t sched_state) {
  switch (sched_state & 0xFu) {
  case 0:
    return "IDLE";
  case 1:
    return "RUN";
  case 2:
    return "DRAIN";
  case 3:
    return "DONE";
  case 4:
    return "ABORT";
  default:
    return "UNKNOWN";
  }
}

const char *venus_dbg_opcode_str(uint8_t opcode) {
  switch (opcode & 0xFu) {
  case 0:
    return "NOP";
  case 1:
    return "CONV2D";
  case 2:
    return "PWCONV";
  case 3:
    return "DWCONV";
  case 4:
    return "MATMUL";
  case 5:
    return "AVGPOOL";
  case 7:
    return "CFG";
  default:
    return "RESERVED";
  }
}

// =============================================================================
// 1) 错误码字符串
// =============================================================================

const char *venus_strerror(venus_status_t st) {
  switch (st) {
  case VENUS_OK:
    return "VENUS_OK";
  case VENUS_ERR_INVALID_ARG:
    return "VENUS_ERR_INVALID_ARG";
  case VENUS_ERR_NO_MEM:
    return "VENUS_ERR_NO_MEM";
  case VENUS_ERR_BUNDLE_FORMAT:
    return "VENUS_ERR_BUNDLE_FORMAT";
  case VENUS_ERR_RELOC_UNSUPPORTED:
    return "VENUS_ERR_RELOC_UNSUPPORTED";
  case VENUS_ERR_ADDR_RANGE:
    return "VENUS_ERR_ADDR_RANGE";
  case VENUS_ERR_ADDR_TRUNC:
    return "VENUS_ERR_ADDR_TRUNC";
  case VENUS_ERR_UNSUPPORTED:
    return "VENUS_ERR_UNSUPPORTED";
  case VENUS_ERR_TIMEOUT:
    return "VENUS_ERR_TIMEOUT";
  case VENUS_ERR_HARDWARE:
    return "VENUS_ERR_HARDWARE";
  default:
    return "VENUS_ERR_UNKNOWN";
  }
}

// =============================================================================
// 2) Cache hooks（weak 默认空实现）
// =============================================================================


#if defined(__GNUC__)
__attribute__((weak))
#endif
void venus_cache_flush(const void *addr, size_t size) {
  (void)addr;
  (void)size;
  Xil_DCacheFlushRange((INTPTR)addr,size);
}

#if defined(__GNUC__)
__attribute__((weak))
#endif
void venus_cache_invalidate(void *addr, size_t size) {
  (void)addr;
  (void)size;
    Xil_DCacheInvalidateRange((INTPTR)addr,size);
}

// =============================================================================
// 3) runtime memory（纯物理，不做 VA/映射）
// =============================================================================

void venus_mem_init(venus_mem_t *mem, void *uops_buf, uintptr_t uops_buf_pa,
                    uint32_t uops_buf_bytes, void *params_buf,
                    uintptr_t params_buf_pa, uint32_t params_buf_bytes,
                    void *act_buf, uintptr_t act_base_pa,
                    uint32_t act_buf_bytes) {
  if (!mem) {
    return;
  }
  mem->uops_buf = uops_buf;
  mem->uops_buf_pa = uops_buf_pa;
  mem->uops_buf_bytes = uops_buf_bytes;

  mem->params_buf = params_buf;
  mem->params_buf_pa = params_buf_pa;
  mem->params_buf_bytes = params_buf_bytes;

  mem->act_buf = act_buf;
  mem->act_base_pa = act_base_pa;
  mem->act_buf_bytes = act_buf_bytes;

  mem->npu_visible_base_pa = 0;
  mem->npu_visible_high_pa = 0;
}

void venus_mem_set_npu_visible_range(venus_mem_t *mem, uintptr_t base_pa,
                                     uintptr_t high_pa) {
  if (!mem) {
    return;
  }
  mem->npu_visible_base_pa = base_pa;
  mem->npu_visible_high_pa = high_pa;
}

// =============================================================================
// 4) NPU 控制：init / reset / abort / submit / wait
// =============================================================================

void venus_init(void) {
  // SOFT_RESET
  venus_reg_write(VENUS_REG_CTRL, CTRL_RESET_MASK);

  // 简单延时确保复位生效
  for (volatile uint32_t i = 0; i < 200u; ++i) {
    __asm__ volatile("" ::: "memory");
  }

  // 禁止中断 + 清除遗留中断状态（RW1C 写 1 清除）
  venus_reg_write(VENUS_REG_INT_EN, 0u);
  venus_reg_write(VENUS_REG_INT_STATUS, (INT_DONE_MASK | INT_ERROR_MASK));
}

void venus_soft_reset(void) {
  venus_reg_write(VENUS_REG_CTRL, CTRL_RESET_MASK);
}

void venus_abort(void) { venus_reg_write(VENUS_REG_CTRL, CTRL_ABORT_MASK); }

venus_status_t venus_submit_and_start(uintptr_t uop_base_pa,
                                      uint32_t uop_count) {
  if (uop_count == 0u) {
    return VENUS_ERR_INVALID_ARG;
  }

  // NPU 寄存器地址宽度 32-bit：64 位 host 指针无法直接用于真实硬件
  if (sizeof(uintptr_t) > 4u) {
    if (uop_base_pa > 0xFFFFFFFFu) {
      return VENUS_ERR_ADDR_TRUNC;
    }
  }

  // 清 pending 中断
  venus_reg_write(VENUS_REG_INT_STATUS, (INT_DONE_MASK | INT_ERROR_MASK));

  // 写 uOP base/count
  venus_reg_write(VENUS_REG_UOP_BASE, (uint32_t)uop_base_pa);
  venus_reg_write(VENUS_REG_UOP_COUNT, uop_count);

  // 启动
  venus_reg_write(VENUS_REG_CTRL, CTRL_START_MASK);
  return VENUS_OK;
}

venus_status_t venus_wait_idle(uint32_t timeout_cycles, uint32_t *out_status) {
  uint32_t status = 0;

  for (uint32_t i = 0; i < timeout_cycles; ++i) {
    status = venus_reg_read(VENUS_REG_STATUS);

    if (status & STATUS_ERROR_MASK) {
      if (out_status) {
        *out_status = status;
      }
      venus_read_debug_snapshot(&g_last_dbg_snapshot);
      g_last_dbg_valid = 1u;
      return VENUS_ERR_HARDWARE;
    }

    if ((status & STATUS_BUSY_MASK) == 0u) {
      if (out_status) {
        *out_status = status;
      }
      return VENUS_OK;
    }
  }

  if (out_status) {
    *out_status = status;
  }
  venus_read_debug_snapshot(&g_last_dbg_snapshot);
  g_last_dbg_valid = 1u;
  return VENUS_ERR_TIMEOUT;
}

// =============================================================================
// 5) 内部：拷贝/校验/patch
// =============================================================================

static inline void write_le_u32(uint8_t *dst4, uint32_t w) {
  dst4[0] = (uint8_t)(w & 0xFFu);
  dst4[1] = (uint8_t)((w >> 8) & 0xFFu);
  dst4[2] = (uint8_t)((w >> 16) & 0xFFu);
  dst4[3] = (uint8_t)((w >> 24) & 0xFFu);
}

static inline uint32_t read_le_u32(const uint8_t *src4) {
  return ((uint32_t)src4[0]) | ((uint32_t)src4[1] << 8) |
         ((uint32_t)src4[2] << 16) | ((uint32_t)src4[3] << 24);
}

static venus_status_t copy_words_to_buf(void *dst_buf, uint32_t dst_bytes,
                                        const uint32_t *src_words,
                                        size_t src_len_words) {
  if (!dst_buf || !src_words) {
    return VENUS_ERR_INVALID_ARG;
  }
  const uint32_t need = (uint32_t)(src_len_words * 4u);
  if (need > dst_bytes) {
    return VENUS_ERR_NO_MEM;
  }
  // bundle.h 导出的 uops_words/params_words 是“u32 words（小端）”。
  // 为了避免 CPU 端字节序差异（如 big-endian CPU），这里显式写成 little-endian
  // 字节序列，确保 NPU DMA 看到的总是 LE 布局。
  uint8_t *dst = (uint8_t *)dst_buf;
  for (size_t i = 0; i < src_len_words; ++i) {
    write_le_u32(&dst[i * 4u], src_words[i]);
  }
  return VENUS_OK;
}

static bool npu_visible_enabled(const venus_mem_t *mem) {
  if (!mem) {
    return false;
  }
  if (mem->npu_visible_base_pa == 0u && mem->npu_visible_high_pa == 0u) {
    return false;
  }
  return mem->npu_visible_high_pa >= mem->npu_visible_base_pa;
}

static venus_status_t check_npu_visible_range(const venus_mem_t *mem,
                                              uintptr_t pa,
                                              uint32_t bytes) {
  if (!mem || bytes == 0u) {
    return VENUS_ERR_INVALID_ARG;
  }
  if (!npu_visible_enabled(mem)) {
    return VENUS_OK;
  }

  const uintptr_t base = mem->npu_visible_base_pa;
  const uintptr_t high = mem->npu_visible_high_pa;
  const uintptr_t end = pa + (uintptr_t)bytes - 1u;
  if (end < pa) {
    return VENUS_ERR_INVALID_ARG;
  }
  if (pa < base || end > high) {
    return VENUS_ERR_ADDR_RANGE;
  }
  return VENUS_OK;
}

static venus_status_t check_pa_32bit(uintptr_t pa) {
  if (sizeof(uintptr_t) > 4u) {
    if (pa > 0xFFFFFFFFu) {
      return VENUS_ERR_ADDR_TRUNC;
    }
  }
  return VENUS_OK;
}

static venus_status_t patch_uops_offset_mode(uint8_t *uops_buf,
                                             size_t uops_len_words,
                                             uintptr_t act_base_pa,
                                             uintptr_t param_base_pa) {
  if (!uops_buf) {
    return VENUS_ERR_INVALID_ARG;
  }
  if ((uops_len_words % VENUS_UOP_WORDS) != 0u) {
    return VENUS_ERR_BUNDLE_FORMAT;
  }

  if (sizeof(uintptr_t) > 4u) {
    if (act_base_pa > 0xFFFFFFFFu || param_base_pa > 0xFFFFFFFFu) {
      return VENUS_ERR_ADDR_TRUNC;
    }
  }

  const uint32_t act32 = (uint32_t)act_base_pa;
  const uint32_t par32 = (uint32_t)param_base_pa;
  const size_t uop_cnt = uops_len_words / VENUS_UOP_WORDS;

  for (size_t i = 0; i < uop_cnt; ++i) {
    const size_t base_word = i * VENUS_UOP_WORDS;
    const size_t base_byte = base_word * 4u;

    const uint32_t w3 = read_le_u32(&uops_buf[base_byte + 3u * 4u]);
    const uint32_t w4 = read_le_u32(&uops_buf[base_byte + 4u * 4u]);
    const uint32_t w5 = read_le_u32(&uops_buf[base_byte + 5u * 4u]);

    write_le_u32(&uops_buf[base_byte + 3u * 4u],
                 (uint32_t)(w3 + par32)); // PARAM_ADDR += param_base
    write_le_u32(&uops_buf[base_byte + 4u * 4u],
                 (uint32_t)(w4 + act32)); // FI_ADDR    += act_base
    write_le_u32(&uops_buf[base_byte + 5u * 4u],
                 (uint32_t)(w5 + act32)); // FO_ADDR    += act_base
  }
  return VENUS_OK;
}

static venus_status_t patch_uops_absolute_mode(uint8_t *uops_buf,
                                               size_t uops_len_words,
                                               const venus_bundle_t *bundle,
                                               const venus_mem_t *mem) {
  if (!uops_buf || !bundle || !mem) {
    return VENUS_ERR_INVALID_ARG;
  }
  if ((uops_len_words % VENUS_UOP_WORDS) != 0u) {
    return VENUS_ERR_BUNDLE_FORMAT;
  }

  if (check_pa_32bit(mem->params_buf_pa) != VENUS_OK) {
    return VENUS_ERR_ADDR_TRUNC;
  }
  if ((uint32_t)mem->params_buf_pa != bundle->param_base) {
    return VENUS_ERR_RELOC_UNSUPPORTED;
  }

  venus_status_t st;
  st = check_npu_visible_range(mem, mem->uops_buf_pa,
                               (uint32_t)(uops_len_words * 4u));
  if (st != VENUS_OK) {
    return st;
  }
  st = check_npu_visible_range(mem, mem->params_buf_pa,
                               (uint32_t)(bundle->params_len_words * 4u));
  if (st != VENUS_OK) {
    return st;
  }
  if (mem->act_buf_bytes != 0u) {
    st = check_npu_visible_range(mem, mem->act_base_pa, mem->act_buf_bytes);
    if (st != VENUS_OK) {
      return st;
    }
  }

  const size_t uop_cnt = uops_len_words / VENUS_UOP_WORDS;
  for (size_t i = 0; i < uop_cnt; ++i) {
    const size_t base_word = i * VENUS_UOP_WORDS;
    const size_t base_byte = base_word * 4u;

    const uint32_t pa_param = read_le_u32(&uops_buf[base_byte + 3u * 4u]);
    const uint32_t pa_ifm = read_le_u32(&uops_buf[base_byte + 4u * 4u]);
    const uint32_t pa_ofm = read_le_u32(&uops_buf[base_byte + 5u * 4u]);

    st = check_pa_32bit((uintptr_t)pa_param);
    if (st != VENUS_OK)
      return st;
    st = check_pa_32bit((uintptr_t)pa_ifm);
    if (st != VENUS_OK)
      return st;
    st = check_pa_32bit((uintptr_t)pa_ofm);
    if (st != VENUS_OK)
      return st;

    st = check_npu_visible_range(mem, (uintptr_t)pa_param, 1u);
    if (st != VENUS_OK)
      return st;
    st = check_npu_visible_range(mem, (uintptr_t)pa_ifm, 1u);
    if (st != VENUS_OK)
      return st;
    st = check_npu_visible_range(mem, (uintptr_t)pa_ofm, 1u);
    if (st != VENUS_OK)
      return st;
  }

  return VENUS_OK;
}

// =============================================================================
// 6) 一键运行：venus_run_bundle
// =============================================================================

#if (VC_HAS_PLAN != 0u)
static const vc_tensor_desc_t *find_tensor_desc(const vc_plan_t *plan,
                                                uint16_t tensor_id) {
  if (!plan || !plan->tensors) {
    return NULL;
  }
  if (tensor_id < plan->tensor_count) {
    const vc_tensor_desc_t *t = &plan->tensors[tensor_id];
    if (t->tensor_id == tensor_id) {
      return t;
    }
  }
  for (uint32_t i = 0; i < plan->tensor_count; ++i) {
    const vc_tensor_desc_t *t = &plan->tensors[i];
    if (t->tensor_id == tensor_id) {
      return t;
    }
  }
  return NULL;
}

static int get_tensor_scale(const vc_plan_t *plan, const vc_tensor_desc_t *t,
                            float *out_scale) {
  if (!out_scale) {
    return 0;
  }
  *out_scale = 0.0f;
  if (!plan || !t || !plan->quant_scales) {
    return 0;
  }
  if (t->quant_index == 0xFFFFu) {
    return 0;
  }
  if (t->quant_index >= plan->quant_scale_count) {
    return 0;
  }
  *out_scale = plan->quant_scales[t->quant_index];
  return (*out_scale > 0.0f) ? 1 : 0;
}

static venus_status_t tensor_data_ptr(const venus_bundle_t *bundle,
                                      const venus_mem_t *mem,
                                      const vc_tensor_desc_t *t,
                                      uint8_t **out_ptr) {
  if (!bundle || !mem || !t || !out_ptr) {
    return VENUS_ERR_INVALID_ARG;
  }
  if (bundle->address_mode_offset != 0u) {
    if (!mem->act_buf) {
      return VENUS_ERR_INVALID_ARG;
    }
    if ((uint32_t)t->offset_bytes + (uint32_t)t->size_bytes >
        mem->act_buf_bytes) {
      return VENUS_ERR_ADDR_RANGE;
    }
    *out_ptr = (uint8_t *)mem->act_buf + (uint32_t)t->offset_bytes;
    return VENUS_OK;
  }

  if (!mem->act_buf || mem->act_buf_bytes == 0u) {
    return VENUS_ERR_RELOC_UNSUPPORTED;
  }
  const uintptr_t pa = (uintptr_t)t->offset_bytes;
  const uintptr_t base = mem->act_base_pa;
  const uintptr_t end = pa + (uintptr_t)t->size_bytes;
  if (end < pa) {
    return VENUS_ERR_INVALID_ARG;
  }
  if (pa < base || (end - 1u) >= (base + (uintptr_t)mem->act_buf_bytes)) {
    return VENUS_ERR_ADDR_RANGE;
  }
  *out_ptr = (uint8_t *)mem->act_buf + (uint32_t)(pa - base);
  return VENUS_OK;
}

static inline int8_t sat_i8(int32_t x) {
  if (x > 127) {
    return 127;
  }
  if (x < -128) {
    return -128;
  }
  return (int8_t)x;
}

static void cpu_add_sat(uint8_t *dst, const uint8_t *a, const uint8_t *b,
                        size_t size_bytes) {
  if (!dst || !a || !b) {
    return;
  }
  for (size_t i = 0; i < size_bytes; ++i) {
    int32_t s = (int32_t)(int8_t)a[i] + (int32_t)(int8_t)b[i];
    dst[i] = (uint8_t)sat_i8(s);
  }
}

static void cpu_add_requant(uint8_t *dst, const uint8_t *a, float a_scale,
                            const uint8_t *b, float b_scale, float y_scale,
                            size_t size_bytes) {
  if (!dst || !a || !b) {
    return;
  }
  if (!(a_scale > 0.0f) || !(b_scale > 0.0f) || !(y_scale > 0.0f)) {
    return;
  }
  const float inv_y = 1.0f / y_scale;
  for (size_t i = 0; i < size_bytes; ++i) {
    const int32_t va = (int32_t)(int8_t)a[i];
    const int32_t vb = (int32_t)(int8_t)b[i];
    const float yf = ((float)va * a_scale + (float)vb * b_scale) * inv_y;
    dst[i] = (uint8_t)sat_i8((int32_t)lrintf(yf));
  }
}

static int cpu_concat_c_i8_nchwc4(uint8_t *dst, const uint8_t *const *srcs,
                                  size_t src_count, int n, int h, int w,
                                  const int *c_list, int c_out) {
  if (!dst || !srcs || !c_list || src_count < 2u) {
    return -1;
  }
  if (n != 1 || h <= 0 || w <= 0) {
    return -2;
  }

  const size_t plane_words = (size_t)h * (size_t)w;
  const size_t plane_bytes = plane_words * 4u;

  const int c4_out = (c_out + 3) / 4;
  const size_t dst_bytes = (size_t)c4_out * plane_bytes;
  memset(dst, 0, dst_bytes);

  int out_c4_base = 0;
  int c_sum = 0;
  for (size_t si = 0; si < src_count; ++si) {
    const uint8_t *src = srcs[si];
    const int c = c_list[si];
    if (!src || c <= 0) {
      return -3;
    }
    c_sum += c;
    const int c4 = (c + 3) / 4;
    const int rem = c & 3;
    for (int g = 0; g < c4; ++g) {
      const uint8_t *src_plane = src + (size_t)g * plane_bytes;
      uint8_t *dst_plane = dst + (size_t)(out_c4_base + g) * plane_bytes;
      memcpy(dst_plane, src_plane, plane_bytes);
      if ((g == (c4 - 1)) && (rem != 0)) {
        for (size_t wi = 0; wi < plane_words; ++wi) {
          uint8_t *word = dst_plane + wi * 4u;
          for (int k = rem; k < 4; ++k) {
            word[k] = 0;
          }
        }
      }
    }
    out_c4_base += c4;
  }
  if (c_sum != c_out) {
    return -4;
  }
  if (out_c4_base > c4_out) {
    return -5;
  }
  return 0;
}

static venus_status_t run_plan(const venus_bundle_t *bundle,
                               const venus_mem_t *mem, uint32_t timeout_cycles,
                               uint32_t *out_hw_status) {
  if (!bundle || !mem || !bundle->plan) {
    return VENUS_ERR_INVALID_ARG;
  }
  const vc_plan_t *plan = bundle->plan;
  if (!plan->steps || !plan->tensors) {
    return VENUS_ERR_BUNDLE_FORMAT;
  }

  for (uint32_t si = 0; si < plan->step_count; ++si) {
    const vc_step_desc_t *s = &plan->steps[si];
    if (!s) {
      continue;
    }

    if (s->step_type == VC_STEP_NPU) {
      if ((s->uop_off_words + s->uop_words) > bundle->uops_len_words) {
        return VENUS_ERR_BUNDLE_FORMAT;
      }
      if ((s->uop_off_words % VENUS_UOP_WORDS) != 0u ||
          (s->uop_words % VENUS_UOP_WORDS) != 0u) {
        return VENUS_ERR_BUNDLE_FORMAT;
      }

      const uintptr_t step_uops_pa =
          mem->uops_buf_pa + (uintptr_t)s->uop_off_words * 4u;
      const uint32_t step_uop_count =
          (uint32_t)(s->uop_words / VENUS_UOP_WORDS);

      venus_status_t st = venus_submit_and_start(step_uops_pa, step_uop_count);
      if (st != VENUS_OK) {
        return st;
      }
      uint32_t hw_status = 0;
      st = venus_wait_idle(timeout_cycles, &hw_status);
      if (out_hw_status) {
        *out_hw_status = hw_status;
      }
      if (st != VENUS_OK) {
        return st;
      }
      continue;
    }

    if (s->step_type == VC_STEP_ALIAS) {
      continue;
    }

    if (s->step_type != VC_STEP_CPU) {
      return VENUS_ERR_UNSUPPORTED;
    }

    if (s->cpu_kernel == VC_CPU_ADD) {
      if (s->input_count != 2u || s->output_count != 1u) {
        return VENUS_ERR_BUNDLE_FORMAT;
      }
      const vc_tensor_desc_t *ta = find_tensor_desc(plan, s->inputs[0]);
      const vc_tensor_desc_t *tb = find_tensor_desc(plan, s->inputs[1]);
      const vc_tensor_desc_t *ty = find_tensor_desc(plan, s->outputs[0]);
      if (!ta || !tb || !ty) {
        return VENUS_ERR_BUNDLE_FORMAT;
      }

      uint8_t *a = NULL;
      uint8_t *b = NULL;
      uint8_t *y = NULL;
      venus_status_t st;
      st = tensor_data_ptr(bundle, mem, ta, &a);
      if (st != VENUS_OK)
        return st;
      st = tensor_data_ptr(bundle, mem, tb, &b);
      if (st != VENUS_OK)
        return st;
      st = tensor_data_ptr(bundle, mem, ty, &y);
      if (st != VENUS_OK)
        return st;

      venus_cache_invalidate(a, (size_t)ta->size_bytes);
      venus_cache_invalidate(b, (size_t)tb->size_bytes);

      float sa = 0.0f, sb = 0.0f, sy = 0.0f;
      const int ha = get_tensor_scale(plan, ta, &sa);
      const int hb = get_tensor_scale(plan, tb, &sb);
      const int hy = get_tensor_scale(plan, ty, &sy);
      if (ha && hb && hy && (sa != sb || sa != sy)) {
        cpu_add_requant(y, a, sa, b, sb, sy, (size_t)ty->size_bytes);
      } else {
        cpu_add_sat(y, a, b, (size_t)ty->size_bytes);
      }

      venus_cache_flush(y, (size_t)ty->size_bytes);
      continue;
    }

    if (s->cpu_kernel == VC_CPU_CONCAT_C) {
      if (s->input_count < 2u || s->output_count != 1u) {
        return VENUS_ERR_BUNDLE_FORMAT;
      }
      const vc_tensor_desc_t *ty = find_tensor_desc(plan, s->outputs[0]);
      if (!ty) {
        return VENUS_ERR_BUNDLE_FORMAT;
      }
      uint8_t *y = NULL;
      venus_status_t st = tensor_data_ptr(bundle, mem, ty, &y);
      if (st != VENUS_OK)
        return st;

      const uint8_t *srcs[4] = {0};
      int c_list[4] = {0};
      for (uint32_t i = 0; i < s->input_count && i < 4u; ++i) {
        const vc_tensor_desc_t *tx = find_tensor_desc(plan, s->inputs[i]);
        if (!tx) {
          return VENUS_ERR_BUNDLE_FORMAT;
        }
        uint8_t *x = NULL;
        st = tensor_data_ptr(bundle, mem, tx, &x);
        if (st != VENUS_OK)
          return st;
        venus_cache_invalidate(x, (size_t)tx->size_bytes);
        srcs[i] = x;
        c_list[i] = (int)tx->c;
      }

      const int rc = cpu_concat_c_i8_nchwc4(
          y, srcs, (size_t)s->input_count, (int)ty->n, (int)ty->h, (int)ty->w,
          c_list, (int)ty->c);
      if (rc != 0) {
        return VENUS_ERR_UNSUPPORTED;
      }
      venus_cache_flush(y, (size_t)ty->size_bytes);
      continue;
    }

    return VENUS_ERR_UNSUPPORTED;
  }

  return VENUS_OK;
}
#endif /* VC_HAS_PLAN */

venus_status_t venus_run_bundle(const venus_bundle_t *bundle,
                                const venus_mem_t *mem,
                                uint32_t timeout_cycles,
                                uint32_t *out_hw_status) {
  if (!bundle || !mem) {
    return VENUS_ERR_INVALID_ARG;
  }
  if (!mem->uops_buf || !mem->params_buf) {
    return VENUS_ERR_INVALID_ARG;
  }
  if (bundle->uops_len_words == 0u || bundle->params_len_words == 0u) {
    return VENUS_ERR_BUNDLE_FORMAT;
  }
  if ((bundle->uops_len_words % VENUS_UOP_WORDS) != 0u) {
    return VENUS_ERR_BUNDLE_FORMAT;
  }

  venus_status_t st;
  st = check_pa_32bit(mem->uops_buf_pa);
  if (st != VENUS_OK)
    return st;
  st = check_pa_32bit(mem->params_buf_pa);
  if (st != VENUS_OK)
    return st;
  st = check_pa_32bit(mem->act_base_pa);
  if (st != VENUS_OK)
    return st;

  const uint32_t uops_bytes32 = (uint32_t)(bundle->uops_len_words * 4u);
  st = check_npu_visible_range(mem, mem->uops_buf_pa, uops_bytes32);
  if (st != VENUS_OK)
    return st;
  st = check_npu_visible_range(mem, mem->params_buf_pa,
                               (uint32_t)(bundle->params_len_words * 4u));
  if (st != VENUS_OK)
    return st;
  if (mem->act_buf_bytes != 0u) {
    st = check_npu_visible_range(mem, mem->act_base_pa, mem->act_buf_bytes);
    if (st != VENUS_OK)
      return st;
  }

  if (bundle->address_mode_offset != 0u) {
    if (!mem->act_buf || mem->act_buf_bytes == 0u) {
      return VENUS_ERR_INVALID_ARG;
    }
  }

  // 1) 拷贝 params
  st = copy_words_to_buf(mem->params_buf, mem->params_buf_bytes,
                         bundle->params_words, bundle->params_len_words);
  if (st != VENUS_OK)
    return st;

  venus_cache_flush(mem->params_buf,
                    (size_t)(bundle->params_len_words * 4u));

  // 2) 拷贝 uops
  st = copy_words_to_buf(mem->uops_buf, mem->uops_buf_bytes,
                         bundle->uops_words, bundle->uops_len_words);
  if (st != VENUS_OK)
    return st;

  // 3) 重定位/翻译 uops
  uint8_t *uops_rw = (uint8_t *)mem->uops_buf;
  if (bundle->address_mode_offset != 0u) {
    st = patch_uops_offset_mode(uops_rw, bundle->uops_len_words,
                                mem->act_base_pa, mem->params_buf_pa);
    if (st != VENUS_OK)
      return st;
  } else {
    st = patch_uops_absolute_mode(uops_rw, bundle->uops_len_words, bundle, mem);
    if (st != VENUS_OK)
      return st;
  }

  const size_t uops_bytes = (size_t)(bundle->uops_len_words * 4u);
  venus_cache_flush(mem->uops_buf, uops_bytes);

  // 4) 执行：Plan step-by-step（若存在），否则执行全量 uOP 一次
#if (VC_HAS_PLAN != 0u)
  if (bundle->plan) {
    __asm__ volatile("dmb sy\n dsb sy" ::: "memory");
    st = run_plan(bundle, mem, timeout_cycles, out_hw_status);
    if (st != VENUS_OK)
      return st;
  } else
#endif
  {
    const uint32_t uop_count =
        (uint32_t)(bundle->uops_len_words / VENUS_UOP_WORDS);
    st = venus_submit_and_start(mem->uops_buf_pa, uop_count);
    if (st != VENUS_OK)
      return st;

    uint32_t hw_status = 0;
    st = venus_wait_idle(timeout_cycles, &hw_status);
    if (out_hw_status) {
      *out_hw_status = hw_status;
    }
    if (st != VENUS_OK)
      return st;
  }
  __asm__ volatile("dmb sy\n dsb sy" ::: "memory");
  return VENUS_OK;
}
