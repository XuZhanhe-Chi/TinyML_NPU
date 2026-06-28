// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "board_result.h"
#include "bundle.h"
#include "kws_testvector_fpga.h"
#include "venus_driver.h"
#include "venus_hw.h"

#define ALIGN_UP_CONST(x, a) (((x) + ((a)-1u)) & ~((a)-1u))

#define SHARED_ALIGN_BYTES 64u
#define UOPS_OFFSET 0u
#define PARAMS_OFFSET ALIGN_UP_CONST(UOPS_OFFSET + UOPS_LEN_BYTES, SHARED_ALIGN_BYTES)
#define ACT_OFFSET ALIGN_UP_CONST(PARAMS_OFFSET + PARAMS_LEN_BYTES, SHARED_ALIGN_BYTES)
#define SHARED_USED_BYTES ALIGN_UP_CONST(ACT_OFFSET + ACTIVATION_PEAK_BYTES, SHARED_ALIGN_BYTES)
#define SHARED_TOTAL_BYTES (SHARED_MEM_HIGH - SHARED_MEM_BASE + 1u)
#if SHARED_USED_BYTES > TINYML_NPU_RESULT_OFFSET
#error "KWS demo bundle does not fit in the 128KB shared BRAM window"
#endif

#define VENUS_TIMEOUT_CYCLES 100000000u

static uint8_t *const g_uops_buf = (uint8_t *)(uintptr_t)(SHARED_MEM_BASE + UOPS_OFFSET);
static uint8_t *const g_params_buf = (uint8_t *)(uintptr_t)(SHARED_MEM_BASE + PARAMS_OFFSET);
static uint8_t *const g_act_buf = (uint8_t *)(uintptr_t)(SHARED_MEM_BASE + ACT_OFFSET);

static volatile tinyml_npu_board_result_t *const g_result =
    (volatile tinyml_npu_board_result_t *)(uintptr_t)TINYML_NPU_RESULT_ADDR;

static inline uint32_t mmio_read32(uint32_t off) {
  return *((volatile uint32_t *)(uintptr_t)(VENUS_REG_BASE + off));
}

static inline uint8_t packed_byte(const uint32_t *words, uint32_t byte_index) {
  const uint32_t word = words[byte_index >> 2];
  return (uint8_t)((word >> ((byte_index & 3u) * 8u)) & 0xFFu);
}

static void write_le32(uint8_t *dst, uint32_t word) {
  dst[0] = (uint8_t)(word & 0xFFu);
  dst[1] = (uint8_t)((word >> 8) & 0xFFu);
  dst[2] = (uint8_t)((word >> 16) & 0xFFu);
  dst[3] = (uint8_t)((word >> 24) & 0xFFu);
}

static void load_kws_input(void) {
  uint8_t *input = g_act_buf + INPUT_BASE;
  for (uint32_t i = 0; i < VC_KWS_INPUT_WORDS; ++i) {
    write_le32(input + i * 4u, VC_KWS_INPUT_WORDS_U8X4_LE[i]);
  }
  venus_cache_flush(input, INPUT_SIZE);
}

static int argmax_i8(const uint8_t *data, uint32_t count) {
  int best_idx = 0;
  int8_t best_val = (int8_t)data[0];
  for (uint32_t i = 1; i < count; ++i) {
    const int8_t v = (int8_t)data[i];
    if (v > best_val) {
      best_val = v;
      best_idx = (int)i;
    }
  }
  return best_idx;
}

static uint32_t compare_output(const uint8_t *out, uint32_t *max_abs_error) {
  uint32_t mismatches = 0;
  uint32_t max_error = 0;
  for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
    const uint8_t got = out[i];
    const uint8_t exp = packed_byte(VC_KWS_EXPECTED_OUTPUT_WORDS_U8X4_LE, i);
    const int32_t delta = (int32_t)(int8_t)got - (int32_t)(int8_t)exp;
    const uint32_t abs_error = (uint32_t)(delta < 0 ? -delta : delta);
    if (abs_error > max_error) {
      max_error = abs_error;
    }
    if (got != exp) {
      ++mismatches;
    }
  }
  *max_abs_error = max_error;
  return mismatches;
}

static void dump_output_mismatches(const uint8_t *out) {
  uint32_t printed = 0;
  for (uint32_t i = 0; i < OUTPUT_SIZE && printed < 8u; ++i) {
    const uint8_t got = out[i];
    const uint8_t exp = packed_byte(VC_KWS_EXPECTED_OUTPUT_WORDS_U8X4_LE, i);
    if (got != exp) {
      printf("mismatch[%lu]: got=%d exp=%d raw=0x%02X/0x%02X\r\n",
             (unsigned long)i, (int)(int8_t)got, (int)(int8_t)exp,
             (unsigned)got, (unsigned)exp);
      ++printed;
    }
  }
}

static void dump_failure_debug(uint32_t hw_status) {
  venus_debug_snapshot_t dbg;
  venus_read_debug_snapshot(&dbg);

  printf("STATUS=0x%08lX DEBUG0=0x%08lX DEBUG1=0x%08lX HW_STATUS=0x%08lX\r\n",
         (unsigned long)dbg.status, (unsigned long)dbg.debug0,
         (unsigned long)dbg.debug1, (unsigned long)hw_status);
  printf("fetch=%s sched=%s opcode=%s err=0x%02X flags=0x%03X\r\n",
         venus_dbg_fetch_state_str(dbg.dbg_fetch_state),
         venus_dbg_sched_state_str(dbg.dbg_sched_state),
         venus_dbg_opcode_str(dbg.dbg_curr_opcode),
         (unsigned)dbg.dbg_error_code, (unsigned)dbg.dbg_flags);
  printf("uop0: %08lX %08lX %08lX %08lX %08lX %08lX %08lX %08lX\r\n",
         (unsigned long)uops_words[0], (unsigned long)uops_words[1],
         (unsigned long)uops_words[2], (unsigned long)uops_words[3],
         (unsigned long)uops_words[4], (unsigned long)uops_words[5],
         (unsigned long)uops_words[6], (unsigned long)uops_words[7]);
}

static void clear_board_result(void) {
  memset((void *)g_result, 0, sizeof(*g_result));
  venus_cache_flush((const void *)g_result, sizeof(*g_result));
}

static void publish_board_result(uint32_t code, uint32_t hw_status, uint32_t top1,
                                 uint32_t mismatches, uint32_t max_abs_error) {
  venus_debug_snapshot_t dbg;
  venus_read_debug_snapshot(&dbg);

  g_result->magic = 0u;
  g_result->code = code;
  g_result->hw_status = hw_status;
  g_result->top1 = top1;
  g_result->mismatches = mismatches;
  g_result->max_abs_error = max_abs_error;
  g_result->status = dbg.status;
  g_result->debug0 = dbg.debug0;
  g_result->debug1 = dbg.debug1;
  venus_cache_flush((const void *)g_result, sizeof(*g_result));

  g_result->magic = TINYML_NPU_RESULT_MAGIC;
  venus_cache_flush((const void *)g_result, sizeof(g_result->magic));
}

int main(void) {
  clear_board_result();
  printf("\r\nTinyML_NPU ZYBO7010 KWS testvector demo\r\n");
  printf("NPU version: 0x%08lX\r\n", (unsigned long)mmio_read32(VENUS_REG_VERSION));
  printf("uOP count  : %lu\r\n", (unsigned long)(UOPS_LEN_WORDS / VENUS_UOP_WORDS));
  printf("shared BRAM: base=0x%08lX used=%lu bytes\r\n",
         (unsigned long)SHARED_MEM_BASE, (unsigned long)SHARED_USED_BYTES);

  memset(g_uops_buf, 0, SHARED_USED_BYTES);
  venus_cache_flush(g_uops_buf, SHARED_USED_BYTES);

  load_kws_input();

  venus_bundle_t bundle = VENUS_BUNDLE_FROM_BUNDLE_H();
  venus_mem_t mem;
  venus_mem_init(&mem, g_uops_buf, SHARED_MEM_BASE + UOPS_OFFSET, UOPS_LEN_BYTES,
                 g_params_buf, SHARED_MEM_BASE + PARAMS_OFFSET, PARAMS_LEN_BYTES,
                 g_act_buf, SHARED_MEM_BASE + ACT_OFFSET, ACTIVATION_PEAK_BYTES);
  venus_mem_set_npu_visible_range(&mem, SHARED_MEM_BASE, SHARED_MEM_HIGH);

  venus_init();

  uint32_t hw_status = 0;
  venus_status_t st = venus_run_bundle(&bundle, &mem, VENUS_TIMEOUT_CYCLES, &hw_status);
  if (st != VENUS_OK) {
    printf("venus_run_bundle failed: %s (%d)\r\n", venus_strerror(st), (int)st);
    dump_failure_debug(hw_status);
    publish_board_result(TINYML_NPU_RESULT_RUNTIME_FAILURE, hw_status,
                         0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    printf("TEST FAILED\r\n");
    return 1;
  }

  uint8_t *out = g_act_buf + OUTPUT_BASE;
  venus_cache_invalidate(out, OUTPUT_SIZE);

  const int top1 = argmax_i8(out, OUTPUT_SIZE);
  uint32_t max_abs_error = 0;
  const uint32_t mismatches = compare_output(out, &max_abs_error);
  printf("top1=%d expected=%lu\r\n", top1, (unsigned long)VC_KWS_EXPECTED_TOP1_I8);
  printf("output mismatches=%lu max_abs_error=%lu tolerance=%lu\r\n",
         (unsigned long)mismatches, (unsigned long)max_abs_error,
         (unsigned long)TINYML_NPU_MAX_ABS_ERROR);

  if ((uint32_t)top1 == VC_KWS_EXPECTED_TOP1_I8 &&
      max_abs_error <= TINYML_NPU_MAX_ABS_ERROR) {
    publish_board_result(TINYML_NPU_RESULT_PASS, hw_status, (uint32_t)top1,
                         mismatches, max_abs_error);
    printf("TEST PASSED\r\n");
    return 0;
  }

  dump_output_mismatches(out);
  dump_failure_debug(hw_status);
  publish_board_result(TINYML_NPU_RESULT_OUTPUT_FAILURE, hw_status,
                       (uint32_t)top1, mismatches, max_abs_error);
  printf("TEST FAILED\r\n");
  return 1;
}
