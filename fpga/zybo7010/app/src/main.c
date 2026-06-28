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

#if VC_KWS_SAMPLE_COUNT < VC_KWS_MIN_SAMPLE_COUNT
#error "KWS board demo must carry at least 100 samples"
#endif

#if VC_KWS_SAMPLE_COUNT != TINYML_NPU_EXPECTED_SAMPLE_COUNT
#error "KWS sample count must match board result ABI constants"
#endif

#if VC_KWS_EXPECTED_REF_TOP1_MATCH != TINYML_NPU_EXPECTED_REF_TOP1_MATCH
#error "KWS expected reference top1 count must match board result ABI constants"
#endif

#if VC_KWS_EXPECTED_LABEL_CORRECT != TINYML_NPU_MIN_LABEL_CORRECT
#error "KWS expected label accuracy count must match board result ABI constants"
#endif

#if VC_KWS_INPUT_BYTES != INPUT_SIZE
#error "KWS input testvector size does not match compiled bundle INPUT_SIZE"
#endif

#if VC_KWS_OUTPUT_C != OUTPUT_SIZE
#error "KWS output class count does not match compiled bundle OUTPUT_SIZE"
#endif

#define VENUS_TIMEOUT_CYCLES 100000000u

typedef struct {
  uint32_t sample_count;
  uint32_t label_correct;
  uint32_t ref_top1_match;
  uint32_t max_abs_error;
  uint32_t total_mismatches;
  uint32_t first_failure_sample;
  uint32_t first_failure_top1;
  uint32_t first_failure_expected_top1;
  uint32_t first_failure_label;
  uint32_t total_cycles;
} kws_run_stats_t;

static uint8_t *const g_uops_buf = (uint8_t *)(uintptr_t)(SHARED_MEM_BASE + UOPS_OFFSET);
static uint8_t *const g_params_buf = (uint8_t *)(uintptr_t)(SHARED_MEM_BASE + PARAMS_OFFSET);
static uint8_t *const g_act_buf = (uint8_t *)(uintptr_t)(SHARED_MEM_BASE + ACT_OFFSET);

static volatile tinyml_npu_board_result_t *const g_result =
    (volatile tinyml_npu_board_result_t *)(uintptr_t)TINYML_NPU_RESULT_ADDR;

static inline uint32_t mmio_read32(uint32_t off) {
  return *((volatile uint32_t *)(uintptr_t)(VENUS_REG_BASE + off));
}

static void init_stats(kws_run_stats_t *stats) {
  memset(stats, 0, sizeof(*stats));
  stats->first_failure_sample = UINT32_MAX;
  stats->first_failure_top1 = UINT32_MAX;
  stats->first_failure_expected_top1 = UINT32_MAX;
  stats->first_failure_label = UINT32_MAX;
}

static void remember_first_failure(kws_run_stats_t *stats, uint32_t sample,
                                   uint32_t top1, uint32_t expected_top1,
                                   uint32_t label) {
  if (stats->first_failure_sample == UINT32_MAX) {
    stats->first_failure_sample = sample;
    stats->first_failure_top1 = top1;
    stats->first_failure_expected_top1 = expected_top1;
    stats->first_failure_label = label;
  }
}

static void load_kws_input(uint32_t sample_idx) {
  uint8_t *input = g_act_buf + INPUT_BASE;
  const uint8_t *src = VC_KWS_INPUT_I8[sample_idx];
  for (uint32_t i = 0; i < VC_KWS_INPUT_COMPACT_BYTES; ++i) {
    input[i * 4u + 0u] = src[i];
    input[i * 4u + 1u] = 0u;
    input[i * 4u + 2u] = 0u;
    input[i * 4u + 3u] = 0u;
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

static uint32_t compare_output(uint32_t sample_idx, const uint8_t *out,
                               uint32_t *max_abs_error) {
  uint32_t mismatches = 0;
  uint32_t max_error = 0;
  for (uint32_t i = 0; i < VC_KWS_OUTPUT_C; ++i) {
    const uint8_t got = out[i];
    const uint8_t exp = VC_KWS_EXPECTED_OUTPUT_I8[sample_idx][i];
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

static void dump_output_mismatches(uint32_t sample_idx, const uint8_t *out) {
  uint32_t printed = 0;
  for (uint32_t i = 0; i < VC_KWS_OUTPUT_C && printed < 8u; ++i) {
    const uint8_t got = out[i];
    const uint8_t exp = VC_KWS_EXPECTED_OUTPUT_I8[sample_idx][i];
    if (got != exp) {
      printf("sample[%lu] mismatch[%lu]: got=%d exp=%d raw=0x%02X/0x%02X\r\n",
             (unsigned long)sample_idx, (unsigned long)i, (int)(int8_t)got,
             (int)(int8_t)exp, (unsigned)got, (unsigned)exp);
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

static void publish_board_result(uint32_t code, uint32_t hw_status,
                                 const kws_run_stats_t *stats) {
  venus_debug_snapshot_t dbg;
  venus_read_debug_snapshot(&dbg);

  g_result->magic = 0u;
  g_result->code = code;
  g_result->hw_status = hw_status;
  g_result->sample_count = stats->sample_count;
  g_result->label_correct = stats->label_correct;
  g_result->ref_top1_match = stats->ref_top1_match;
  g_result->max_abs_error = stats->max_abs_error;
  g_result->total_mismatches = stats->total_mismatches;
  g_result->first_failure_sample = stats->first_failure_sample;
  g_result->first_failure_top1 = stats->first_failure_top1;
  g_result->first_failure_expected_top1 = stats->first_failure_expected_top1;
  g_result->first_failure_label = stats->first_failure_label;
  g_result->status = dbg.status;
  g_result->debug0 = dbg.debug0;
  g_result->debug1 = dbg.debug1;
  g_result->total_cycles = stats->total_cycles;
  venus_cache_flush((const void *)g_result, sizeof(*g_result));

  g_result->magic = TINYML_NPU_RESULT_MAGIC;
  venus_cache_flush((const void *)g_result, sizeof(g_result->magic));
}

static void print_accuracy(const char *name, uint32_t correct, uint32_t total) {
  const uint32_t pct_x100 = (total == 0u) ? 0u : (uint32_t)(((uint64_t)correct * 10000u) / total);
  printf("%s=%lu/%lu (%lu.%02lu%%)\r\n", name, (unsigned long)correct,
         (unsigned long)total, (unsigned long)(pct_x100 / 100u),
         (unsigned long)(pct_x100 % 100u));
}

int main(void) {
  kws_run_stats_t stats;
  init_stats(&stats);

  clear_board_result();
  printf("\r\nTinyML_NPU ZYBO7010 KWS multivector demo\r\n");
  printf("NPU version: 0x%08lX\r\n", (unsigned long)mmio_read32(VENUS_REG_VERSION));
  printf("uOP count  : %lu\r\n", (unsigned long)(UOPS_LEN_WORDS / VENUS_UOP_WORDS));
  printf("shared BRAM: base=0x%08lX used=%lu bytes\r\n",
         (unsigned long)SHARED_MEM_BASE, (unsigned long)SHARED_USED_BYTES);
  printf("samples    : %lu classes=%lu min_samples=%lu\r\n",
         (unsigned long)VC_KWS_SAMPLE_COUNT, (unsigned long)VC_KWS_CLASS_COUNT,
         (unsigned long)VC_KWS_MIN_SAMPLE_COUNT);
  printf("reference  : label_correct=%lu/%lu ref_top1_match=%lu/%lu\r\n",
         (unsigned long)VC_KWS_EXPECTED_LABEL_CORRECT,
         (unsigned long)VC_KWS_SAMPLE_COUNT,
         (unsigned long)VC_KWS_EXPECTED_REF_TOP1_MATCH,
         (unsigned long)VC_KWS_SAMPLE_COUNT);

  memset(g_uops_buf, 0, SHARED_USED_BYTES);
  venus_cache_flush(g_uops_buf, SHARED_USED_BYTES);

  venus_bundle_t bundle = VENUS_BUNDLE_FROM_BUNDLE_H();
  venus_mem_t mem;
  venus_mem_init(&mem, g_uops_buf, SHARED_MEM_BASE + UOPS_OFFSET, UOPS_LEN_BYTES,
                 g_params_buf, SHARED_MEM_BASE + PARAMS_OFFSET, PARAMS_LEN_BYTES,
                 g_act_buf, SHARED_MEM_BASE + ACT_OFFSET, ACTIVATION_PEAK_BYTES);
  venus_mem_set_npu_visible_range(&mem, SHARED_MEM_BASE, SHARED_MEM_HIGH);

  venus_init();

  uint32_t hw_status = 0;
  for (uint32_t sample = 0; sample < VC_KWS_SAMPLE_COUNT; ++sample) {
    memset(g_act_buf, 0, ACTIVATION_PEAK_BYTES);
    venus_cache_flush(g_act_buf, ACTIVATION_PEAK_BYTES);
    load_kws_input(sample);

    venus_status_t st = venus_run_bundle(&bundle, &mem, VENUS_TIMEOUT_CYCLES, &hw_status);
    if (st != VENUS_OK) {
      printf("sample[%lu] venus_run_bundle failed: %s (%d)\r\n",
             (unsigned long)sample, venus_strerror(st), (int)st);
      remember_first_failure(&stats, sample, UINT32_MAX,
                             (uint32_t)VC_KWS_EXPECTED_TOP1[sample],
                             (uint32_t)VC_KWS_LABELS[sample]);
      dump_failure_debug(hw_status);
      publish_board_result(TINYML_NPU_RESULT_RUNTIME_FAILURE, hw_status, &stats);
      printf("TEST FAILED\r\n");
      return 1;
    }

    uint8_t *out = g_act_buf + OUTPUT_BASE;
    venus_cache_invalidate(out, OUTPUT_SIZE);

    const uint32_t top1 = (uint32_t)argmax_i8(out, VC_KWS_OUTPUT_C);
    const uint32_t expected_top1 = (uint32_t)VC_KWS_EXPECTED_TOP1[sample];
    const uint32_t label = (uint32_t)VC_KWS_LABELS[sample];
    uint32_t max_abs_error = 0;
    const uint32_t mismatches = compare_output(sample, out, &max_abs_error);

    stats.sample_count++;
    stats.total_mismatches += mismatches;
    if (max_abs_error > stats.max_abs_error) {
      stats.max_abs_error = max_abs_error;
    }
    if (top1 == label) {
      stats.label_correct++;
    }
    if (top1 == expected_top1) {
      stats.ref_top1_match++;
    }

    venus_debug_snapshot_t dbg;
    venus_read_debug_snapshot(&dbg);
    stats.total_cycles += dbg.debug0;

    if (top1 != expected_top1 || max_abs_error > TINYML_NPU_MAX_ABS_ERROR) {
      remember_first_failure(&stats, sample, top1, expected_top1, label);
      printf("sample[%03lu] label=%lu ref=%lu top1=%lu max_abs_error=%lu mismatches=%lu FAIL\r\n",
             (unsigned long)sample, (unsigned long)label,
             (unsigned long)expected_top1, (unsigned long)top1,
             (unsigned long)max_abs_error, (unsigned long)mismatches);
      dump_output_mismatches(sample, out);
    } else if ((sample % 10u) == 0u || sample == VC_KWS_SAMPLE_COUNT - 1u) {
      printf("sample[%03lu] label=%lu ref=%lu top1=%lu max_abs_error=%lu cycles=%lu\r\n",
             (unsigned long)sample, (unsigned long)label,
             (unsigned long)expected_top1, (unsigned long)top1,
             (unsigned long)max_abs_error, (unsigned long)dbg.debug0);
    }
  }

  print_accuracy("label_accuracy", stats.label_correct, stats.sample_count);
  print_accuracy("ref_top1_match", stats.ref_top1_match, stats.sample_count);
  printf("output max_abs_error=%lu tolerance=%lu total_mismatches=%lu\r\n",
         (unsigned long)stats.max_abs_error,
         (unsigned long)TINYML_NPU_MAX_ABS_ERROR,
         (unsigned long)stats.total_mismatches);
  printf("npu_total_cycles=%lu active_ms_x1000_at_50MHz=%lu\r\n",
         (unsigned long)stats.total_cycles,
         (unsigned long)(((uint64_t)stats.total_cycles * 1000u) / 50000u));

  if (stats.sample_count >= VC_KWS_MIN_SAMPLE_COUNT &&
      stats.sample_count == VC_KWS_SAMPLE_COUNT &&
      stats.ref_top1_match == VC_KWS_EXPECTED_REF_TOP1_MATCH &&
      stats.label_correct >= VC_KWS_EXPECTED_LABEL_CORRECT &&
      stats.max_abs_error <= TINYML_NPU_MAX_ABS_ERROR) {
    publish_board_result(TINYML_NPU_RESULT_PASS, hw_status, &stats);
    printf("TEST PASSED\r\n");
    return 0;
  }

  dump_failure_debug(hw_status);
  publish_board_result(TINYML_NPU_RESULT_OUTPUT_FAILURE, hw_status, &stats);
  printf("TEST FAILED\r\n");
  return 1;
}
