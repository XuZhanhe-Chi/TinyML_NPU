// SPDX-License-Identifier: Apache-2.0

#ifndef TINYML_NPU_BOARD_RESULT_H
#define TINYML_NPU_BOARD_RESULT_H

#include <stdint.h>

#define TINYML_NPU_RESULT_ABI_VERSION 2u
#define TINYML_NPU_RESULT_ADDR 0x4001FFC0u
#define TINYML_NPU_RESULT_OFFSET 0x0001FFC0u
#define TINYML_NPU_RESULT_BYTES 64u
#define TINYML_NPU_RESULT_WORDS 16u
#define TINYML_NPU_RESULT_MAGIC 0x544E5055u

#define TINYML_NPU_RESULT_PASS 0u
#define TINYML_NPU_RESULT_RUNTIME_FAILURE 1u
#define TINYML_NPU_RESULT_OUTPUT_FAILURE 2u

#define TINYML_NPU_EXPECTED_VERSION 0x00050000u
#define TINYML_NPU_EXPECTED_SAMPLE_COUNT 120u
#define TINYML_NPU_EXPECTED_REF_TOP1_MATCH 120u
#define TINYML_NPU_MIN_LABEL_CORRECT 117u
#define TINYML_NPU_MAX_ABS_ERROR 5u

typedef struct {
  uint32_t magic;
  uint32_t code;
  uint32_t hw_status;
  uint32_t sample_count;
  uint32_t label_correct;
  uint32_t ref_top1_match;
  uint32_t max_abs_error;
  uint32_t total_mismatches;
  uint32_t first_failure_sample;
  uint32_t first_failure_top1;
  uint32_t first_failure_expected_top1;
  uint32_t first_failure_label;
  uint32_t status;
  uint32_t debug0;
  uint32_t debug1;
  uint32_t total_cycles;
} tinyml_npu_board_result_t;

_Static_assert(sizeof(tinyml_npu_board_result_t) == TINYML_NPU_RESULT_BYTES,
               "board result ABI must remain 64 bytes");

#endif
