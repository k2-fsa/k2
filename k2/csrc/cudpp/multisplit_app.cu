// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * multisplit_app.cu
 *
 * @brief CUDPP application-level multisplit routines
 */

/** @addtogroup cudpp_app
 * @{
 */

/** @name MultiSplit Functions
 * @{
 */
#include <cub/cub.cuh>
#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"

#include "multisplit_kernel.cuh"

//=========================================================================
// Defined parameters:
//=========================================================================
#define MULTISPLIT_WMS_K_ONE_ROLL 8
#define MULTISPLIT_WMS_K_TWO_ROLL 8
#define MULTISPLIT_WMS_K_THREE_ROLL 4
#define MULTISPLIT_WMS_K_FOUR_ROLL 4
#define MULTISPLIT_WMS_K_FIVE_ROLL 2

#define MULTISPLIT_WMS_KV_ONE_ROLL 4
#define MULTISPLIT_WMS_KV_TWO_ROLL 4
#define MULTISPLIT_WMS_KV_THREE_ROLL 2
#define MULTISPLIT_WMS_KV_FOUR_ROLL 2
#define MULTISPLIT_WMS_KV_FIVE_ROLL 2

#define MULTISPLIT_BMS_K_ONE_ROLL 8
#define MULTISPLIT_BMS_K_TWO_ROLL 8
#define MULTISPLIT_BMS_K_THREE_ROLL 4
#define MULTISPLIT_BMS_K_FOUR_ROLL 4
#define MULTISPLIT_BMS_K_FIVE_ROLL 4

#define MULTISPLIT_BMS_KV_ONE_ROLL 4
#define MULTISPLIT_BMS_KV_TWO_ROLL 4
#define MULTISPLIT_BMS_KV_THREE_ROLL 2
#define MULTISPLIT_BMS_KV_FOUR_ROLL 2
#define MULTISPLIT_BMS_KV_FIVE_ROLL 2

#define MULTISPLIT_SWITCH_STRATEGY_K 8  // among options 1,2,4,8,16,32
#define MULTISPLIT_SWITCH_STRATEGY_KV 8 // among options 1,2,4,8,16,32
#define MULTISPLIT_NUM_WARPS 8
#define MULTISPLIT_LOG_WARPS 3
#define MULTISPLIT_WARP_WIDTH 32
#define MULTISPLIT_TRHEADS_PER_BLOCK (MULTISPLIT_WARP_WIDTH * MULTISPLIT_NUM_WARPS)

class multisplit_context {
public:
  void *d_temp_storage;
  size_t temp_storage_bytes;
  uint32_t *d_histogram;
  multisplit_context() {
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    d_histogram = NULL;
  }
  ~multisplit_context() {
  }
};

//=========================================================================
// Intermediate wrappers:
//=========================================================================
template<typename bucket_t, typename key_type>
void multisplit_WMS_prescan_function(key_type* d_key_in, uint32_t num_elements,
    bucket_t bucket_identifier, uint32_t num_buckets, uint32_t num_blocks_raw,
    uint32_t& num_blocks_pre, uint32_t& num_sub_problems,
    multisplit_context& context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_ONE_ROLL - 1)
        / MULTISPLIT_WMS_K_ONE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_K_ONE_ROLL;
#if MULTISPLIT_SWITCH_STRATEGY_K > 1
    histogram_pre_scan_compaction<MULTISPLIT_NUM_WARPS,
        MULTISPLIT_WMS_K_ONE_ROLL> <<<num_blocks_pre,
        MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
        num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_TWO_ROLL - 1)
        / MULTISPLIT_WMS_K_TWO_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_K_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 2
    case 3:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 3, 2,
          MULTISPLIT_WMS_K_TWO_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 4:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 4, 2,
          MULTISPLIT_WMS_K_TWO_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_THREE_ROLL - 1)
        / MULTISPLIT_WMS_K_THREE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_K_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 4
    case 5:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 5, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 6:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 6, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 7:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 7, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 8:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 8, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FOUR_ROLL - 1)
        / MULTISPLIT_WMS_K_FOUR_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_K_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 8
    case 9:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 9, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 10:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 10, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 11:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 11, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 12:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 12, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 13:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 13, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 14:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 14, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 15:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 15, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 16:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 16, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FIVE_ROLL - 1)
        / MULTISPLIT_WMS_K_FIVE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_K_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 16
    case 17:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 17, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 18:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 18, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 19:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 19, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 20:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 20, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 21:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 21, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 22:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 22, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 23:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 23, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 24:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 24, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 25:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 25, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 26:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 26, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 27:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 27, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 28:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 28, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 29:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 29, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 30:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 30, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 31:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 31, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 32:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 32, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  }
}

template<typename bucket_t, typename key_type>
void multisplit_WMS_pairs_prescan_function(key_type* d_key_in,
    uint32_t num_elements, bucket_t bucket_identifier, uint32_t num_buckets,
    uint32_t num_blocks_raw, uint32_t& num_blocks_pre,
    uint32_t& num_sub_problems, multisplit_context& context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_ONE_ROLL - 1)
        / MULTISPLIT_WMS_KV_ONE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_KV_ONE_ROLL;
#if MULTISPLIT_SWITCH_STRATEGY_KV > 1
    histogram_pre_scan_compaction<MULTISPLIT_NUM_WARPS,
        MULTISPLIT_WMS_KV_ONE_ROLL> <<<num_blocks_pre,
        MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
        num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_TWO_ROLL - 1)
        / MULTISPLIT_WMS_KV_TWO_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_KV_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 2
    case 3:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 3, 2,
          MULTISPLIT_WMS_KV_TWO_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 4:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 4, 2,
          MULTISPLIT_WMS_KV_TWO_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_THREE_ROLL - 1)
        / MULTISPLIT_WMS_KV_THREE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_KV_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 4
    case 5:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 5, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 6:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 6, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 7:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 7, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 8:
      multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 8, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FOUR_ROLL - 1)
        / MULTISPLIT_WMS_KV_FOUR_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_KV_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 8
    case 9:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 9, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 10:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 10, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 11:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 11, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 12:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 12, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 13:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 13, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 14:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 14, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 15:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 15, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 16:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 16, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FIVE_ROLL - 1)
        / MULTISPLIT_WMS_KV_FIVE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
        * MULTISPLIT_WMS_KV_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 16
    case 17:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 17, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 18:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 18, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 19:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 19, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 20:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 20, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 21:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 21, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 22:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 22, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 23:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 23, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 24:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 24, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 25:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 25, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 26:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 26, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 27:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 27, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 28:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 28, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 29:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 29, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 30:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 30, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 31:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 31, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 32:
    multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 32, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  }
}

template<typename bucket_t, typename key_type>
void multisplit_WMS_postscan_function(key_type* d_key_in, key_type* d_key_out,
    uint32_t num_elements, bucket_t bucket_identifier, uint32_t num_buckets,
    uint32_t num_blocks_post, multisplit_context& context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 1
    split_post_scan_compaction<MULTISPLIT_NUM_WARPS, MULTISPLIT_WMS_K_ONE_ROLL> <<<
        num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in,
        context.d_histogram, d_key_out, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 2
    case 3:
      multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 3, 2,
          MULTISPLIT_WMS_K_TWO_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 4:
      multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 4, 2,
          MULTISPLIT_WMS_K_TWO_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 4
    case 5:
      multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 5, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 6:
      multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 6, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 7:
      multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 7, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 8:
      multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 8, 3,
          MULTISPLIT_WMS_K_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 8
    case 9:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 9, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 10:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 10, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 11:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 11, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 12:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 12, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 13:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 13, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 14:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 14, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 15:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 15, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 16:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 16, 4, MULTISPLIT_WMS_K_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 16
    case 17:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 17, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 18:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 18, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 19:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 19, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 20:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 20, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 21:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 21, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 22:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 22, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 23:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 23, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 24:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 24, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 25:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 25, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 26:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 26, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 27:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 27, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 28:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 28, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 29:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 29, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 30:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 30, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 31:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 31, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 32:
    multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 32, 5, MULTISPLIT_WMS_K_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  }
}

template<typename bucket_t, typename key_type, typename value_type>
void multisplit_WMS_pairs_postscan_function(key_type* d_key_in,
    value_type *d_value_in, key_type* d_key_out, value_type *d_value_out,
    uint32_t num_elements, bucket_t bucket_identifier, uint32_t num_buckets,
    uint32_t num_blocks_post, multisplit_context& context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 1
    split_post_scan_pairs_compaction<MULTISPLIT_NUM_WARPS,
        MULTISPLIT_WMS_KV_ONE_ROLL> <<<num_blocks_post,
        MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
        context.d_histogram, d_key_out, d_value_out, num_elements,
        bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 2
    case 3:
      multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 3, 2,
          MULTISPLIT_WMS_KV_TWO_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 4:
      multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 4, 2,
          MULTISPLIT_WMS_KV_TWO_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 4
    case 5:
      multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 5, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 6:
      multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 6, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 7:
      multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 7, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 8:
      multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 8, 3,
          MULTISPLIT_WMS_KV_THREE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 8
    case 9:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 9, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 10:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 10, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 11:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 11, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 12:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 12, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 13:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 13, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 14:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 14, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 15:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 15, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 16:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 16, 4, MULTISPLIT_WMS_KV_FOUR_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 16
    case 17:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 17, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 18:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 18, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 19:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 19, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 20:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 20, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 21:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 21, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 22:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 22, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 23:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 23, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 24:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 24, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 25:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 25, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 26:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 26, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 27:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 27, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 28:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 28, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 29:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 29, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 30:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 30, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 31:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 31, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 32:
    multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 32, 5, MULTISPLIT_WMS_KV_FIVE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_out, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  }
}

template<typename bucket_t, typename key_type>
void multisplit_BMS_prescan_function(key_type* d_key_in, uint32_t num_elements,
    bucket_t bucket_identifier, uint32_t num_buckets, uint32_t num_blocks_raw,
    uint32_t& num_blocks_pre, uint32_t& num_sub_problems,
    multisplit_context& context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_ONE_ROLL - 1)
        / MULTISPLIT_BMS_K_ONE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_ONE_ROLL;

#if MULTISPLIT_SWITCH_STRATEGY_K <= 1
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1, MULTISPLIT_BMS_K_TWO_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_TWO_ROLL - 1)
        / MULTISPLIT_BMS_K_TWO_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 2
    case 3:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3, 2, MULTISPLIT_BMS_K_TWO_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 4:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4, 2, MULTISPLIT_BMS_K_TWO_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_THREE_ROLL - 1)
        / MULTISPLIT_BMS_K_THREE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 4
    case 5:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 6:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 7:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 8:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FOUR_ROLL - 1)
        / MULTISPLIT_BMS_K_FOUR_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 8
    case 9:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 9, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 10:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 10, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 11:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 11, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 12:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 12, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 13:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 13, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 14:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 14, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 15:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 15, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 16:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 16, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FIVE_ROLL - 1)
        / MULTISPLIT_BMS_K_FIVE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 16
    case 17:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 17, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 18:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 18, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 19:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 19, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 20:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 20, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 21:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 21, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 22:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 22, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 23:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 23, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 24:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 24, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 25:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 25, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 26:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 26, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 27:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 27, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 28:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 28, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 29:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 29, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 30:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 30, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 31:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 31, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 32:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 32, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  }
}

template<typename bucket_t, typename key_type>
void multisplit_BMS_pairs_prescan_function(key_type* d_key_in,
    uint32_t num_elements, bucket_t bucket_identifier, uint32_t num_buckets,
    uint32_t num_blocks_raw, uint32_t& num_blocks_pre,
    uint32_t& num_sub_problems, multisplit_context& context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_ONE_ROLL - 1)
        / MULTISPLIT_BMS_KV_ONE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_ONE_ROLL;
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 1
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1, MULTISPLIT_BMS_KV_TWO_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_TWO_ROLL - 1)
        / MULTISPLIT_BMS_KV_TWO_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 2
    case 3:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3, 2, MULTISPLIT_BMS_KV_TWO_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 4:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4, 2, MULTISPLIT_BMS_KV_TWO_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_THREE_ROLL - 1)
        / MULTISPLIT_BMS_KV_THREE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 4
    case 5:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 6:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 7:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
    case 8:
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FOUR_ROLL - 1)
        / MULTISPLIT_BMS_KV_FOUR_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 8
    case 9:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 9, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 10:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 10, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 11:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 11, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 12:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 12, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 13:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 13, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 14:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 14, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 15:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 15, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 16:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 16, 4,
          MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FIVE_ROLL - 1)
        / MULTISPLIT_BMS_KV_FIVE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 16
    case 17:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 17, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 18:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 18, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 19:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 19, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 20:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 20, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 21:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 21, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 22:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 22, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 23:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 23, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 24:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 24, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 25:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 25, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 26:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 26, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 27:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 27, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 28:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 28, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 29:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 29, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 30:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 30, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 31:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 31, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
    case 32:
      multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 32, 5,
          MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_pre,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  }
}
template<typename bucket_t, typename key_type>
void multisplit_BMS_postscan_function(key_type* d_key_in, key_type* d_key_out,
    uint32_t num_elements, bucket_t bucket_identifier, uint32_t num_buckets,
    uint32_t num_blocks_post, multisplit_context& context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 1
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1, MULTISPLIT_BMS_K_TWO_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 2
    case 3:
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3, 2, MULTISPLIT_BMS_K_TWO_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 4:
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4, 2, MULTISPLIT_BMS_K_TWO_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 4
    case 5:
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 6:
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 7:
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
    case 8:
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8, 3, MULTISPLIT_BMS_K_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram, d_key_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 8
    case 9:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 9, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 10:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 10, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 11:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 11, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 12:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 12, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 13:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 13, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 14:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 14, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 15:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 15, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 16:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 16, 4,
          MULTISPLIT_BMS_K_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 16
    case 17:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 17, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 18:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 18, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 19:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 19, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 20:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 20, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 21:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 21, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 22:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 22, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 23:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 23, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 24:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 24, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 25:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 25, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 26:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 26, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 27:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 27, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 28:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 28, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 29:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 29, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 30:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 30, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 31:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 31, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
    case 32:
      multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 32, 5,
          MULTISPLIT_BMS_K_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, context.d_histogram,
          d_key_out, num_elements, bucket_identifier);
      break;
#endif
    default:
      break;
    }
  }
}

template<typename bucket_t, typename key_type, typename value_type>
void multisplit_BMS_pairs_postscan_function(key_type* d_key_in,
    value_type* d_value_in, key_type* d_key_out, value_type* d_value_out,
    uint32_t num_elements, bucket_t bucket_identifier, uint32_t num_buckets,
    uint32_t num_blocks_post, multisplit_context& context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 1
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1, MULTISPLIT_BMS_KV_ONE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 2
    case 3:
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3, 2, MULTISPLIT_BMS_KV_TWO_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 4:
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4, 2, MULTISPLIT_BMS_KV_TWO_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 8) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 4
    case 5:
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 6:
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 7:
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
    case 8:
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8, 3, MULTISPLIT_BMS_KV_THREE_ROLL><<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in, context.d_histogram, d_key_out, d_value_out, num_elements, bucket_identifier);
    break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 8
    case 9:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          9, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 10:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          10, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 11:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          11, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 12:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          12, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 13:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          13, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 14:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          14, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 15:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          15, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 16:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          16, 4, MULTISPLIT_BMS_KV_FOUR_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
#endif
    default:
      break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 16
    case 17:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          17, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 18:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          18, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 19:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          19, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 20:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          20, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 21:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          21, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 22:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          22, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 23:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          23, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 24:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          24, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 25:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          25, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 26:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          26, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 27:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          27, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 28:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          28, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 29:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          29, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 30:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          30, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 31:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          31, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
    case 32:
      multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS,
          32, 5, MULTISPLIT_BMS_KV_FIVE_ROLL> <<<num_blocks_post,
          MULTISPLIT_TRHEADS_PER_BLOCK>>>(d_key_in, d_value_in,
          context.d_histogram, d_key_out, d_value_out, num_elements,
          bucket_identifier);
      break;
#endif
    default:
      break;
    }
  }
}

//=========================================================================
// Multisplit API:
//=========================================================================

//----------------------
// Memory allocations:
//----------------------
void multisplit_allocate_key_only(size_t num_elements, uint32_t num_buckets,
    multisplit_context& context) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1)
      / MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre = 0;
  uint32_t num_sub_problems = 0;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_K) { // using Warp-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_ONE_ROLL - 1)
          / MULTISPLIT_WMS_K_ONE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_K_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_TWO_ROLL - 1)
          / MULTISPLIT_WMS_K_TWO_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_K_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_THREE_ROLL - 1)
          / MULTISPLIT_WMS_K_THREE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_K_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FOUR_ROLL - 1)
          / MULTISPLIT_WMS_K_FOUR_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_K_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FIVE_ROLL - 1)
          / MULTISPLIT_WMS_K_FIVE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_K_FIVE_ROLL;
    }
  } else { // using Block-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_ONE_ROLL - 1)
          / MULTISPLIT_BMS_K_ONE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_TWO_ROLL - 1)
          / MULTISPLIT_BMS_K_TWO_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_THREE_ROLL - 1)
          / MULTISPLIT_BMS_K_THREE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FOUR_ROLL - 1)
          / MULTISPLIT_BMS_K_FOUR_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FIVE_ROLL - 1)
          / MULTISPLIT_BMS_K_FIVE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FIVE_ROLL;
    }
  }

  cudaMalloc((void**) &context.d_histogram,
      sizeof(uint32_t) * num_buckets * num_sub_problems);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage,
      context.temp_storage_bytes, context.d_histogram, context.d_histogram,
      num_buckets * num_sub_problems);
  cudaMalloc((void**) &context.d_temp_storage, context.temp_storage_bytes);
}
//=============
void multisplit_allocate_key_value(size_t num_elements, uint32_t num_buckets,
    multisplit_context& context) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1)
      / MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre = 0;
  uint32_t num_sub_problems = 0;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_KV) { // using Warp-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_ONE_ROLL - 1)
          / MULTISPLIT_WMS_KV_ONE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_KV_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_TWO_ROLL - 1)
          / MULTISPLIT_WMS_KV_TWO_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_KV_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_THREE_ROLL - 1)
          / MULTISPLIT_WMS_KV_THREE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_KV_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FOUR_ROLL - 1)
          / MULTISPLIT_WMS_KV_FOUR_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_KV_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FIVE_ROLL - 1)
          / MULTISPLIT_WMS_KV_FIVE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_NUM_WARPS
          * MULTISPLIT_WMS_KV_FIVE_ROLL;
    }
  } else { // using Block-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_ONE_ROLL - 1)
          / MULTISPLIT_BMS_KV_ONE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_TWO_ROLL - 1)
          / MULTISPLIT_BMS_KV_TWO_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_THREE_ROLL - 1)
          / MULTISPLIT_BMS_KV_THREE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FOUR_ROLL - 1)
          / MULTISPLIT_BMS_KV_FOUR_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FIVE_ROLL - 1)
          / MULTISPLIT_BMS_KV_FIVE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FIVE_ROLL;
    }
  }

  cudaMalloc((void**) &context.d_histogram,
      sizeof(uint32_t) * num_buckets * num_sub_problems);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage,
      context.temp_storage_bytes, context.d_histogram, context.d_histogram,
      num_buckets * num_sub_problems);
  cudaMalloc((void**) &context.d_temp_storage, context.temp_storage_bytes);
}

//----------------------
// Memory releases:
//----------------------
void multisplit_release_memory(multisplit_context& context) {
  cudaFree(context.d_histogram);
  cudaFree(context.d_temp_storage);
}
template<typename key_type>
struct sample_bucket: public std::unary_function<key_type, uint32_t> {
  __forceinline__
  __device__  __host__ uint32_t operator()(key_type a) const {
    return (a & 0x01);
  }
};

//===============================================
// Definitions:
//===============================================

/** @brief Performs multisplit on keys only.
 *
 *
 * This function performs multisplit on a list of keys for a number of buckets
 * less than or equal to 32. If the number of buckets is less than a threshold,
 * a warp-level multisplit is used. If the number of buckets is greater than
 * the threshold, a block-level multisplit is used. This function also supports
 * copying the results of multisplit back into the input array. In addition,
 * the offset indices marking the locations of the buckets in the result array
 * can optionally be saved.
 *
 * @param[in,out] d_key_in Input keys to be multisplit.
 * @param[out] d_key_out Output keys after multisplit.
 * @param[in] num_elements Number of elements.
 * @param[in] num_buckets Number of buckets.
 * @param[in] context Intermediate data storage for multisplit.
 * @param[in] bucket_identifier Functor to map an element to a bucket number.
 * @param[in] in_place Flag to indicate if results are copied back to the input.
 * @param[out] bucket_offsets Optional output list of bucket indices.
 **/
template<typename key_type, typename bucket_t>
void multisplit_key_only(key_type* d_key_in, key_type* d_key_out,
    size_t num_elements, uint32_t num_buckets, multisplit_context& context,
    bucket_t bucket_identifier, bool in_place, uint32_t* bucket_offsets = NULL) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1)
      / MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre;
  uint32_t& num_blocks_post = num_blocks_pre;
  uint32_t num_sub_problems;

  if (num_buckets == 1)
    return;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_K) // Warp-level MS
  {
    multisplit_WMS_prescan_function(d_key_in, num_elements, bucket_identifier,
        num_buckets, num_blocks_raw, num_blocks_pre, num_sub_problems, context);

    // ============ Scan stage:
    cub::DeviceScan::ExclusiveSum(context.d_temp_storage,
        context.temp_storage_bytes, context.d_histogram, context.d_histogram,
        num_buckets * num_sub_problems);

    // ============ Post scan stage:
    multisplit_WMS_postscan_function(d_key_in, d_key_out, num_elements,
        bucket_identifier, num_buckets, num_blocks_post, context);
  } else if (num_buckets <= 32) // Block-level MS
      {
    // ===== Prescan stage:
    multisplit_BMS_prescan_function(d_key_in, num_elements, bucket_identifier,
        num_buckets, num_blocks_raw, num_blocks_pre, num_sub_problems, context);

    // ===== Scan stage
    cub::DeviceScan::ExclusiveSum(context.d_temp_storage,
        context.temp_storage_bytes, context.d_histogram, context.d_histogram,
        num_buckets * num_sub_problems);

    // ===== Postscan stage
    multisplit_BMS_postscan_function(d_key_in, d_key_out, num_elements,
        bucket_identifier, num_buckets, num_blocks_post, context);
  }

  if (in_place) {
    cudaMemcpy(d_key_in, d_key_out, sizeof(key_type) * num_elements,
        cudaMemcpyDeviceToDevice);
  }

  // collecting the bucket offset indices
  if (bucket_offsets != NULL && num_buckets <= 32) {
    bucket_offsets[0] = 0;
    for (uint32_t i = 1; i < num_buckets; i++) {
      cudaMemcpy(&bucket_offsets[i], context.d_histogram + i * num_sub_problems,
          sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
  }
}

/** @brief Performs multisplit on key-value pairs.
 *
 *
 * This function performs multisplit on a list of keys and a
 * list of values for a number of buckets less than or equal to 32.
 * If the number of buckets is less than a threshold,
 * a warp-level multisplit is used. If the number of buckets is
 * greater than the threshold, a block-level multisplit is used.
 * This function also supports copying the results of multisplit back
 * into the key and value input arrays. In addition, the offset indices
 * marking the locations of the buckets in the result arrays
 * can optionally be saved.
 *
 * @param[in,out] d_key_in Input keys to be multisplit.
 * @param[in,out] d_value_in Input values to be multisplit along with keys.
 * @param[out] d_key_out Output keys after multisplit.
 * @param[out] d_value_out Output keys after multisplit.
 * @param[in] num_elements Number of elements.
 * @param[in] num_buckets Number of buckets.
 * @param[in] context Intermediate data storage for multisplit.
 * @param[in] bucket_identifier Functor to map an element to a bucket number.
 * @param[in] in_place Flag to indicate if results are copied back to inputs.
 * @param[out] bucket_offsets Optional output list of bucket indices.
 **/
template<typename key_type, typename value_type, typename bucket_t>
void multisplit_key_value(key_type* d_key_in, value_type *d_value_in,
    key_type* d_key_out, value_type* d_value_out, size_t num_elements,
    uint32_t num_buckets, multisplit_context& context,
    bucket_t bucket_identifier, bool in_place, uint32_t* bucket_offsets = NULL) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1)
      / MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre;
  uint32_t& num_blocks_post = num_blocks_pre;
  uint32_t num_sub_problems;

  if (num_buckets == 1)
    return;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_KV) // Warp-level MS
  {
    multisplit_WMS_pairs_prescan_function(d_key_in, num_elements,
        bucket_identifier, num_buckets, num_blocks_raw, num_blocks_pre,
        num_sub_problems, context);

    // ============ Scan stage:
    cub::DeviceScan::ExclusiveSum(context.d_temp_storage,
        context.temp_storage_bytes, context.d_histogram, context.d_histogram,
        num_buckets * num_sub_problems);

    // ============ Post scan stage:
    multisplit_WMS_pairs_postscan_function(d_key_in, d_value_in, d_key_out,
        d_value_out, num_elements, bucket_identifier, num_buckets,
        num_blocks_post, context);
  } else if (num_buckets <= 32) // Block-level MS
      {
    // ===== Prescan stage:
    multisplit_BMS_pairs_prescan_function(d_key_in, num_elements,
        bucket_identifier, num_buckets, num_blocks_raw, num_blocks_pre,
        num_sub_problems, context);

    // ===== Scan stage
    cub::DeviceScan::ExclusiveSum(context.d_temp_storage,
        context.temp_storage_bytes, context.d_histogram, context.d_histogram,
        num_buckets * num_sub_problems);

    // ===== Postscan stage
    multisplit_BMS_pairs_postscan_function(d_key_in, d_value_in, d_key_out,
        d_value_out, num_elements, bucket_identifier, num_buckets,
        num_blocks_post, context);
  }

  if (in_place) {
    cudaMemcpy(d_key_in, d_key_out, sizeof(key_type) * num_elements,
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_value_in, d_value_out, sizeof(value_type) * num_elements,
        cudaMemcpyDeviceToDevice);
  }

  // collecting the bucket offset indices
  if (bucket_offsets != NULL && num_buckets <= 32) {
    bucket_offsets[0] = 0;
    for (uint32_t i = 1; i < num_buckets; i++) {
      cudaMemcpy(&bucket_offsets[i], context.d_histogram + i * num_sub_problems,
          sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
  }
}

//===============================================
// Global
//===============================================
cub::CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory
//===============================================
// Definitions:
//===============================================

/** @brief Performs multisplit on keys only using the reduced-bit sort method.
 *
 *
 * This function uses radix sort to perform a multisplit. It is suitable 
 * when the number of buckets is large.
 *
 * @param[in,out] d_inp Keys to be multisplit.
 * @param[in] numElements Number of elements.
 * @param[in] numBuckets Number of buckets.
 * @param[in] bucketMapper Functor that maps an element to a bucket number.
 * @param[in] plan Configuration plan for multisplit.
 **/
template<class T>
void reducedBitSortKeysOnly(unsigned int *d_inp, uint numElements,
    uint numBuckets, T bucketMapper, const CUDPPMultiSplitPlan *plan) {
  unsigned int numThreads = MULTISPLIT_NUM_WARPS * 32;
  unsigned int numBlocks = (numElements + numThreads - 1) / numThreads;
  unsigned int logBuckets = ceil(log2((float) numBuckets));
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  if (numBuckets == 1)
    return;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      plan->m_d_mask, plan->m_d_out, d_inp, plan->m_d_fin, numElements, 0,
      logBuckets);
  g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

  markBins_general<<<numBlocks, numThreads>>>(plan->m_d_mask, d_inp,
      numElements, numBuckets, bucketMapper);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      plan->m_d_mask, plan->m_d_out, d_inp, plan->m_d_fin, numElements, 0,
      int(ceil(log2(float(numBuckets)))));

  CUDA_SAFE_CALL(
      cudaMemcpy(d_inp, plan->m_d_fin, numElements * sizeof(unsigned int),
          cudaMemcpyDeviceToDevice));

  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

/** @brief Performs multisplit on key-value pairs using a reduced-bit sort.
 *
 *
 * This function uses radix sort to perform a multisplit on a list of keys 
 * and a list of values. It is suitable when the number of buckets is large.
 *
 * @param[in,out] d_keys Keys to be multisplit.
 * @param[in,out] d_values Associated values to be multisplit
 * @param[in] numElements Number of key-value pairs.
 * @param[in] numBuckets Number of buckets.
 * @param[in] bucketMapper Functor that maps an element to a bucket number.
 * @param[in] plan Configuration information for multisplit.
 **/
template<class T>
void reducedBitSortKeyValue(unsigned int *d_keys, unsigned int *d_values,
    unsigned int numElements, unsigned int numBuckets, T bucketMapper,
    const CUDPPMultiSplitPlan *plan) {
  unsigned int numThreads = MULTISPLIT_NUM_WARPS * 32;
  unsigned int numBlocks = (numElements + numThreads - 1) / numThreads;
  unsigned int logBuckets = ceil(log2((float) numBuckets));
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      plan->m_d_mask, plan->m_d_out, plan->m_d_key_value_pairs,
      plan->m_d_key_value_pairs, numElements, 0,
      int(ceil(log2(float(numBuckets)))));

  g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

  markBins_general<<<numBlocks, numThreads>>>(plan->m_d_mask, d_keys,
      numElements, numBuckets, bucketMapper);
  packingKeyValuePairs<<<numBlocks, numThreads>>>(plan->m_d_key_value_pairs,
      d_keys, d_values, numElements);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      plan->m_d_mask, plan->m_d_out, plan->m_d_key_value_pairs,
      plan->m_d_key_value_pairs, numElements, 0,
      int(ceil(log2(float(numBuckets)))));
  unpackingKeyValuePairs<<<numBlocks, numThreads>>>(plan->m_d_key_value_pairs,
      d_keys, d_values, numElements);

  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief From the programmer-specified multisplit configuration,
 *        creates internal memory for performing the multisplit.
 *        Different storage amounts are required depending on the
 *        number of buckets.
 *
 * @param[in] plan Pointer to CUDPPMultiSplitPlan object
 **/
void allocMultiSplitStorage(CUDPPMultiSplitPlan *plan) {
  unsigned int nB = ceil(plan->m_numElements / (MULTISPLIT_NUM_WARPS * 32));

  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    CUDA_SAFE_CALL(
        cudaMalloc((void** ) &plan->m_d_key_value_pairs,
            plan->m_numElements * sizeof(uint64))); // key value pair intermediate vector.
  }

  if (plan->m_numBuckets > 32) {
    CUDA_SAFE_CALL(
        cudaMalloc((void** ) &plan->m_d_mask,
            (plan->m_numElements + 1) * sizeof(unsigned int))); // mask verctor, +1 added only for the near-far implementation
    CUDA_SAFE_CALL(
        cudaMalloc((void** ) &plan->m_d_out,
            plan->m_numElements * sizeof(unsigned int))); // gpu output
  }
  CUDA_SAFE_CALL(
      cudaMalloc((void** ) &plan->m_d_fin,
          plan->m_numElements * sizeof(unsigned int))); // final masks (used for reduced bit method, etc.)

  if (plan->m_numBuckets > 32) {
    CUDA_SAFE_CALL(
        cudaMemset(plan->m_d_mask, 0,
            sizeof(unsigned int) * (plan->m_numElements + 1)));
    CUDA_SAFE_CALL(
        cudaMemset(plan->m_d_out, 0,
            sizeof(unsigned int) * plan->m_numElements));
  }
  CUDA_SAFE_CALL(
      cudaMemset(plan->m_d_fin, 0, sizeof(unsigned int) * plan->m_numElements));
}

/** @brief Deallocates intermediate memory from allocMultiSplitStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPMultiSplitPlan object
 **/
void freeMultiSplitStorage(CUDPPMultiSplitPlan* plan) {
  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    cudaFree(plan->m_d_key_value_pairs);
  }
  if (plan->m_numBuckets > 32) {
    cudaFree(plan->m_d_mask);
    cudaFree(plan->m_d_out);
  }
  cudaFree(plan->m_d_fin);
}

/** @brief Dispatch function to perform multisplit on an array of
 * elements into a number of buckets.
 *
 * This is the dispatch routine which calls multiSplit...() with
 * appropriate parameters, including the bucket mapping function
 * specified by plan's configuration.
 *
 * Currently only splits unsigned integers.
 * @param[in,out] keys Keys to be split.
 * @param[in,out] values Optional associated values to be split (through keys), can be NULL.
 * @param[in] numElements Number of elements to be split.
 * @param[in] plan Configuration information for multiSplit.
 **/
void cudppMultiSplitDispatch(unsigned int *d_keys, unsigned int *d_values,
    size_t numElements, size_t numBuckets,
    BucketMappingFunc bucketMappingFunc,
    const CUDPPMultiSplitPlan *plan) {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  multisplit_context ms_context;
  if (numBuckets <= 32)
    multisplit_allocate_key_only(numElements, numBuckets, ms_context);

  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    switch (plan->m_config.bucket_mapper) {
    case CUDPP_CUSTOM_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_value(d_keys, d_values, plan->m_d_temp_keys,
            plan->m_d_temp_values, numElements, numBuckets, ms_context,
            CustomBucketMapper(bucketMappingFunc), true);
      else
        reducedBitSortKeyValue(d_keys, d_values, numElements, numBuckets,
            CustomBucketMapper(bucketMappingFunc), plan);
      break;
    case CUDPP_DEFAULT_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_value(d_keys, d_values, plan->m_d_temp_keys,
            plan->m_d_temp_values, numElements, numBuckets, ms_context,
            OrderedCyclicBucketMapper(numElements, numBuckets), true);
      else
        reducedBitSortKeyValue(d_keys, d_values, numElements, numBuckets,
            OrderedCyclicBucketMapper(numElements, numBuckets), plan);
      break;
    case CUDPP_MSB_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_value(d_keys, d_values, plan->m_d_temp_keys,
            plan->m_d_temp_values, numElements, numBuckets, ms_context,
            MSBBucketMapper(numBuckets), true);
      else
        reducedBitSortKeyValue(d_keys, d_values, numElements, numBuckets,
            MSBBucketMapper(numBuckets), plan);
      break;
    case CUDPP_LSB_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_value(d_keys, d_values, plan->m_d_temp_keys,
            plan->m_d_temp_values, numElements, numBuckets, ms_context,
            LSBBucketMapper(numBuckets), true);
      else
        reducedBitSortKeyValue(d_keys, d_values, numElements, numBuckets,
            LSBBucketMapper(numBuckets), plan);
      break;
    default:
      if (numBuckets <= 32)
        multisplit_key_value(d_keys, d_values, plan->m_d_temp_keys,
            plan->m_d_temp_values, numElements, numBuckets, ms_context,
            OrderedCyclicBucketMapper(numElements, numBuckets), true);
      else
        reducedBitSortKeyValue(d_keys, d_values, numElements, numBuckets,
            OrderedCyclicBucketMapper(numElements, numBuckets), plan);
      break;
    }
  } else {
    switch (plan->m_config.bucket_mapper) {
    case CUDPP_CUSTOM_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_only(d_keys, plan->m_d_fin, numElements, numBuckets,
            ms_context, CustomBucketMapper(bucketMappingFunc), true);
      else
        reducedBitSortKeysOnly(d_keys, numElements, numBuckets,
            CustomBucketMapper(bucketMappingFunc), plan);
      break;
    case CUDPP_DEFAULT_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_only(d_keys, plan->m_d_fin, numElements, numBuckets,
            ms_context, OrderedCyclicBucketMapper(numElements, numBuckets),
            true);
      else
        reducedBitSortKeysOnly(d_keys, numElements, numBuckets,
            OrderedCyclicBucketMapper(numElements, numBuckets), plan);
      break;
    case CUDPP_MSB_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_only(d_keys, plan->m_d_fin, numElements, numBuckets,
            ms_context, MSBBucketMapper(numBuckets), true);
      else
        reducedBitSortKeysOnly(d_keys, numElements, numBuckets,
            MSBBucketMapper(numBuckets), plan);
      break;
    case CUDPP_LSB_BUCKET_MAPPER:
      if (numBuckets <= 32)
        multisplit_key_only(d_keys, plan->m_d_fin, numElements, numBuckets,
            ms_context, LSBBucketMapper(numBuckets), true);
      else
        reducedBitSortKeysOnly(d_keys, numElements, numBuckets,
            LSBBucketMapper(numBuckets), plan);
      break;
    default:
      if (numBuckets <= 32)
        multisplit_key_only(d_keys, plan->m_d_fin, numElements, numBuckets,
            ms_context, OrderedCyclicBucketMapper(numElements, numBuckets),
            true);
      else
        reducedBitSortKeysOnly(d_keys, numElements, numBuckets,
            OrderedCyclicBucketMapper(numElements, numBuckets), plan);
      break;
    }
  }

  if (numBuckets <= 32)
    multisplit_release_memory(ms_context);
}

#ifdef __cplusplus
}

#endif

/** @} */ // end multisplit functions
/** @} */// end cudpp_app
