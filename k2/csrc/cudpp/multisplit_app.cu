// this file is copied/modified from
// https://github.com/cudpp/cudpp/blob/master/src/cudpp/app/multisplit_app.cu
#include "cub/cub.cuh"
#include "k2/csrc/cudpp/cuda_util.h"
#include "k2/csrc/cudpp/cudpp.h"

__global__ void unpackingKeyValuePairs(uint64_t *packed, uint32_t *out_key,
                                       uint32_t *out_value,
                                       uint32_t numElements) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements) return;

  uint64_t myPacked = packed[tid];
  out_value[tid] = static_cast<uint>(myPacked & 0x00000000FFFFFFFF);
  out_key[tid] = static_cast<uint>(myPacked >> 32);
}

__global__ void packingKeyValuePairs(uint64_t *packed, uint32_t *input_key,
                                     uint32_t *input_value,
                                     uint32_t numElements) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements) return;

  uint32_t myKey = input_key[tid];
  uint32_t myValue = input_value[tid];
  // putting the key as the more significant 32 bits.
  uint64_t output =
      (static_cast<uint64_t>(myKey) << 32) + static_cast<uint>(myValue);
  packed[tid] = output;
}

//=========================================================================
// Multisplit API:
//=========================================================================

//----------------------
// Memory allocations:
//----------------------
void multisplit_allocate_key_only(size_t num_elements, uint32_t num_buckets,
                                  multisplit_context &context) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1) /
                            MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre = 0;
  uint32_t num_sub_problems = 0;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_K) {  // using Warp-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_ONE_ROLL - 1) /
                       MULTISPLIT_WMS_K_ONE_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_TWO_ROLL - 1) /
                       MULTISPLIT_WMS_K_TWO_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_THREE_ROLL - 1) /
                       MULTISPLIT_WMS_K_THREE_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FOUR_ROLL - 1) /
                       MULTISPLIT_WMS_K_FOUR_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FIVE_ROLL - 1) /
                       MULTISPLIT_WMS_K_FIVE_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_FIVE_ROLL;
    }
  } else {  // using Block-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_ONE_ROLL - 1) /
                       MULTISPLIT_BMS_K_ONE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_TWO_ROLL - 1) /
                       MULTISPLIT_BMS_K_TWO_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_THREE_ROLL - 1) /
                       MULTISPLIT_BMS_K_THREE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FOUR_ROLL - 1) /
                       MULTISPLIT_BMS_K_FOUR_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FIVE_ROLL - 1) /
                       MULTISPLIT_BMS_K_FIVE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FIVE_ROLL;
    }
  }

  context.d_histogram =
      k2::Array1<uint32_t>(context.context, num_buckets * num_sub_problems);

  cub::DeviceScan::ExclusiveSum(
      nullptr, context.temp_storage_bytes, context.d_histogram.Data(),
      context.d_histogram.Data(), num_buckets * num_sub_problems);

  context.d_temp_storage =
      k2::Array1<int8_t>(context.context, context.temp_storage_bytes);
}
//=============
void multisplit_allocate_key_value(size_t num_elements, uint32_t num_buckets,
                                   multisplit_context &context) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1) /
                            MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre = 0;
  uint32_t num_sub_problems = 0;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_KV) {  // using Warp-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_ONE_ROLL - 1) /
                       MULTISPLIT_WMS_KV_ONE_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_TWO_ROLL - 1) /
                       MULTISPLIT_WMS_KV_TWO_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_THREE_ROLL - 1) /
                       MULTISPLIT_WMS_KV_THREE_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FOUR_ROLL - 1) /
                       MULTISPLIT_WMS_KV_FOUR_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FIVE_ROLL - 1) /
                       MULTISPLIT_WMS_KV_FIVE_ROLL;
      num_sub_problems =
          num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_FIVE_ROLL;
    }
  } else {  // using Block-level MS
    if (num_buckets == 2) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_ONE_ROLL - 1) /
                       MULTISPLIT_BMS_KV_ONE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_ONE_ROLL;
    } else if (num_buckets <= 4) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_TWO_ROLL - 1) /
                       MULTISPLIT_BMS_KV_TWO_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_TWO_ROLL;
    } else if (num_buckets <= 8) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_THREE_ROLL - 1) /
                       MULTISPLIT_BMS_KV_THREE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_THREE_ROLL;
    } else if (num_buckets <= 16) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FOUR_ROLL - 1) /
                       MULTISPLIT_BMS_KV_FOUR_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FOUR_ROLL;
    } else if (num_buckets <= 32) {
      num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FIVE_ROLL - 1) /
                       MULTISPLIT_BMS_KV_FIVE_ROLL;
      num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FIVE_ROLL;
    }
  }

  context.d_histogram =
      k2::Array1<uint32_t>(context.context, num_buckets * num_sub_problems);

  cub::DeviceScan::ExclusiveSum(
      nullptr, context.temp_storage_bytes, context.d_histogram.Data(),
      context.d_histogram.Data(), num_buckets * num_sub_problems);

  context.d_temp_storage =
      k2::Array1<int8_t>(context.context, context.temp_storage_bytes);
}

//----------------------
// Memory releases:
//----------------------
void multisplit_release_memory(multisplit_context &context) {}
template <typename key_type>
struct sample_bucket : public std::unary_function<key_type, uint32_t> {
  __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return (a & 0x01);
  }
};

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
    plan->m_d_key_value_pairs =
        k2::Array1<uint64_t>(plan->m_config.context, plan->m_numElements);
  }

  if (plan->m_numBuckets > 32) {
    plan->m_d_mask = k2::Array1<uint32_t>(plan->m_config.context,
                                          plan->m_numElements + 1, 0);

    plan->m_d_out =
        k2::Array1<uint32_t>(plan->m_config.context, plan->m_numElements, 0);
  }
  plan->m_d_fin =
      k2::Array1<uint32_t>(plan->m_config.context, plan->m_numElements, 0);
}
