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
#include "cudpp_util.h"

__global__ void unpackingKeyValuePairs(uint64_t* packed, uint32_t* out_key,
    uint32_t* out_value, uint32_t numElements) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements)
    return;

  uint64_t myPacked = packed[tid];
  out_value[tid] = static_cast<uint>(myPacked & 0x00000000FFFFFFFF);
  out_key[tid] = static_cast<uint>(myPacked >> 32);
}


__global__ void packingKeyValuePairs(uint64_t* packed, uint32_t* input_key,
    uint32_t* input_value, uint32_t numElements) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements)
    return;

  uint32_t myKey = input_key[tid];
  uint32_t myValue = input_value[tid];
  // putting the key as the more significant 32 bits.
  uint64_t output = (static_cast<uint64_t>(myKey) << 32)
      + static_cast<uint>(myValue);
  packed[tid] = output;
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
// Global
//===============================================
cub::CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory
//===============================================
// Definitions:
//===============================================



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


/** @} */ // end multisplit functions
/** @} */// end cudpp_app
