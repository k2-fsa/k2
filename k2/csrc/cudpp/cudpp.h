// this file is copied/modified from
// https://github.com/cudpp/cudpp/blob/master/include/cudpp.h
#ifndef K2_CSRC_CUDPP_CUDPP_H_
#define K2_CSRC_CUDPP_CUDPP_H_

#include <stdint.h>

#include "cub/cub.cuh"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/cudpp/cuda_util.h"
#include "k2/csrc/cudpp/multisplit_kernel.cuh"

typedef unsigned int (*BucketMappingFunc)(unsigned int);

enum CUDPPOption {
  CUDPP_OPTION_KEYS_ONLY = 0x20,       /**< No associated value to a key
                                        * (for global radix sort) */
  CUDPP_OPTION_KEY_VALUE_PAIRS = 0x40, /**< Each key has an associated value */
};

enum CUDPPBucketMapper {
  CUDPP_CUSTOM_BUCKET_MAPPER,  //!< The bucket mapping is a user-specified
                               //!< function.
};

/**
 * @brief Configuration struct used to specify algorithm, datatype,
 * operator, and options when creating a plan for CUDPP algorithms.
 *
 * @see cudppPlan
 */
struct CUDPPConfiguration {
  unsigned int options;  //!< Options to configure the algorithm
  CUDPPBucketMapper bucket_mapper;
  k2::ContextPtr context;
};

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

#define MULTISPLIT_SWITCH_STRATEGY_K 8   // among options 1,2,4,8,16,32
#define MULTISPLIT_SWITCH_STRATEGY_KV 8  // among options 1,2,4,8,16,32
#define MULTISPLIT_NUM_WARPS 8
#define MULTISPLIT_LOG_WARPS 3
#define MULTISPLIT_WARP_WIDTH 32
#define MULTISPLIT_TRHEADS_PER_BLOCK \
  (MULTISPLIT_WARP_WIDTH * MULTISPLIT_NUM_WARPS)

template <typename Lambda>
class CustomBucketMapper {
 public:
  CustomBucketMapper(Lambda &bucketMappingFunc)
      : bucketMapper(bucketMappingFunc) {}

  __device__ unsigned int operator()(unsigned int element) {
    return bucketMapper(element);
  }

 private:
  Lambda bucketMapper;
};

struct multisplit_context {
  k2::Array1<int8_t> d_temp_storage;
  k2::Array1<uint32_t> d_histogram;
  k2::ContextPtr context;
  size_t temp_storage_bytes = 0;
};

class CUDPPMultiSplitPlan {
 public:
  CUDPPMultiSplitPlan(CUDPPConfiguration config, size_t numElements,
                      size_t numBuckets);
  ~CUDPPMultiSplitPlan();
  CUDPPConfiguration m_config;  //!< @internal Options structure

  unsigned int m_numElements;
  unsigned int m_numBuckets;
  k2::Array1<uint32_t> m_d_mask;
  k2::Array1<uint32_t> m_d_out;
  k2::Array1<uint32_t> m_d_fin;
  unsigned int *m_d_temp_keys;
  unsigned int *m_d_temp_values;
  // unsigned long long int *m_d_key_value_pairs;
  uint64_t *m_d_key_value_pairs;
};

void allocMultiSplitStorage(CUDPPMultiSplitPlan *plan);

void freeMultiSplitStorage(CUDPPMultiSplitPlan *plan);

void multisplit_allocate_key_only(size_t num_elements, uint32_t num_buckets,
                                  multisplit_context &context);

void multisplit_release_memory(multisplit_context &context);

template <uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B,
          uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_BMS_prescan(key_type *input, uint32_t *bin,
                                       uint32_t numElements,
                                       bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t>
__global__ void split_post_scan_compaction(key_t *key_input,
                                           uint32_t *warpOffsets,
                                           key_t *key_output,
                                           uint32_t numElements,
                                           bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B,
          uint32_t DEPTH, typename bucket_t, typename key_type,
          typename value_type>
__global__ void multisplit_BMS_pairs_postscan(
    key_type *key_input, value_type *value_input, uint32_t *blockOffsets,
    key_type *key_output, value_type *value_output, uint32_t numElements,
    bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B,
          uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_BMS_postscan(key_type *key_input,
                                        uint32_t *blockOffsets,
                                        key_type *key_output,
                                        uint32_t numElements,
                                        bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH,
          typename bucket_t, typename key_type, typename value_type>
__global__ void multisplit_WMS_pairs_postscan(
    key_type *key_input, value_type *value_input, uint32_t *warpOffsets,
    key_type *key_output, value_type *value_output, uint32_t numElements,
    bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t,
          typename value_t>
__global__ void split_post_scan_pairs_compaction(
    key_t *key_input, value_t *value_input, uint32_t *warpOffsets,
    key_t *key_output, value_t *value_output, uint32_t numElements,
    bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH,
          typename bucket_t, typename key_type>
__global__ void multisplit_WMS_postscan(key_type *key_input,
                                        uint32_t *warpOffsets,
                                        key_type *key_output,
                                        uint32_t numElements,
                                        bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH,
          typename bucket_t, typename key_type>
__global__ void multisplit_WMS_prescan(key_type *input, uint32_t *bin,
                                       uint32_t numElements,
                                       bucket_t bucket_identifier);

template <uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t>
__global__ void histogram_pre_scan_compaction(key_t *input, uint32_t *bin,
                                              uint32_t numElements,
                                              bucket_t bucket_identifier);

__global__ void packingKeyValuePairs(uint64_t *packed, uint32_t *input_key,
                                     uint32_t *input_value,
                                     uint32_t numElements);

__global__ void unpackingKeyValuePairs(uint64_t *packed, uint32_t *out_key,
                                       uint32_t *out_value,
                                       uint32_t numElements);

template <class T>
__global__ void markBins_general(uint32_t *d_mark, uint32_t *d_elements,
                                 uint32_t numElements, uint32_t numBuckets,
                                 T bucketMapper) {
  unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int offset = blockDim.x * gridDim.x;
  unsigned int logBuckets = ceil(log2((float)numBuckets));

  for (int i = myId; i < numElements; i += offset) {
    unsigned int myVal = d_elements[i];
    unsigned int myBucket = bucketMapper(myVal);
    d_mark[i] = myBucket;
  }
}

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
template <class T>
void reducedBitSortKeysOnly(unsigned int *d_inp, uint numElements,
                            uint numBuckets, T bucketMapper,
                            CUDPPMultiSplitPlan *plan) {
  unsigned int numThreads = MULTISPLIT_NUM_WARPS * 32;
  unsigned int numBlocks = (numElements + numThreads - 1) / numThreads;
  unsigned int logBuckets = ceil(log2((float)numBuckets));
  size_t temp_storage_bytes = 0;

  if (numBuckets == 1) return;

  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, plan->m_d_mask.Data(), plan->m_d_out.Data(),
      d_inp, plan->m_d_fin.Data(), numElements, 0, logBuckets);
  k2::Array1<int8_t> d_temp_storage(plan->m_config.context, temp_storage_bytes);

  markBins_general<<<numBlocks, numThreads>>>(
      plan->m_d_mask.Data(), d_inp, numElements, numBuckets, bucketMapper);
  cub::DeviceRadixSort::SortPairs(d_temp_storage.Data(), temp_storage_bytes,
                                  plan->m_d_mask.Data(), plan->m_d_out.Data(),
                                  d_inp, plan->m_d_fin.Data(), numElements, 0,
                                  int(ceil(log2(float(numBuckets)))));

  CUDA_SAFE_CALL(cudaMemcpy(d_inp, plan->m_d_fin.Data(),
                            numElements * sizeof(unsigned int),
                            cudaMemcpyDeviceToDevice));
}

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
template <typename key_type, typename bucket_t>
void multisplit_key_only(key_type *d_key_in, key_type *d_key_out,
                         size_t num_elements, uint32_t num_buckets,
                         multisplit_context &context,
                         bucket_t bucket_identifier, bool in_place,
                         uint32_t *bucket_offsets = NULL) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1) /
                            MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre;
  uint32_t &num_blocks_post = num_blocks_pre;
  uint32_t num_sub_problems;

  if (num_buckets == 1) return;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_K)  // Warp-level MS
  {
    multisplit_WMS_prescan_function(d_key_in, num_elements, bucket_identifier,
                                    num_buckets, num_blocks_raw, num_blocks_pre,
                                    num_sub_problems, context);

    // ============ Scan stage:
    cub::DeviceScan::ExclusiveSum(
        context.d_temp_storage.Data(), context.temp_storage_bytes,
        context.d_histogram.Data(), context.d_histogram.Data(),
        num_buckets * num_sub_problems);

    // ============ Post scan stage:
    multisplit_WMS_postscan_function(d_key_in, d_key_out, num_elements,
                                     bucket_identifier, num_buckets,
                                     num_blocks_post, context);
  } else if (num_buckets <= 32)  // Block-level MS
  {
    // ===== Prescan stage:
    multisplit_BMS_prescan_function(d_key_in, num_elements, bucket_identifier,
                                    num_buckets, num_blocks_raw, num_blocks_pre,
                                    num_sub_problems, context);

    // ===== Scan stage
    cub::DeviceScan::ExclusiveSum(
        context.d_temp_storage.Data(), context.temp_storage_bytes,
        context.d_histogram.Data(), context.d_histogram.Data(),
        num_buckets * num_sub_problems);

    // ===== Postscan stage
    multisplit_BMS_postscan_function(d_key_in, d_key_out, num_elements,
                                     bucket_identifier, num_buckets,
                                     num_blocks_post, context);
  }

  if (in_place) {
    cudaMemcpy(d_key_in, d_key_out, sizeof(key_type) * num_elements,
               cudaMemcpyDeviceToDevice);
  }

  // collecting the bucket offset indices
  if (bucket_offsets != NULL && num_buckets <= 32) {
    bucket_offsets[0] = 0;
    for (uint32_t i = 1; i < num_buckets; i++) {
      cudaMemcpy(&bucket_offsets[i],
                 context.d_histogram.Data() + i * num_sub_problems,
                 sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
  }
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
template <class T>
void reducedBitSortKeyValue(unsigned int *d_keys, unsigned int *d_values,
                            unsigned int numElements, unsigned int numBuckets,
                            T bucketMapper, CUDPPMultiSplitPlan *plan) {
  unsigned int numThreads = MULTISPLIT_NUM_WARPS * 32;
  unsigned int numBlocks = (numElements + numThreads - 1) / numThreads;
  unsigned int logBuckets = ceil(log2((float)numBuckets));
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, plan->m_d_mask.Data(), plan->m_d_out.Data(),
      plan->m_d_key_value_pairs, plan->m_d_key_value_pairs, numElements, 0,
      int(ceil(log2(float(numBuckets)))));

  k2::Array1<int8_t> d_temp_storage(plan->m_config.context, temp_storage_bytes);

  markBins_general<<<numBlocks, numThreads>>>(
      plan->m_d_mask.Data(), d_keys, numElements, numBuckets, bucketMapper);
  packingKeyValuePairs<<<numBlocks, numThreads>>>(
      plan->m_d_key_value_pairs, d_keys, d_values, numElements);
  cub::DeviceRadixSort::SortPairs(d_temp_storage.Data(), temp_storage_bytes,
                                  plan->m_d_mask.Data(), plan->m_d_out.Data(),
                                  plan->m_d_key_value_pairs,
                                  plan->m_d_key_value_pairs, numElements, 0,
                                  int(ceil(log2(float(numBuckets)))));
  unpackingKeyValuePairs<<<numBlocks, numThreads>>>(
      plan->m_d_key_value_pairs, d_keys, d_values, numElements);
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
template <typename key_type, typename value_type, typename bucket_t>
void multisplit_key_value(key_type *d_key_in, value_type *d_value_in,
                          key_type *d_key_out, value_type *d_value_out,
                          size_t num_elements, uint32_t num_buckets,
                          multisplit_context &context,
                          bucket_t bucket_identifier, bool in_place,
                          uint32_t *bucket_offsets = NULL) {
  uint32_t num_blocks_raw = (num_elements + MULTISPLIT_TRHEADS_PER_BLOCK - 1) /
                            MULTISPLIT_TRHEADS_PER_BLOCK;
  uint32_t num_blocks_pre;
  uint32_t &num_blocks_post = num_blocks_pre;
  uint32_t num_sub_problems;

  if (num_buckets == 1) return;

  if (num_buckets <= MULTISPLIT_SWITCH_STRATEGY_KV)  // Warp-level MS
  {
    multisplit_WMS_pairs_prescan_function(
        d_key_in, num_elements, bucket_identifier, num_buckets, num_blocks_raw,
        num_blocks_pre, num_sub_problems, context);

    // ============ Scan stage:
    cub::DeviceScan::ExclusiveSum(
        context.d_temp_storage.Data(), context.temp_storage_bytes,
        context.d_histogram.Data(), context.d_histogram.Data(),
        num_buckets * num_sub_problems);

    // ============ Post scan stage:
    multisplit_WMS_pairs_postscan_function(
        d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
        bucket_identifier, num_buckets, num_blocks_post, context);
  } else if (num_buckets <= 32)  // Block-level MS
  {
    // ===== Prescan stage:
    multisplit_BMS_pairs_prescan_function(
        d_key_in, num_elements, bucket_identifier, num_buckets, num_blocks_raw,
        num_blocks_pre, num_sub_problems, context);

    // ===== Scan stage
    cub::DeviceScan::ExclusiveSum(
        context.d_temp_storage.Data(), context.temp_storage_bytes,
        context.d_histogram.Data(), context.d_histogram.Data(),
        num_buckets * num_sub_problems);

    // ===== Postscan stage
    multisplit_BMS_pairs_postscan_function(
        d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
        bucket_identifier, num_buckets, num_blocks_post, context);
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
      cudaMemcpy(&bucket_offsets[i],
                 context.d_histogram.Data() + i * num_sub_problems,
                 sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
  }
}

//=========================================================================
// Intermediate wrappers:
//=========================================================================
template <typename bucket_t, typename key_type>
void multisplit_WMS_prescan_function(
    key_type *d_key_in, uint32_t num_elements, bucket_t bucket_identifier,
    uint32_t num_buckets, uint32_t num_blocks_raw, uint32_t &num_blocks_pre,
    uint32_t &num_sub_problems, multisplit_context &context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_ONE_ROLL - 1) /
                     MULTISPLIT_WMS_K_ONE_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_ONE_ROLL;
#if MULTISPLIT_SWITCH_STRATEGY_K > 1
    histogram_pre_scan_compaction<MULTISPLIT_NUM_WARPS,
                                  MULTISPLIT_WMS_K_ONE_ROLL>
        <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, context.d_histogram.Data(), num_elements,
            bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_TWO_ROLL - 1) /
                     MULTISPLIT_WMS_K_TWO_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 2
      case 3:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 3, 2,
                               MULTISPLIT_WMS_K_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 4:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 4, 2,
                               MULTISPLIT_WMS_K_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_THREE_ROLL - 1) /
                     MULTISPLIT_WMS_K_THREE_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 4
      case 5:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 5, 3,
                               MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 6:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 6, 3,
                               MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 7:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 7, 3,
                               MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 8:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 8, 3,
                               MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FOUR_ROLL - 1) /
                     MULTISPLIT_WMS_K_FOUR_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 8
      case 9:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 9, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 10:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 10, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 11:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 11, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 12:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 12, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 13:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 13, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 14:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 14, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 15:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 15, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 16:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 16, 4,
                               MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_K_FIVE_ROLL - 1) /
                     MULTISPLIT_WMS_K_FIVE_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_K_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 16
      case 17:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 17, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 18:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 18, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 19:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 19, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 20:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 20, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 21:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 21, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 22:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 22, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 23:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 23, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 24:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 24, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 25:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 25, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 26:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 26, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 27:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 27, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 28:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 28, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 29:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 29, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 30:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 30, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 31:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 31, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 32:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 32, 5,
                               MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}

template <typename bucket_t, typename key_type>
void multisplit_WMS_pairs_prescan_function(
    key_type *d_key_in, uint32_t num_elements, bucket_t bucket_identifier,
    uint32_t num_buckets, uint32_t num_blocks_raw, uint32_t &num_blocks_pre,
    uint32_t &num_sub_problems, multisplit_context &context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_ONE_ROLL - 1) /
                     MULTISPLIT_WMS_KV_ONE_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_ONE_ROLL;
#if MULTISPLIT_SWITCH_STRATEGY_KV > 1
    histogram_pre_scan_compaction<MULTISPLIT_NUM_WARPS,
                                  MULTISPLIT_WMS_KV_ONE_ROLL>
        <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, context.d_histogram.Data(), num_elements,
            bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_TWO_ROLL - 1) /
                     MULTISPLIT_WMS_KV_TWO_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 2
      case 3:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 3, 2,
                               MULTISPLIT_WMS_KV_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 4:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 4, 2,
                               MULTISPLIT_WMS_KV_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_THREE_ROLL - 1) /
                     MULTISPLIT_WMS_KV_THREE_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 4
      case 5:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 5, 3,
                               MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 6:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 6, 3,
                               MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 7:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 7, 3,
                               MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 8:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 8, 3,
                               MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FOUR_ROLL - 1) /
                     MULTISPLIT_WMS_KV_FOUR_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 8
      case 9:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 9, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 10:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 10, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 11:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 11, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 12:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 12, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 13:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 13, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 14:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 14, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 15:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 15, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 16:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 16, 4,
                               MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_WMS_KV_FIVE_ROLL - 1) /
                     MULTISPLIT_WMS_KV_FIVE_ROLL;
    num_sub_problems =
        num_blocks_pre * MULTISPLIT_NUM_WARPS * MULTISPLIT_WMS_KV_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 16
      case 17:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 17, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 18:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 18, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 19:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 19, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 20:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 20, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 21:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 21, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 22:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 22, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 23:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 23, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 24:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 24, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 25:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 25, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 26:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 26, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 27:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 27, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 28:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 28, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 29:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 29, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 30:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 30, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 31:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 31, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 32:
        multisplit_WMS_prescan<MULTISPLIT_NUM_WARPS, 32, 5,
                               MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}

template <typename bucket_t, typename key_type>
void multisplit_WMS_postscan_function(key_type *d_key_in, key_type *d_key_out,
                                      uint32_t num_elements,
                                      bucket_t bucket_identifier,
                                      uint32_t num_buckets,
                                      uint32_t num_blocks_post,
                                      multisplit_context &context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 1
    split_post_scan_compaction<MULTISPLIT_NUM_WARPS, MULTISPLIT_WMS_K_ONE_ROLL>
        <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
            bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 2
      case 3:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 3, 2,
                                MULTISPLIT_WMS_K_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 4:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 4, 2,
                                MULTISPLIT_WMS_K_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
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
                                MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 6:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 6, 3,
                                MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 7:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 7, 3,
                                MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 8:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 8, 3,
                                MULTISPLIT_WMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 8
      case 9:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 9, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 10:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 10, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 11:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 11, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 12:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 12, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 13:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 13, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 14:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 14, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 15:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 15, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 16:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 16, 4,
                                MULTISPLIT_WMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K > 16
      case 17:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 17, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 18:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 18, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 19:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 19, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 20:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 20, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 21:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 21, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 22:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 22, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 23:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 23, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 24:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 24, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 25:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 25, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 26:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 26, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 27:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 27, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 28:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 28, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 29:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 29, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 30:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 30, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 31:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 31, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 32:
        multisplit_WMS_postscan<MULTISPLIT_NUM_WARPS, 32, 5,
                                MULTISPLIT_WMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}

template <typename bucket_t, typename key_type, typename value_type>
void multisplit_WMS_pairs_postscan_function(
    key_type *d_key_in, value_type *d_value_in, key_type *d_key_out,
    value_type *d_value_out, uint32_t num_elements, bucket_t bucket_identifier,
    uint32_t num_buckets, uint32_t num_blocks_post,
    multisplit_context &context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 1
    split_post_scan_pairs_compaction<MULTISPLIT_NUM_WARPS,
                                     MULTISPLIT_WMS_KV_ONE_ROLL>
        <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
            d_value_out, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 2
      case 3:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 3, 2,
                                      MULTISPLIT_WMS_KV_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 4:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 4, 2,
                                      MULTISPLIT_WMS_KV_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
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
                                      MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 6:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 6, 3,
                                      MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 7:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 7, 3,
                                      MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 8:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 8, 3,
                                      MULTISPLIT_WMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 8
      case 9:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 9, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 10:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 10, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 11:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 11, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 12:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 12, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 13:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 13, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 14:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 14, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 15:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 15, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 16:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 16, 4,
                                      MULTISPLIT_WMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV > 16
      case 17:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 17, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 18:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 18, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 19:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 19, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 20:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 20, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 21:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 21, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 22:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 22, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 23:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 23, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 24:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 24, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 25:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 25, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 26:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 26, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 27:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 27, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 28:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 28, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 29:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 29, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 30:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 30, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 31:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 31, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 32:
        multisplit_WMS_pairs_postscan<MULTISPLIT_NUM_WARPS, 32, 5,
                                      MULTISPLIT_WMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_out, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}

template <typename bucket_t, typename key_type>
void multisplit_BMS_prescan_function(
    key_type *d_key_in, uint32_t num_elements, bucket_t bucket_identifier,
    uint32_t num_buckets, uint32_t num_blocks_raw, uint32_t &num_blocks_pre,
    uint32_t &num_sub_problems, multisplit_context &context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_ONE_ROLL - 1) /
                     MULTISPLIT_BMS_K_ONE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_ONE_ROLL;

#if MULTISPLIT_SWITCH_STRATEGY_K <= 1
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1,
                           MULTISPLIT_BMS_K_TWO_ROLL>
        <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, context.d_histogram.Data(), num_elements,
            bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_TWO_ROLL - 1) /
                     MULTISPLIT_BMS_K_TWO_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 2
      case 3:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3, 2,
                               MULTISPLIT_BMS_K_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 4:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4, 2,
                               MULTISPLIT_BMS_K_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_THREE_ROLL - 1) /
                     MULTISPLIT_BMS_K_THREE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 4
      case 5:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5, 3,
                               MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 6:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6, 3,
                               MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 7:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7, 3,
                               MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 8:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8, 3,
                               MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FOUR_ROLL - 1) /
                     MULTISPLIT_BMS_K_FOUR_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 8
      case 9:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 9, 4,
                               MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 10:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 10,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 11:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 11,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 12:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 12,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 13:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 13,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 14:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 14,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 15:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 15,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 16:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 16,
                               4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_K_FIVE_ROLL - 1) /
                     MULTISPLIT_BMS_K_FIVE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_K_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 16
      case 17:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 17,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 18:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 18,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 19:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 19,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 20:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 20,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 21:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 21,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 22:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 22,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 23:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 23,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 24:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 24,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 25:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 25,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 26:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 26,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 27:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 27,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 28:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 28,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 29:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 29,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 30:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 30,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 31:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 31,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 32:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 32,
                               5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}

template <typename bucket_t, typename key_type>
void multisplit_BMS_pairs_prescan_function(
    key_type *d_key_in, uint32_t num_elements, bucket_t bucket_identifier,
    uint32_t num_buckets, uint32_t num_blocks_raw, uint32_t &num_blocks_pre,
    uint32_t &num_sub_problems, multisplit_context &context) {
  if (num_buckets == 2) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_ONE_ROLL - 1) /
                     MULTISPLIT_BMS_KV_ONE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_ONE_ROLL;
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 1
    multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1,
                           MULTISPLIT_BMS_KV_TWO_ROLL>
        <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, context.d_histogram.Data(), num_elements,
            bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_TWO_ROLL - 1) /
                     MULTISPLIT_BMS_KV_TWO_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_TWO_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 2
      case 3:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3, 2,
                               MULTISPLIT_BMS_KV_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 4:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4, 2,
                               MULTISPLIT_BMS_KV_TWO_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 8) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_THREE_ROLL - 1) /
                     MULTISPLIT_BMS_KV_THREE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_THREE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 4
      case 5:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5, 3,
                               MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 6:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6, 3,
                               MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 7:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7, 3,
                               MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 8:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8, 3,
                               MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FOUR_ROLL - 1) /
                     MULTISPLIT_BMS_KV_FOUR_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FOUR_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 8
      case 9:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 9, 4,
                               MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 10:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 10,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 11:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 11,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 12:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 12,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 13:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 13,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 14:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 14,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 15:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 15,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 16:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 16,
                               4, MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    num_blocks_pre = (num_blocks_raw + MULTISPLIT_BMS_KV_FIVE_ROLL - 1) /
                     MULTISPLIT_BMS_KV_FIVE_ROLL;
    num_sub_problems = num_blocks_pre * MULTISPLIT_BMS_KV_FIVE_ROLL;
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 16
      case 17:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 17,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 18:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 18,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 19:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 19,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 20:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 20,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 21:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 21,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 22:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 22,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 23:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 23,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 24:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 24,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 25:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 25,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 26:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 26,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 27:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 27,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 28:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 28,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 29:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 29,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 30:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 30,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 31:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 31,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
      case 32:
        multisplit_BMS_prescan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 32,
                               5, MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_pre, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}
template <typename bucket_t, typename key_type>
void multisplit_BMS_postscan_function(key_type *d_key_in, key_type *d_key_out,
                                      uint32_t num_elements,
                                      bucket_t bucket_identifier,
                                      uint32_t num_buckets,
                                      uint32_t num_blocks_post,
                                      multisplit_context &context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 1
    multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2, 1,
                            MULTISPLIT_BMS_K_TWO_ROLL>
        <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
            bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 2
      case 3:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 3,
                                2, MULTISPLIT_BMS_K_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 4:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 4,
                                2, MULTISPLIT_BMS_K_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 8) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 4
      case 5:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 5,
                                3, MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 6:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 6,
                                3, MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 7:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 7,
                                3, MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 8:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 8,
                                3, MULTISPLIT_BMS_K_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 8
      case 9:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 9,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 10:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 10,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 11:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 11,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 12:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 12,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 13:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 13,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 14:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 14,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 15:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 15,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 16:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 16,
                                4, MULTISPLIT_BMS_K_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_K <= 16
      case 17:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 17,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 18:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 18,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 19:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 19,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 20:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 20,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 21:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 21,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 22:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 22,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 23:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 23,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 24:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 24,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 25:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 25,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 26:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 26,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 27:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 27,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 28:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 28,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 29:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 29,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 30:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 30,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 31:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 31,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
      case 32:
        multisplit_BMS_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 32,
                                5, MULTISPLIT_BMS_K_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, context.d_histogram.Data(), d_key_out, num_elements,
                bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
}

template <typename bucket_t, typename key_type, typename value_type>
void multisplit_BMS_pairs_postscan_function(
    key_type *d_key_in, value_type *d_value_in, key_type *d_key_out,
    value_type *d_value_out, uint32_t num_elements, bucket_t bucket_identifier,
    uint32_t num_buckets, uint32_t num_blocks_post,
    multisplit_context &context) {
  if (num_buckets == 2) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 1
    multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS, MULTISPLIT_LOG_WARPS, 2,
                                  1, MULTISPLIT_BMS_KV_ONE_ROLL>
        <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
            d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
            d_value_out, num_elements, bucket_identifier);
#endif
  } else if (num_buckets <= 4) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 2
      case 3:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 3, 2,
                                      MULTISPLIT_BMS_KV_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 4:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 4, 2,
                                      MULTISPLIT_BMS_KV_TWO_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 8) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 4
      case 5:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 5, 3,
                                      MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 6:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 6, 3,
                                      MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 7:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 7, 3,
                                      MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 8:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 8, 3,
                                      MULTISPLIT_BMS_KV_THREE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 16) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 8
      case 9:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 9, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 10:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 10, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 11:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 11, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 12:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 12, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 13:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 13, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 14:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 14, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 15:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 15, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 16:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 16, 4,
                                      MULTISPLIT_BMS_KV_FOUR_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  } else if (num_buckets <= 32) {
    switch (num_buckets) {
#if MULTISPLIT_SWITCH_STRATEGY_KV <= 16
      case 17:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 17, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 18:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 18, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 19:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 19, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 20:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 20, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 21:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 21, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 22:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 22, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 23:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 23, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 24:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 24, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 25:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 25, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 26:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 26, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 27:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 27, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 28:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 28, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 29:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 29, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 30:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 30, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 31:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 31, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
      case 32:
        multisplit_BMS_pairs_postscan<MULTISPLIT_NUM_WARPS,
                                      MULTISPLIT_LOG_WARPS, 32, 5,
                                      MULTISPLIT_BMS_KV_FIVE_ROLL>
            <<<num_blocks_post, MULTISPLIT_TRHEADS_PER_BLOCK>>>(
                d_key_in, d_value_in, context.d_histogram.Data(), d_key_out,
                d_value_out, num_elements, bucket_identifier);
        break;
#endif
      default:
        break;
    }
  }
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
 * @param[in,out] values Optional associated values to be split (through keys),
 *can be NULL.
 * @param[in] numElements Number of elements to be split.
 * @param[in] plan Configuration information for multiSplit.
 **/
template <typename Lambda>
void cudppMultiSplitDispatch(unsigned int *d_keys, unsigned int *d_values,
                             size_t numElements, size_t numBuckets,
                             Lambda &bucketMappingFunc,
                             CUDPPMultiSplitPlan *plan) {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  multisplit_context ms_context;
  ms_context.context = plan->m_config.context;
  if (numBuckets <= 32)
    multisplit_allocate_key_only(numElements, numBuckets, ms_context);

  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    switch (plan->m_config.bucket_mapper) {
      case CUDPP_CUSTOM_BUCKET_MAPPER:
        if (numBuckets <= 32)
          multisplit_key_value(
              d_keys, d_values, plan->m_d_temp_keys, plan->m_d_temp_values,
              numElements, numBuckets, ms_context,
              CustomBucketMapper<Lambda>(bucketMappingFunc), true);
        else
          reducedBitSortKeyValue(d_keys, d_values, numElements, numBuckets,
                                 CustomBucketMapper<Lambda>(bucketMappingFunc),
                                 plan);
        break;
    }
  } else {
    switch (plan->m_config.bucket_mapper) {
      case CUDPP_CUSTOM_BUCKET_MAPPER:
        if (numBuckets <= 32)
          multisplit_key_only(
              d_keys, plan->m_d_fin.Data(), numElements, numBuckets, ms_context,
              CustomBucketMapper<Lambda>(bucketMappingFunc), true);
        else
          reducedBitSortKeysOnly(d_keys, numElements, numBuckets,
                                 CustomBucketMapper<Lambda>(bucketMappingFunc),
                                 plan);
        break;
    }
  }

  if (numBuckets <= 32) multisplit_release_memory(ms_context);
}

template <typename Lambda>
void cudppMultiSplitCustomBucketMapper(CUDPPMultiSplitPlan *plan,
                                       unsigned int *d_keys,
                                       unsigned int *d_values,
                                       size_t numElements, size_t numBuckets,
                                       Lambda &bucketMappingFunc) {
  cudppMultiSplitDispatch<Lambda>(d_keys, d_values, numElements, numBuckets,
                                  bucketMappingFunc, plan);
}

#endif  // K2_CSRC_CUDPP_CUDPP_H_
