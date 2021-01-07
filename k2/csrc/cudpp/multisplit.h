// k2/csrc/cudpp/multisplit.h
/*
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * This file is modified from
 * https://github.com/cudpp/cudpp/blob/master/src/cudpp/app/multisplit_app.cu
 */

#ifndef K2_CSRC_CUDPP_MULTISPLIT_H_
#define K2_CSRC_CUDPP_MULTISPLIT_H_

#include "cub/cub.cuh"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "math.h"

namespace k2 {

/* Pack a pair of (key, value) into a single integer.
 *
 *  packed_key_value = (key << 32) | value
 *
 * @param [in] num_elements   Number of elements in `input_key`, `input_value`
 *                            and `packed`.
 * @param [in] input_key      Pointer to the input key array.
 * @param [in] input_value    Pointer to the input value array.
 * @param [out] packed        Pointer to the output packed array.
 */
__global__ void PackingKeyValuePairs(int32_t num_elements,
                                     const uint32_t *input_key,
                                     const uint32_t *input_value,
                                     uint64_t *packed);

/* Unpack a packed (key, value) into two integers.
 *
 *  key = packed >> 32;
 *  value = packed & 0x00000000ffffffff;
 *
 * @param [in] num_elements   Number of elements in `packed`, `out_key` and
 *                            `out_value`.
 * @param [in] packed         It is the output array of `PackingKeyValuePairs`.
 * @param [out] out_key       Pointer to the output key array.
 * @param [out] out_value     Pointer to the output value array.
 */
__global__ void UnpackingKeyValuePairs(int32_t num_elements,
                                       const uint64_t *packed,
                                       uint32_t *out_key, uint32_t *out_value);

/* Map an element to a bucket using `bucket_mapping_func`.
 *
 *  bucket_of_element_i = bucket_mapping_func(elements[i])
 *
 * @param [in] elements         Pointer to the elements array.
 * @param [in] num_elements     Number of elements in the `elements` and
 *                              `mark` array.
 * @param [in] bucket_mapping_func
 *                              It is a function like object supporting the
 *                              following call operator:
 *                              `uint32_t operator(uint32_t i)`
 *
 * @param [out] mark            Pointer to an array contain bucket information.
 *                              `mark[i]` indicates which bucket `elements[i]`
 *                              belongs to.
 */
template <typename Lambda>
static __global__ void MarkBinsGeneral(const uint32_t *elements,
                                       int32_t num_elements,
                                       Lambda bucket_mapping_func,
                                       uint32_t *mark) {
  int32_t my_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (my_id >= num_elements) return;
  int32_t offset = blockDim.x * gridDim.x;

  for (int32_t i = my_id; i < num_elements; i += offset) {
    uint32_t my_val = elements[i];
    uint32_t my_bucket = bucket_mapping_func(my_val);
    mark[i] = my_bucket;
  }
}

/* MultiSplit using RadixSort.
 *
 * @param [in] context      Context indicating where the input data pointers,
 *                          e.g., `in_keys` and `in_values`, come from.
 * @param [in] num_elements Number of elements in `in_keys`, `in_values`,
 *                          `out_keys` and `out_values`.
 * @param [in] bucket_mapping_func
 *                          It is a function like object support the following
 *                          call operator: `uint32_t operator(uint32_t i)`.
 *                          If it is NULL, we assume it is an identity function.
 *                          That is, `in_keys[i]` also represents the bucket
 *                          that `in_keys[i]` belongs to.
 * @param [in] in_keys      Pointer to the input keys array.
 * @param [in] in_values    Pointer to the input values array.
 * @param [in] out_keys     Pointer to the output keys array. If not NULL,
 *                          it will contain a sorted version of `in_keys`. The
 *                          sort is performed according to the bucket
 *                          information of the input elements and it is sorted
 *                          in ascending order.
 * @param [in] out_values   Pointer to the output values array. If not NULL,
 *                          it will contain a sorted version of `in_values`.
 */
template <typename Lambda>
void MultiSplitKeyValuePairs(ContextPtr context, int32_t num_elements,
                             int32_t num_buckets, Lambda *bucket_mapping_func,
                             const uint32_t *in_keys, const uint32_t *in_values,
                             uint32_t *out_keys, uint32_t *out_values) {
  K2_CHECK_GT(num_elements, 0);
  K2_CHECK_GT(num_buckets, 0);
  int32_t log_buckets = static_cast<int32_t>(ceilf(log2f(num_buckets)));
  cudaStream_t stream = context->GetCudaStream();

  constexpr int32_t kMultiSplitNumWarps = 8;
  int32_t num_threads = kMultiSplitNumWarps * 32;
  int32_t num_blocks = (num_elements + num_threads - 1) / num_threads;

  Array1<uint32_t> d_mask(context, num_elements, 0);
  const uint32_t *mask_data = in_keys;

  if (bucket_mapping_func != nullptr) {
    K2_CUDA_SAFE_CALL(MarkBinsGeneral<<<num_blocks, num_threads, 0, stream>>>(
        in_keys, num_elements, *bucket_mapping_func, d_mask.Data()));
    mask_data = d_mask.Data();
  }

  Array1<uint64_t> d_key_value_pairs(context, num_elements);
  K2_CUDA_SAFE_CALL(
      PackingKeyValuePairs<<<num_blocks, num_threads, 0, stream>>>(
          num_elements, in_keys, in_values, d_key_value_pairs.Data()));

  Array1<uint32_t> d_out(context, num_elements, 0);
  size_t temp_storage_bytes = 0;

  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, mask_data, d_out.Data(),
      d_key_value_pairs.Data(), d_key_value_pairs.Data(), num_elements, 0,
      log_buckets, stream));

  Array1<int8_t> d_temp_storage(context, temp_storage_bytes);

  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      d_temp_storage.Data(), temp_storage_bytes, mask_data, d_out.Data(),
      d_key_value_pairs.Data(), d_key_value_pairs.Data(), num_elements, 0,
      log_buckets, stream));

  K2_CUDA_SAFE_CALL(
      UnpackingKeyValuePairs<<<num_blocks, num_threads, 0, stream>>>(
          num_elements, d_key_value_pairs.Data(), out_keys, out_values));
}

template <typename Lambda>
void MultiSplitKeysOnly(ContextPtr context, int32_t num_elements,
                        int32_t num_buckets, Lambda *bucket_mapping_func,
                        const uint32_t *in_keys, uint32_t *out_keys) {
  K2_CHECK_GT(num_elements, 0);
  K2_CHECK_GT(num_buckets, 0);
  int32_t log_buckets = static_cast<int32_t>(ceilf(log2f(num_buckets)));
  cudaStream_t stream = context->GetCudaStream();

  constexpr int32_t kMultiSplitNumWarps = 8;
  int32_t num_threads = kMultiSplitNumWarps * 32;
  int32_t num_blocks = (num_elements + num_threads - 1) / num_threads;

  Array1<uint32_t> d_mask(context, num_elements, 0);
  const uint32_t *mask_data = in_keys;

  if (bucket_mapping_func != nullptr) {
    K2_CUDA_SAFE_CALL(MarkBinsGeneral<<<num_blocks, num_threads, 0, stream>>>(
        in_keys, num_elements, *bucket_mapping_func, d_mask.Data()));
    mask_data = d_mask.Data();
  }

  Array1<uint32_t> d_out(context, num_elements, 0);
  size_t temp_storage_bytes = 0;

  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, mask_data, d_out.Data(), in_keys, out_keys,
      num_elements, 0, log_buckets, stream));

  Array1<int8_t> d_temp_storage(context, temp_storage_bytes);
  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      d_temp_storage.Data(), temp_storage_bytes, mask_data, d_out.Data(),
      in_keys, out_keys, num_elements, 0, log_buckets, stream));
}

}  // namespace k2

#endif  // K2_CSRC_CUDPP_MULTISPLIT_H_
