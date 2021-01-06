

#ifndef K2_CSRC_CUDPP_MULTISPLIT_H_
#define K2_CSRC_CUDPP_MULTISPLIT_H_

#include "cub/cub.cuh"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace k2 {

// TODO(fangjun): add doc
__global__ void PackingKeyValuePairs(int32_t num_elements,
                                     const uint32_t *input_key,
                                     const uint32_t *input_value,
                                     uint64_t *packed);

// TODO(fangjun): add doc
__global__ void UnpackingKeyValuePairs(int32_t num_elements,
                                       const uint64_t *packed,
                                       uint32_t *out_key, uint32_t *out_value);

// accepted lambda: [](uint32_t)->uint32_t
// mapping a key to a bucket
// TODO(fangjun): add doc
template <typename Lambda>
static __global__ void MarkBinsGeneral(const uint32_t *elements,
                                       int32_t num_elements,
                                       int32_t num_buckets,
                                       Lambda bucket_mapping_func,
                                       uint32_t *mark) {
  int32_t my_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (my_id >= num_elements) return;
  int32_t offset = blockDim.x * gridDim.x;
  int32_t log_buckets = ceil(log2((float)num_buckets));

  for (int32_t i = my_id; i < num_elements; i += offset) {
    uint32_t my_val = elements[i];
    uint32_t my_bucket = bucket_mapping_func(my_val);
    mark[i] = my_bucket;
  }
}

// TODO(fangjun): add doc
template <typename Lambda>
void MultiSplit(ContextPtr context, int32_t num_elements, int32_t num_buckets,
                Lambda bucket_mapping_func, const uint32_t *in_keys,
                const uint32_t *in_values, uint32_t *out_keys,
                uint32_t *out_values) {
  K2_CHECK_GT(num_elements, 0);
  K2_CHECK_GT(num_buckets, 0);

  constexpr int32_t kMultiSplitNumWarps = 8;
  int32_t num_threads = kMultiSplitNumWarps * 32;
  int32_t num_blocks = (num_elements + num_threads - 1) / num_threads;

  Array1<uint32_t> d_mask(context, num_elements + 1, 0);
  Array1<uint32_t> d_out(context, num_elements, 0);

  Array1<uint64_t> d_key_value_pairs(context, num_elements);

  size_t temp_storage_bytes = 0;
  K2_CUDA_SAFE_CALL(
      cub::DeviceRadixSort::SortPairs(
          nullptr, temp_storage_bytes, d_mask.Data(), d_out.Data(),
          d_key_value_pairs.Data(), d_key_value_pairs.Data(), num_elements, 0,
          int32_t(ceil(log2(float(num_buckets))))),
      context->GetCudaStream());

  K2_CUDA_SAFE_CALL(
      MarkBinsGeneral<<<num_blocks, num_threads, 0, context->GetCudaStream()>>>(
          in_keys, num_elements, num_buckets, bucket_mapping_func,
          d_mask.Data()));

  K2_CUDA_SAFE_CALL(PackingKeyValuePairs<<<num_blocks, num_threads, 0,
                                           context->GetCudaStream()>>>(
      num_elements, in_keys, in_values, d_key_value_pairs.Data()));

  Array1<int8_t> d_temp_storage(context, temp_storage_bytes);

  K2_CUDA_SAFE_CALL(
      cub::DeviceRadixSort::SortPairs(
          d_temp_storage.Data(), temp_storage_bytes, d_mask.Data(),
          d_out.Data(), d_key_value_pairs.Data(), d_key_value_pairs.Data(),
          num_elements, 0, int32_t(ceil(log2(float(num_buckets))))),
      context->GetCudaStream());

  K2_CUDA_SAFE_CALL(UnpackingKeyValuePairs<<<num_blocks, num_threads, 0,
                                             context->GetCudaStream()>>>(
      num_elements, d_key_value_pairs.Data(), out_keys, out_values));
}

}  // namespace k2

#endif  // K2_CSRC_CUDPP_MULTISPLIT_H_
