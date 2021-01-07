// k2/csrc/cudpp/multisplit.cu
/*
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * This file is modified from
 * https://github.com/cudpp/cudpp/blob/master/src/cudpp/kernel/multisplit_kernel.cuh
 */

#include "k2/csrc/multisplit.h"

namespace k2 {

__global__ void PackingKeyValuePairs(int32_t num_elements,
                                     const uint32_t *input_key,
                                     const uint32_t *input_value,
                                     uint64_t *packed) {
  int32_t my_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (my_id >= num_elements) return;

  uint32_t my_key = input_key[my_id];
  uint32_t my_value = input_value[my_id];

  // putting the key as the more significant 32 bits.
  uint64_t output = (static_cast<uint64_t>(my_key) << 32) + my_value;

  packed[my_id] = output;
}

__global__ void UnpackingKeyValuePairs(int32_t num_elements,
                                       const uint64_t *packed,
                                       uint32_t *out_key, uint32_t *out_value) {
  int32_t my_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (my_id >= num_elements) return;

  uint64_t my_packed = packed[my_id];

  out_key[my_id] = static_cast<uint32_t>(my_packed >> 32);

  out_value[my_id] = static_cast<uint32_t>(my_packed & 0x00000000ffffffff);
}

}  // namespace k2
