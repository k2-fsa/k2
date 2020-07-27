// k2/csrc/cuda/utils_inl.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
//                      Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

/* Don't include this file directly; it is included by utils.h.
   It contains implementation code. */

#include <assert.h>
#include <type_traits>

#include "cub/cub.cuh"

#ifndef IS_IN_K2_CSRC_CUDA_UTILS_H_
#error "this file is supposed to be included only by utils.h"
#endif

namespace k2 {

template <typename SrcPtr, typename DestPtr>
void ExclusivePrefixSum(ContextPtr &c, int n, SrcPtr src, DestPtr dest) {
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
  if (d == kCpu) {
    SumType sum = 0;
    for (int i = 0; i != n; ++i) {
      dest[i] = sum;
      sum += src[i];
    }
  } else {
    assert(d == kCuda);
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // since d_temp_storeage is nullptr, the following function will computes
    // the number of required bytes for d_temp_storage
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dest,
                                  n, cudaStreamPerThread);
    d_temp_storage = c->Allocate(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dest,
                                  n);
    c->Deallocate(d_temp_storage);
  }
}

}  // namespace k2
