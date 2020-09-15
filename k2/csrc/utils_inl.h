/**
 * @brief
 * utils_inl
 *
 * @note
 * Don't include this file directly; it is included by utils.h.
 * It contains implementation code
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Fangjun Kuang (csukuangfj@gmail.com)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_UTILS_INL_H_
#define K2_CSRC_UTILS_INL_H_

#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <type_traits>

namespace k2 {

template <typename SrcPtr, typename DestPtr>
void ExclusiveSum(ContextPtr &c, int32_t n, SrcPtr src, DestPtr dest) {
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
  if (d == kCpu) {
    SumType sum = 0;
    for (int32_t i = 0; i != n; ++i) {
      dest[i] = sum;
      sum += src[i];
    }
  } else {
    assert(d == kCuda);
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // since d_temp_storage is nullptr, the following function will compute
    // the number of required bytes for d_temp_storage
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dest,
                                  n, c->GetCudaStream());
    void *deleter_context;
    d_temp_storage = c->Allocate(temp_storage_bytes, &deleter_context);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dest,
                                  n, c->GetCudaStream());
    c->Deallocate(d_temp_storage, deleter_context);
  }
}
}  // namespace k2

#endif  // K2_CSRC_UTILS_INL_H_
