/**
 * @brief
 * utils_inl
 *
 * @note
 * Don't include this file directly; it is included by utils.h.
 * It contains implementation code
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
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

#include "k2/csrc/array.h"

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
    K2_CHECK_EQ(d, kCuda);
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // since d_temp_storage is nullptr, the following function will compute
    // the number of required bytes for d_temp_storage
    K2_CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, src, dest, n, c->GetCudaStream()));
    void *deleter_context;
    d_temp_storage = c->Allocate(temp_storage_bytes, &deleter_context);
    K2_CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, src, dest, n, c->GetCudaStream()));
    c->Deallocate(d_temp_storage, deleter_context);
  }
}
template <typename T>
T MaxValue(ContextPtr &c, int32_t nelems, T *t) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    // not the return value is initialized with T(0)
    T result = T(0);
    for (int32_t i = 0; i < nelems; ++i) {
      if (result < t[i]) result = t[i];
    }
    return result;
  } else {
    K2_CHECK_EQ(d, kCuda);
    MaxOp<T> max_op;
    T init = T(0);
    Array1<T> max_array(c, 1, T(0));
    T *max_value = max_array.Data();
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // the first time is to determine temporary device storage requirements
    K2_CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, t, max_value, nelems, max_op, init,
        c->GetCudaStream()));
    void *deleter_context;
    d_temp_storage = c->Allocate(temp_storage_bytes, &deleter_context);
    K2_CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, t, max_value, nelems, max_op, init,
        c->GetCudaStream()));
    c->Deallocate(d_temp_storage, deleter_context);
    // this will convert to memory on CPU
    return max_array[0];
  }
}
}  // namespace k2

#endif  // K2_CSRC_UTILS_INL_H_
