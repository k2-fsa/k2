/**
 * @brief
 * utils_inl
 *
 * @note
 * Don't include this file directly; it is included by utils.h.
 * It contains implementation code
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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
void ExclusiveSum(ContextPtr c, int32_t n, const SrcPtr src, DestPtr dest) {
  K2_CHECK_GE(n, 0);
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
  if (d == kCpu) {
    SumType sum = 0;
    for (int32_t i = 0; i != n; ++i) {
      auto prev = src[i];  // save a copy since src and dest
                           // may share the underlying memory
      dest[i] = sum;
      sum += prev;
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    // Determine temporary device storage requirements
    std::size_t temp_storage_bytes = 0;
    // the following function will compute the number of required bytes
    // for ExclusiveSum
    K2_CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, src, dest, n, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(
        cub::DeviceScan::ExclusiveSum(d_temp_storage.Data(), temp_storage_bytes,
                                      src, dest, n, c->GetCudaStream()));
  }
}
template <typename T>
T MaxValue(ContextPtr c, int32_t nelems, const T *t) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    // note the return value is initialized with T(0)
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
    std::size_t temp_storage_bytes = 0;
    // the first time is to determine temporary device storage requirements
    K2_CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(
        nullptr, temp_storage_bytes, t, max_value, nelems, max_op, init,
        c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(
        d_temp_storage.Data(), temp_storage_bytes, t, max_value, nelems, max_op,
        init, c->GetCudaStream()));
    // this will convert to memory on CPU
    return max_array[0];
  }
}
}  // namespace k2

#endif  // K2_CSRC_UTILS_INL_H_
