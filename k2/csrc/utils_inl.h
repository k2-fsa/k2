/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef K2_CSRC_UTILS_INL_H_
#define K2_CSRC_UTILS_INL_H_

#include <type_traits>

#include "k2/csrc/array.h"
#include "k2/csrc/cub.h"
#ifdef K2_WITH_MPS
#include "k2/csrc/mps_utils.h"
#endif

namespace k2 {
template <typename SrcPtr, typename DestPtr>
void ExclusiveSum(ContextPtr c, int32_t n, const SrcPtr src, DestPtr dest) {
  K2_CHECK_GE(n, 0);
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
#ifdef K2_WITH_MPS
  if (d == kMps) {
    // Dispatch to Metal-safe ATen cumsum. Only int32_t raw pointers are
    // supported on MPS (all k2 row_splits / row_ids use int32_t).
    using RawSrc = std::decay_t<SrcPtr>;
    using RawDest = std::decay_t<DestPtr>;
    if constexpr (std::is_pointer_v<RawSrc> && std::is_pointer_v<RawDest> &&
                  std::is_same_v<
                      std::remove_cv_t<std::remove_pointer_t<RawSrc>>,
                      int32_t> &&
                  std::is_same_v<
                      std::remove_cv_t<std::remove_pointer_t<RawDest>>,
                      int32_t>) {
      mps_ops::ExclusiveSumMps(n,
                               reinterpret_cast<const int32_t *>(src),
                               reinterpret_cast<int32_t *>(dest));
    } else {
      K2_LOG(FATAL)
          << "ExclusiveSum on MPS only supports int32_t raw pointers";
    }
    return;
  }
#endif
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
    // for ExclusiveScan
    //
    // See https://github.com/NVIDIA/cub/issues/302
    // for why to prefer ExclusiveScan over ExclusiveSum
    //

    K2_CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveScan(
        nullptr, temp_storage_bytes, src, dest, cuda::std::plus<SumType>(),
        SumType(0), n, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveScan(
        d_temp_storage.Data(), temp_storage_bytes, src, dest,
        cuda::std::plus<SumType>(), SumType(0), n, c->GetCudaStream()));
  }
}

template <typename SrcPtr, typename DestPtr>
void InclusiveSum(ContextPtr c, int32_t n, const SrcPtr src, DestPtr dest) {
  K2_CHECK_GE(n, 0);
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
#ifdef K2_WITH_MPS
  if (d == kMps) {
    using RawSrc = std::decay_t<SrcPtr>;
    using RawDest = std::decay_t<DestPtr>;
    if constexpr (std::is_pointer_v<RawSrc> && std::is_pointer_v<RawDest> &&
                  std::is_same_v<
                      std::remove_cv_t<std::remove_pointer_t<RawSrc>>,
                      int32_t> &&
                  std::is_same_v<
                      std::remove_cv_t<std::remove_pointer_t<RawDest>>,
                      int32_t>) {
      mps_ops::InclusiveSumMps(n,
                               reinterpret_cast<const int32_t *>(src),
                               reinterpret_cast<int32_t *>(dest));
    } else {
      K2_LOG(FATAL)
          << "InclusiveSum on MPS only supports int32_t raw pointers";
    }
    return;
  }
#endif
  if (d == kCpu) {
    SumType sum = 0;
    for (int32_t i = 0; i != n; ++i) {
      sum += src[i];
      dest[i] = sum;
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    // Determine temporary device storage requirements
    std::size_t temp_storage_bytes = 0;
    // the following function will compute the number of required bytes
    // for InclusiveSum
    K2_CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, src, dest, n, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(
        cub::DeviceScan::InclusiveSum(d_temp_storage.Data(), temp_storage_bytes,
                                      src, dest, n, c->GetCudaStream()));
  }
}

template <typename T>
T MaxValue(ContextPtr c, int32_t nelems, const T *t) {
  DeviceType d = c->GetDeviceType();
#ifdef K2_WITH_MPS
  if (d == kMps) {
    // Use ATen reduction for Metal-safe access. In k2, MaxValue on MPS is
    // always called with int32_t (row_splits); other types are not supported.
    if constexpr (std::is_same_v<T, int32_t>) {
      return static_cast<T>(
          mps_ops::AsMpsTensor(t, static_cast<int64_t>(nelems))
              .max().template item<int32_t>());
    } else {
      K2_LOG(FATAL) << "MaxValue on MPS only supports int32_t";
      return T(0);  // unreachable
    }
  }
#endif
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
    K2_CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes,
                                                  t, max_value, nelems, max_op,
                                                  init, c->GetCudaStream()));
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
