/**
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                2021  Xiaomi Corp.       (author: Daniel Povey,
 *                                                  Haowen Qiu,
 *                                                  Wei Kang)
 *
 * @copyright
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

#include "k2/csrc/context.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/torch/csrc/ops.h"

namespace k2 {

void IndexAdd(torch::Tensor index, torch::Tensor value, torch::Tensor *in_out) {
  NVTX_RANGE(K2_FUNC);
  DeviceGuard guard(GetContext(index));

  Array1<int32_t> indexes = FromTorch<int32_t>(index);
  Tensor src = FromTorch(value, TensorTag{});
  Tensor dest = FromTorch(*in_out, TensorTag{});
  IndexAdd(src, indexes, true, &dest);
}

/* Returns a 1-D tensor which indexes the src tensor using entries
   from `index`.

   @param  [in]  src    A 1-D tensor.
   @param  [in]  index  A 1-D tensor with dtype torch.int32.
                        It has to satisfy:
                            -1 <= index[i] < src.numel()
                            for i in [0, index.numel())
                        CAUTION: We require that index.is_contiguous()
                                 is true.
   @param [in] default_value  The value for ans[i] when index[i] is -1.
   @return
      Returns a 1-D contiguous tensor such that:
          ans[i] = src[index[i]] if index[i] > 0
          ans[i] = default_value if index[i] is -1
 */
template <typename T>
torch::Tensor IndexSelect1D(torch::Tensor src, torch::Tensor index,
                            T default_value) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value)
      << "Expected equal type"
      << " Given : " << src.scalar_type() << ", " << ToScalarType<T>::value;

  K2_CHECK_EQ(index.dim(), 1)
      << "Expected index dim: 1. Given : " << index.dim();
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value)
      << "Expected type int32_t Given : " << index.scalar_type();
  K2_CHECK(index.is_contiguous()) << "Expected contiguous";
  K2_CHECK_EQ(src.device(), index.device())
      << "Expected in the same device"
      << " Given : " << src.device() << ", " << index.device();

  bool allow_minus_one = true;
  Array1<int32_t> index_array = FromTorch<int32_t>(index);
  // If index_array.Dim() equals to zero, the `Index` below would produce an
  // ans with `ans.Data()` be a nullptr, which will cause crash when calling
  // `torch::from_blob`. Just return an empty tensor here.
  // If src is an empty tensor, we should return an empty torch.
  if (index_array.Dim() == 0 || src.numel() == 0)
    return torch::empty({0}, src.options());
  if (src.is_contiguous()) {
    Array1<T> src_array = FromTorch<T>(src);
    Array1<T> ans_array =
        Index(src_array, index_array, allow_minus_one, default_value);
    return ToTorch(ans_array);
  }
  Tensor tensor = FromTorch(src, TensorTag{});
  Tensor ans = Index(tensor, index_array, allow_minus_one, default_value);
  return ToTorch(ans);
}

/* Returns a 2-D tensor which indexes the src tensor using entries
   from `index`.

   @param  [in]  src    A 2-D tensor. If it is non-contiguous, then it
                        has to satisfy src.strides()[1] == 1.

   @param  [in]  index  A 1-D tensor with dtype torch.int32.
                        It has to satisfy:
                            -1 <= index[i] < src.shape()[0]
                            for i in [0, index.numel())
                        CAUTION: We require that index.is_contiguous()
                                 is true.
   @return
      Returns a 2-D contiguous tensor such that:
          ans[i] = src[index[i]] if index[i] > 0
          ans[i] = zero tensor whose numel() is src.shape()[1],
                   if index[i] is -1
 */
template <typename T>
torch::Tensor IndexSelect2D(torch::Tensor src, torch::Tensor index) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.dim(), 2) << "Expected dim: 2. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);

  K2_CHECK_EQ(index.dim(), 1);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
  K2_CHECK(index.is_contiguous());
  K2_CHECK_EQ(src.device(), index.device());

  Array2<T> src_array = FromTorch<T>(src, Array2Tag{});
  Array1<int32_t> index_array = FromTorch<int32_t>(index);
  // If index_array.Dim() equals to zero, the `IndexRows` below would produce
  // an ans with `ans.Data()` be a nullptr, which will cause crash when
  // calling `torch::from_blob`. Just return an empty tensor here.
  // If src is an empty tensor, we should return an empty torch.
  if (index_array.Dim() == 0 || src.sizes()[0] == 0)
    return torch::empty({0, src.sizes()[1]}, src.options());
  bool allow_minus_one = true;
  Array2<T> ans_array = IndexRows(src_array, index_array, allow_minus_one);

  return ToTorch(ans_array);
}

torch::Tensor IndexSelect(torch::Tensor src, torch::Tensor index,
                          double default_value /*= 0*/) {
  NVTX_RANGE(K2_FUNC);
  DeviceGuard guard(GetContext(src));
  auto scalar_type = src.scalar_type();
  if (src.dim() == 1) {
    switch (scalar_type) {
      case ToScalarType<int32_t>::value: {
        int32_t i = static_cast<int32_t>(default_value);
        K2_CHECK_EQ(static_cast<double>(i), default_value);
        return IndexSelect1D<int32_t>(src, index, i);
      }
      case ToScalarType<int64_t>::value: {
        int64_t i = static_cast<int64_t>(default_value);
        K2_CHECK_EQ(static_cast<double>(i), default_value);
        return IndexSelect1D<int64_t>(src, index, i);
      }
      case ToScalarType<float>::value:
        return IndexSelect1D<float>(src, index, default_value);
      case ToScalarType<double>::value:
        return IndexSelect1D<double>(src, index, default_value);
      default:
        K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
        return {};
    }
  } else if (src.dim() == 2) {
    switch (scalar_type) {
      case ToScalarType<int32_t>::value:
        return IndexSelect2D<int32_t>(src, index);
      case ToScalarType<int64_t>::value:
        return IndexSelect2D<int64_t>(src, index);
      case ToScalarType<float>::value:
        return IndexSelect2D<float>(src, index);
      case ToScalarType<double>::value:
        return IndexSelect2D<double>(src, index);
      default:
        K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
        return {};
    }
  } else {
    K2_LOG(FATAL) << "Unsupported dim: " << src.dim()
                  << ".\nIt supports only 1-D and 2-D tensors.";
    return {};
  }
}

/*
  Returns a 1-D Tensor that is a result of indexing 1-D `src` with Ragged
  array `indexes` whose NumAxes() is 2. ans.numel() will equal to
  indexes.Dim0() as we suppose there is at most one non-zero element in `src`
  for any indexes sub-list in `indexes`.

     @param [in] src  Source tensor, to be indexed.
     @param [in] indexes   Indexes to use whose NumAxes() == 2, for any
                      sub-list `i` in `indexes`, we suppose there is at most
                      one non-zero values in `src` and we'll set ans[i]
                      with that non-zero value; if all values for
                      sub-list `i` is zero or the sub-list is empty, we just
                      set ans[i] == 0.
     @return   Returns a Tensor with the same dtype as `src` and shape
                     (indexes.Dim0()), i.e. a 1-D tensor with numel() equal
                     to `indexes.Dim0()`.
                     Noted the ans would be contiguous even though `src`
                     is not contiguous.
 */
template <typename T>
torch::Tensor SimpleRaggedIndexSelect1D(torch::Tensor src,
                                        Ragged<int32_t> &indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);
  K2_CHECK_EQ(indexes.NumAxes(), 2);
  ContextPtr context = GetContext(src);
  K2_CHECK(context->IsCompatible(*indexes.Context()));

  Tensor tensor = FromTorch(src, TensorTag{});
  Tensor ans = SimpleRaggedIndexSelect1D(tensor, indexes);
  return ToTorch(ans);
}

torch::Tensor SimpleRaggedIndexSelect(torch::Tensor src, RaggedAny &ragged) {
  DeviceGuard guard(GetContext(src));
  Ragged<int32_t> indexes = ragged.any.Specialize<int32_t>();
  auto scalar_type = src.scalar_type();
  if (src.dim() == 1) {
    switch (scalar_type) {
      case ToScalarType<int32_t>::value:
        return SimpleRaggedIndexSelect1D<int32_t>(src, indexes);
      case ToScalarType<float>::value:
        return SimpleRaggedIndexSelect1D<float>(src, indexes);
      default:
        K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
        return {};
    }
  } else {
    K2_LOG(FATAL) << "Unsupported dim: " << src.dim()
                  << ". It supports only 1-D tensors for now";
    return {};
  }
}

}  // namespace k2
