/**
 * @brief Index select for k2.
 *
 * Unlike torch.index_select, when an entry is -1, it sets
 * the destination entry to 0.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corp.       (author: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/python/csrc/torch/index_select.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

/* Returns a 1-D tensor which indexes the src tensor using entries
   from `index`.

   @param  [in]  src    A 1-D tensor.
   @param  [in]  index  A 1-D tensor with dtype torch.int32.
                        It has to satisfy:
                            -1 <= index[i] < src.numel()
                            for i in [0, index.numel())
                        CAUTION: We require that index.is_contiguous() is true.
   @return
      Returns a 1-D contiguous tensor such that:
          ans[i] = src[index[i]] if index[i] > 0
          ans[i] = 0 if index[i] is -1
 */
template <typename T>
static torch::Tensor IndexSelect1D(torch::Tensor src, torch::Tensor index) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);

  K2_CHECK_EQ(index.dim(), 1);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
  K2_CHECK(index.is_contiguous());
  K2_CHECK_EQ(src.device(), index.device());

  Array1<int32_t> index_array = FromTensor<int32_t>(index);
  if (src.is_contiguous()) {
    Array1<T> src_array = FromTensor<T>(src);
    bool allow_minus_one = true;
    Array1<T> ans_array = Index(src_array, index_array, allow_minus_one);
    return ToTensor(ans_array);
  }

  Tensor tensor = FromTensor(src, TensorTag{});
  Tensor ans = Index(tensor, index_array);
  return ToTensor(ans);
}

/* Returns a 2-D tensor which indexes the src tensor using entries
   from `index`.

   @param  [in]  src    A 2-D tensor. If it is non-contiguous, then it
                        has to satisfy src.strides()[1] == 1.

   @param  [in]  index  A 1-D tensor with dtype torch.int32.
                        It has to satisfy:
                            -1 <= index[i] < src.shape()[0]
                            for i in [0, index.numel())
                        CAUTION: We require that index.is_contiguous() is true.
   @return
      Returns a 2-D contiguous tensor such that:
          ans[i] = src[index[i]] if index[i] > 0
          ans[i] = zero tensor whose numel() is src.shape()[1],
                   if index[i] is -1
 */
template <typename T>
static torch::Tensor IndexSelect2D(torch::Tensor src, torch::Tensor index) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.dim(), 2) << "Expected dim: 2. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);

  K2_CHECK_EQ(index.dim(), 1);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
  K2_CHECK(index.is_contiguous());
  K2_CHECK_EQ(src.device(), index.device());

  Array2<T> src_array = FromTensor<T>(src, Array2Tag{});
  Array1<int32_t> index_array = FromTensor<int32_t>(index);
  bool allow_minus_one = true;
  Array2<T> ans_array = IndexRows(src_array, index_array, allow_minus_one);

  return ToTensor(ans_array);
}

static torch::Tensor IndexSelectWrapper(torch::Tensor src,
                                        torch::Tensor index) {
  NVTX_RANGE(K2_FUNC);
  auto scalar_type = src.scalar_type();
  if (src.dim() == 1) {
    switch (scalar_type) {
      case ToScalarType<int32_t>::value:
        return IndexSelect1D<int32_t>(src, index);
      case ToScalarType<int64_t>::value:
        return IndexSelect1D<int64_t>(src, index);
      case ToScalarType<float>::value:
        return IndexSelect1D<float>(src, index);
      case ToScalarType<double>::value:
        return IndexSelect1D<double>(src, index);
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
  Returns a 1-D Tensor that is a result of indexing 1-D `src` with Ragged array
  `indexes` whose NumAxes() is 2. ans.numel() will equal to indexes.Dim0() as we
  suppose there is at most one non-zero element in `src` for any indexes
  sub-list in `indexes`.

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
static torch::Tensor SimpleRaggedIndexSelect1D(torch::Tensor src,
                                               Ragged<int32_t> &indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);
  K2_CHECK_EQ(indexes.NumAxes(), 2);
  ContextPtr context = GetContext(src);
  K2_CHECK(context->IsCompatible(*indexes.Context()));

  Tensor tensor = FromTensor(src, TensorTag{});
  Tensor ans = SimpleRaggedIndexSelect1D(tensor, indexes);
  return ToTensor(ans);
}

static torch::Tensor SimpleRaggedIndexSelectWrapper(torch::Tensor src,
                                                    Ragged<int32_t> &indexes) {
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

static void IndexSelect(py::module &m) {
  m.def("index_select", &IndexSelectWrapper, py::arg("src"), py::arg("index"),
        R"(
      Args:
        src:
          It can be either a 1-D or a 2-D tensor. Supported dtypes are:
          `torch.int32`, `torch.int64`, `torch.float32`, and `torch.float64`.
        index:
          It has to be a 1-D **contiguous** tensor with dtype `torch.int32`.
          Must satisfy `-1 <= index[i] < src.shape[0]`.
      Returns:
        Return a tensor:
          - `ans.ndim == src.ndim`
          - `ans.shape[0] == index.shape[0]`
          - If `ans.ndim == 2`, then `ans.shape[1] == src.shape[1]`
          - `ans[i] = src[index[i]]` if `index[i] != -1`.
          - `ans[i] = 0` if `index[i] == -1`
      )");
  m.def("simple_ragged_index_select", &SimpleRaggedIndexSelectWrapper,
        py::arg("src"), py::arg("indexes"));
}

}  // namespace k2

void PybindIndexSelect(py::module &m) { k2::IndexSelect(m); }
