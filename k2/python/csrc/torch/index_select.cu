/**
 * @brief Index select for k2.
 *
 * Unlike torch.index_select, when an entry is -1, it sets
 * the destination entry to 0.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corp.       (author: Daniel Povey)
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
                        If index[i] is -1, then ans[i] is 0
                        CAUTION: We require that index.is_contiguous() is true.
   @return
      Returns a 1-D tensor such that:
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
                            -1 <= index[i] < src.numel()
                            for i in [0, index.numel())
                        If index[i] is -1, then ans[i] is 0
                        CAUTION: We require that index.is_contiguous() is true.
   @return
      Returns a 1-D tensor such that:
          ans[i] = src[index[i]] if index[i] > 0
          ans[i] = 0 if index[i] is -1
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
      case ToScalarType<float>::value:
        return IndexSelect1D<float>(src, index);
      default:
        K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
        return {};
    }
  } else if (src.dim() == 2) {
    switch (scalar_type) {
      case ToScalarType<int32_t>::value:
        return IndexSelect2D<int32_t>(src, index);
      case ToScalarType<float>::value:
        return IndexSelect2D<float>(src, index);
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
                      one non-zero element in `src` and we'll set src[i]
                      with that non-zero element; if all elements for
                      sub-list `i` is zero or the sub-list is empty, we just
                      set src[i] == 0.
     @return   Returns a Tensor with the same dtype as `src` and shape
                     (indexes.Dim0()), i.e. a 1-D tensor with numel() equal
                     to `indexes.Dim0()`.
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

  int32_t src_numel = src.numel();
  const T *src_data = src.data_ptr<T>();
  int32_t indexes_dim0 = indexes.Dim0(),
          indexes_num_elems = indexes.NumElements();
  const int32_t *indexes_row_splits1_data = indexes.RowSplits(1).Data(),
                *indexes_row_ids1_data = indexes.RowIds(1).Data();
  const int32_t *indexes_data = indexes.values.Data();

  torch::Tensor ans = torch::zeros(indexes_dim0, src.options());
  T *ans_data = ans.data_ptr<T>();
  int64_t src_stride = src.strides()[0];
  int64_t ans_stride = ans.strides()[0];

  // check if there is at most one non-zero element in src for each sub-list
  Ragged<int32_t> non_zero_elems(indexes.shape,
                                 Array1<int32_t>(context, indexes_num_elems));
  int32_t *elems_data = non_zero_elems.values.Data();
  K2_EVAL(
      context, indexes_num_elems, lambda_count_non_zeros, (int32_t i)->void {
        int32_t src_index = indexes_data[i];
        K2_CHECK_GE(src_index, 0);
        K2_CHECK_LT(src_index, src_numel);
        T value = src_data[src_index * src_stride];
        elems_data[i] = (value != 0);
      });
  Array1<int32_t> counts(context, indexes_dim0);
  SumPerSublist(non_zero_elems, 0, &counts);
  const int32_t *counts_data = counts.Data();
  Array1<int32_t> status(context, 1, 0);  // 0 -> success; otherwise 1 + row_id
                                          // of bad row in `indexes`
  int32_t *status_data = status.Data();
  K2_EVAL(
      context, counts.Dim(), lambda_check_status, (int32_t i)->void {
        if (counts_data[i] > 1) status_data[0] = 1 + i;
      });
  int32_t s = status[0];
  if (s != 0) {
    Array1<T> indexed_values(context, indexes_num_elems);
    T *indexed_values_data = indexed_values.Data();
    K2_EVAL(context, indexes_num_elems, lambda_set_values, (int32_t i) -> void {
        int32_t src_index = indexes_data[i];
        indexed_values_data[i] = src_data[src_index * src_stride];
      });
    Array1<int32_t> row_splits = indexes.RowSplits(1);

    K2_LOG(FATAL) << "There must be at most one non-zero "
        "element in src for any sub-list in indexes; sub-list "
                  << (s-1) << " has too many elements: "
                  << indexed_values.Arange(row_splits[s-1],
                                           row_splits[s]);
  }

  K2_EVAL(
      context, indexes_num_elems, lambda_set_ans_data, (int32_t i)->void {
        int32_t src_index = indexes_data[i];
        T value = src_data[src_index * src_stride];
        int32_t ans_index =
            indexes_row_ids1_data[i];  // ans_index is idx0 in indexes
        if (value != 0) {
          ans_data[ans_index * ans_stride] = value;
        }
      });
  return ans;
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
                  << ". It supports only 1-D tensors";
    return {};
  }
}

static void IndexSelect(py::module &m) {
  m.def("index_select", &IndexSelectWrapper, py::arg("src"), py::arg("index"));
  m.def("simple_ragged_index_select", &SimpleRaggedIndexSelectWrapper,
        py::arg("src"), py::arg("indexes"));
}

}  // namespace k2

void PybindIndexSelect(py::module &m) { k2::IndexSelect(m); }
