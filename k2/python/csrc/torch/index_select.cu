/**
 * @brief Index select for k2.
 *
 * Unlike torch.index_select, when an entry is -1, it sets
 * the destination entry to 0.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/python/csrc/torch/index_select.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

template <typename T>
static torch::Tensor IndexSelect1D(torch::Tensor src, torch::Tensor index) {
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);

  K2_CHECK_EQ(index.dim(), 1);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);

  ContextPtr context;
  if (src.device().type() == torch::kCPU) {
    context = GetCpuContext();
  } else {
    K2_CHECK(src.is_cuda());
    context = GetCudaContext(src.device().index());
  }

  const T *src_data = src.data_ptr<T>();
  int32_t src_numel = src.numel();
  const int32_t *index_data = index.data_ptr<int32_t>();

  torch::Tensor ans = torch::empty(index.sizes(), src.options());
  T *ans_data = ans.data_ptr<T>();
  int32_t index_numel = index.numel();

  if (src.is_contiguous() && index.is_contiguous()) {
    if (context->GetDeviceType() == kCpu) {
      for (int32_t i = 0; i != index_numel; ++i) {
        int32_t src_i = index_data[i];
        if (src_i != -1) {
          K2_DCHECK_GE(src_i, 0);
          K2_DCHECK_LT(src_i, src_numel);

          ans_data[i] = src_data[src_i];
        } else {
          ans_data[i] = 0;
        }
      }
    } else {
      auto lambda = [=] __device__(int32_t i) -> void {
        int32_t src_i = index_data[i];
        if (src_i != -1) {
          K2_DCHECK_GE(src_i, 0);
          K2_DCHECK_LT(src_i, src_numel);

          ans_data[i] = src_data[src_i];
        } else {
          ans_data[i] = 0;
        }
      };
      EvalDevice(context, index_numel, lambda);
    }
    return ans;
  }

  // for non contiguous tensors
  int64_t src_stride = src.strides()[0];
  int64_t index_stride = index.strides()[0];
  int64_t ans_stride = ans.strides()[0];

  if (context->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != index_numel; ++i) {
      int32_t src_i = index_data[i * index_stride];
      if (src_i != -1) {
        K2_DCHECK_GE(src_i, 0);
        K2_DCHECK_LT(src_i, src_numel);

        ans_data[i * ans_stride] = src_data[src_i * src_stride];
      } else {
        ans_data[i * ans_stride] = 0;
      }
    }
  } else {
    auto lambda_noncontiguous = [=] __device__(int32_t i) -> void {
      int32_t src_i = index_data[i * index_stride];
      if (src_i != -1) {
        K2_DCHECK_GE(src_i, 0);
        K2_DCHECK_LT(src_i, src_numel);

        ans_data[i * ans_stride] = src_data[src_i * src_stride];
      } else {
        ans_data[i * ans_stride] = 0;
      }
    };
    EvalDevice(context, index_numel, lambda_noncontiguous);
  }
  return ans;
}

torch::Tensor IndexSelectWrapper(torch::Tensor src, torch::Tensor index) {
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
  } else {
    K2_LOG(FATAL) << "Unsupported dim: " << src.dim();
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
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);

  K2_CHECK_EQ(indexes.NumAxes(), 2);

  ContextPtr context;
  if (src.device().type() == torch::kCPU) {
    context = GetCpuContext();
  } else {
    K2_CHECK(src.is_cuda());
    context = GetCudaContext(src.device().index());
  }

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

#if !defined(NDEBUG)
  // check if there is at most one non-zero element in src for each sub-list
  Ragged<int32_t> non_zero_elems(indexes.shape,
                                 Array1<int32_t>(context, indexes_num_elems));
  int32_t *elems_data = non_zero_elems.values.Data();
  auto lambda_count_non_zeros = [=] __host__ __device__(int32_t i) -> void {
    int32_t src_index = indexes_data[i];
    K2_CHECK_GE(src_index, 0);
    K2_CHECK_LT(src_index, src_numel);
    T value = src_data[src_index * src_stride];
    elems_data[i] = (value != 0);
  };
  Eval(context, indexes_num_elems, lambda_count_non_zeros);
  Array1<int32_t> counts(context, indexes_dim0);
  SumPerSublist(non_zero_elems, 0, &counts);
  const int32_t *counts_data = counts.Data();
  Array1<int32_t> status(context, 1, 0);
  int32_t *status_data = status.Data();
  auto lambda_check_status = [=] __host__ __device__(int32_t i) -> void {
    if (counts_data[i] > 1) status_data[0] = 1;
  };
  Eval(context, counts.Dim(), lambda_check_status);
  K2_CHECK_EQ(status[0], 0) << "There must be at most one non-zero "
                               "element in src for any sub-list in indexes";
#endif

  if (context->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != indexes_num_elems; ++i) {
      int32_t src_index = indexes_data[i];
      T value = src_data[src_index * src_stride];
      int32_t ans_index =
          indexes_row_ids1_data[i];  // ans_index is idx0 in indexes
      if (value != 0) {
        ans_data[ans_index * ans_stride] = value;
      }
    }
  } else {
    auto lambda_set_ans_data = [=] __device__(int32_t i) -> void {
      int32_t src_index = indexes_data[i];
      T value = src_data[src_index * src_stride];
      int32_t ans_index =
          indexes_row_ids1_data[i];  // ans_index is idx0 in indexes
      if (value != 0) {
        ans_data[ans_index * ans_stride] = value;
      }
    };
    EvalDevice(context, indexes_num_elems, lambda_set_ans_data);
  }
  return ans;
}

torch::Tensor SimpleRaggedIndexSelectWrapper(torch::Tensor src,
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
    K2_LOG(FATAL) << "Unsupported dim: " << src.dim();
    return {};
  }
}

void IndexSelect(py::module &m) {
  m.def("index_select", &IndexSelectWrapper, py::arg("src"), py::arg("index"));
  m.def("simple_ragged_index_select", &SimpleRaggedIndexSelectWrapper,
        py::arg("src"), py::arg("indexes"));
}

}  // namespace k2

void PybindIndexSelect(py::module &m) { k2::IndexSelect(m); }
