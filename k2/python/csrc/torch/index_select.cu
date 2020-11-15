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

void IndexSelect(py::module &m) {
  m.def("index_select", &IndexSelectWrapper, py::arg("src"), py::arg("index"));
}

}  // namespace k2

void PybindIndexSelect(py::module &m) { k2::IndexSelect(m); }
