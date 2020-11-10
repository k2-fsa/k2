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
  K2_CHECK(src.is_contiguous());

  K2_CHECK(index.is_contiguous());
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

  auto lambda = [=] __host__ __device__(int32_t i) {
    int32_t src_i = index_data[i];
    if (src_i != -1) {
      K2_DCHECK_GE(src_i, 0);
      K2_DCHECK_LT(src_i, src_numel);

      ans_data[i] = src_data[src_i];
    } else {
      ans_data[i] = 0;
    }
  };
  Eval(context, index.numel(), lambda);
  return ans;
}

static torch::Tensor IndexSelect1DBackward(torch::Tensor src,
                                           torch::Tensor index,
                                           torch::Tensor grad) {
  K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
  K2_CHECK_EQ(src.scalar_type(), ToScalarType<float>::value);
  K2_CHECK(src.is_contiguous());

  K2_CHECK(index.is_contiguous());
  K2_CHECK_EQ(index.dim(), 1);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);

  K2_CHECK(grad.is_contiguous());
  K2_CHECK_EQ(grad.dim(), 1);
  K2_CHECK_EQ(index.numel(), grad.numel());
  K2_CHECK_EQ(grad.scalar_type(), ToScalarType<float>::value);

  ContextPtr context;
  if (src.device().type() == torch::kCPU) {
    context = GetCpuContext();
  } else {
    K2_CHECK(src.is_cuda());
    context = GetCudaContext(src.device().index());
  }

  const float *src_data = src.data_ptr<float>();
  const int32_t *index_data = index.data_ptr<int32_t>();
  const float *grad_data = grad.data_ptr<float>();

  torch::Tensor ans = torch::zeros(src.sizes(), src.options());
  float *ans_data = ans.data_ptr<float>();
  auto lambda = [=] __host__ __device__(int32_t i) -> void {
    int32_t src_i = index_data[i];
    if (src_i != -1) {
#ifdef __CUDA_ARCH__
      atomicAdd(ans_data + src_i, grad_data[i]);
#else
      // for host thread, we assume single thread now
      ans_data[src_i] += grad_data[i];
#endif
    }
  };
  Eval(context, index.numel(), lambda);
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
        K2_LOG(FATAL) << "Unknown scalar type: " << scalar_type;
        return {};
    }
  } else {
    K2_LOG(FATAL) << "Unsupported dim: " << src.dim();
    return {};
  }
}

torch::Tensor IndexSelectBackwardWrapper(torch::Tensor src, torch::Tensor index,
                                         torch::Tensor grad) {
  auto scalar_type = src.scalar_type();
  if (src.dim() == 1) {
    return IndexSelect1DBackward(src, index, grad);
  } else {
    K2_LOG(FATAL) << "Unsupported dim: " << src.dim();
    return {};
  }
}

void IndexSelect(py::module &m) {
  m.def("index_select", &IndexSelectWrapper, py::arg("src"), py::arg("index"));
  m.def("index_select_backward", &IndexSelectBackwardWrapper, py::arg("src"),
        py::arg("index"), py::arg("grad"));
}

}  // namespace k2

void PybindIndexSelect(py::module &m) { k2::IndexSelect(m); }
