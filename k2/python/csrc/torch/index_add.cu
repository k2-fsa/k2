/**
 * @brief index_add for k2.
 *
 * It has identical semantics as torch.Tensor.index_add_
 * except that it requires the dtype of the input index
 * to be torch.int32, whereas PyTorch expects the dtype to be
 * torch.int64. Furthermore, it ignores index[i] == -1.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/context.h"
#include "k2/python/csrc/torch/index_add.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

// It implements  (*in_out)[index[i]] += value[i]
// for i in range(index.numel())
void IndexAdd(torch::Tensor index, torch::Tensor value, torch::Tensor *in_out) {
  // We support only 1-D tensors at present
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
  K2_CHECK_EQ(index.dim(), 1);

  K2_CHECK_EQ(value.scalar_type(), ToScalarType<float>::value);
  K2_CHECK_EQ(value.dim(), 1);

  K2_CHECK_EQ(index.numel(), value.numel());

  K2_CHECK_EQ(in_out->scalar_type(), ToScalarType<float>::value);
  K2_CHECK_EQ(in_out->dim(), 1);

  const int32_t *index_data = index.data_ptr<int32_t>();
  const float *value_data = value.data_ptr<float>();
  float *in_out_data = in_out->data_ptr<float>();

  ContextPtr context;
  if (index.device().type() == torch::kCPU) {
    context = GetCpuContext();
  } else {
    K2_CHECK(index.is_cuda());
    context = GetCudaContext(index.device().index());
  }

  int32_t index_numel = index.numel();
  int32_t in_out_numel = in_out->numel();

  if (index.is_contiguous() && value.is_contiguous() &&
      in_out->is_contiguous()) {
    if (context->GetDeviceType() == kCpu) {
      for (int32_t i = 0; i != index_numel; ++i) {
        int32_t in_out_i = index_data[i];
        if (in_out_i == -1) continue;

        K2_DCHECK_GE(in_out_i, 0);
        K2_DCHECK_LT(in_out_i, in_out_numel);

        // for host thread, we assume single thread at present
        in_out_data[in_out_i] += value_data[i];
      }
    } else {
      // for cuda
      auto lambda_contiguous = [=] __device__(int32_t i) -> void {
        int32_t in_out_i = index_data[i];
        if (in_out_i == -1) return;

        K2_DCHECK_GE(in_out_i, 0);
        K2_DCHECK_LT(in_out_i, in_out_numel);

        atomicAdd(in_out_data + in_out_i, value_data[i]);
      };
      EvalDevice(context, index_numel, lambda_contiguous);
    }
    return;
  }

  // for non-contiguous tensors
  int64_t index_stride = index.strides()[0];
  int64_t value_stride = value.strides()[0];
  int64_t in_out_stride = in_out->strides()[0];

  if (context->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != index_numel; ++i) {
      int32_t in_out_i = index_data[i * index_stride];
      if (in_out_i == -1) continue;

      K2_DCHECK_GE(in_out_i, 0);
      K2_DCHECK_LT(in_out_i, in_out_numel);

      // for host thread, we assume single thread at present
      in_out_data[in_out_i * in_out_stride] += value_data[i * value_stride];
    }
  } else {
    // for cuda
    auto lambda_noncontiguous = [=] __device__(int32_t i) -> void {
      int32_t in_out_i = index_data[i * index_stride];
      if (in_out_i == -1) return;

      K2_DCHECK_GE(in_out_i, 0);
      K2_DCHECK_LT(in_out_i, in_out_numel);

      atomicAdd(in_out_data + in_out_i * in_out_stride,
                value_data[i * value_stride]);
    };
    EvalDevice(context, index_numel, lambda_noncontiguous);
  }
}

}  // namespace k2

void PybindIndexAdd(py::module &m) {
  m.def("index_add", &k2::IndexAdd, py::arg("index"), py::arg("value"),
        py::arg("in_out"));
}
