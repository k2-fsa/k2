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
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/python/csrc/torch/index_add.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

/* Accumulate the elements of tensor into the `in_out` tensor by adding to
   the indices in the order given in `index`.

   For example, in_out[index[i]] += value[i] if index[i] >= 0.
   If index[i] is -1, value[i] is ignored.

   @param  [in]  index   A 1-D tensor of dtype torch.int32.
   @param  [in]  value   A 1-D tensor.
   @param  [inout] in_out  A 1-D tensor.
 */
// template<typename T> // TODO(fangjun): change it to a template
static void IndexAdd1D(torch::Tensor index, torch::Tensor value,
                       torch::Tensor *in_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
  K2_CHECK_EQ(index.dim(), 1);

  K2_CHECK_EQ(value.scalar_type(), ToScalarType<float>::value)
      << "index_add is supposed to be used in back propagation, "
         "which means the value type should be of float";
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
    K2_EVAL(
        context, index_numel, lambda_contiguous, (int32_t i)->void {
          int32_t in_out_i = index_data[i];
          if (in_out_i == -1) return;

          K2_DCHECK_GE(in_out_i, 0);
          K2_DCHECK_LT(in_out_i, in_out_numel);

#ifdef __CUDA_ARCH__
          atomicAdd(in_out_data + in_out_i, value_data[i]);
#else
          // for host thread, we assume single thread at present
          in_out_data[in_out_i] += value_data[i];
#endif
        });
    return;
  }

  // for non-contiguous tensors
  int64_t index_stride = index.strides()[0];
  int64_t value_stride = value.strides()[0];
  int64_t in_out_stride = in_out->strides()[0];

  K2_EVAL(
      context, index_numel, lambda_noncontiguous, (int32_t i)->void {
        int32_t in_out_i = index_data[i * index_stride];
        if (in_out_i == -1) return;

        K2_DCHECK_GE(in_out_i, 0);
        K2_DCHECK_LT(in_out_i, in_out_numel);

#ifdef __CUDA_ARCH__
        atomicAdd(in_out_data + in_out_i * in_out_stride,
                  value_data[i * value_stride]);
#else
      // for host thread, we assume single thread at present
      in_out_data[in_out_i * in_out_stride] += value_data[i * value_stride];
#endif
      });
}

/* Accumulate the elements of tensor into the `in_out` tensor by adding to
   the indices in the order given in `index`.

   For example, in_out[index[i]] += value[i] if index[i] >= 0.
   If index[i] is -1, value[i] is ignored. As `value` is a 2-D
   tensor we do element-wise addition here.

   @param  [in]  index   A 1-D  tensor of dtype torch.int32.
   @param  [in]  value   A 2-D tensor.
   @param  [inout] in_out  A 2-D tensor.
 */
// template<typename T> // TODO(fangjun): change it to a template
static void IndexAdd2D(torch::Tensor index, torch::Tensor value,
                       torch::Tensor *in_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
  K2_CHECK_EQ(index.dim(), 1);

  K2_CHECK_EQ(value.scalar_type(), ToScalarType<float>::value)
      << "index_add is supposed to be used in back propagation, "
         "which means the value type should be of float";
  K2_CHECK_EQ(value.dim(), 2);

  K2_CHECK_EQ(in_out->scalar_type(), ToScalarType<float>::value);
  K2_CHECK_EQ(in_out->dim(), 2);
  K2_CHECK_EQ(in_out->sizes()[1], value.sizes()[1]);

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
  int32_t in_out_num_rows = static_cast<int32_t>(in_out->sizes()[0]);
  int32_t in_out_num_cols = static_cast<int32_t>(in_out->sizes()[1]);

  if (index.is_contiguous() && value.is_contiguous() &&
      in_out->is_contiguous()) {
    K2_EVAL(
        context, index_numel, lambda_contiguous, (int32_t i)->void {
          int32_t in_out_i = index_data[i];
          if (in_out_i == -1) return;

          K2_DCHECK_GE(in_out_i, 0);
          K2_DCHECK_LT(in_out_i, in_out_num_rows);
          float *cur_in_out_data = in_out_data + in_out_i * in_out_num_cols;
          const float *cur_value_data = value_data + i * in_out_num_cols;
          for (int32_t j = 0; j != in_out_num_cols; ++j) {
#ifdef __CUDA_ARCH__
            atomicAdd(cur_in_out_data + j, cur_value_data[j]);
#else
            // for host thread, we assume single thread at present
            cur_in_out_data[j] += cur_value_data[j];
#endif
          }
        });
    return;
  }

  // for non-contiguous case
  // we require that the stride for columns is 1
  K2_CHECK_EQ(static_cast<int32_t>(in_out->strides()[1]), 1);
  int64_t in_out_stride = in_out->strides()[0];
  int64_t index_stride = index.strides()[0];

  // NOTE: value.strides() may be `(0, 0)`.
  int64_t value_stride0 = value.strides()[0];
  int64_t value_stride1 = value.strides()[1];

  K2_EVAL(
      context, index_numel, lambda_noncontiguous, (int32_t i)->void {
        int32_t in_out_i = index_data[i * index_stride];
        if (in_out_i == -1) return;
        K2_DCHECK_GE(in_out_i, 0);
        K2_DCHECK_LT(in_out_i, in_out_num_rows);

        float *cur_in_out_data = in_out_data + in_out_i * in_out_stride;
        const float *cur_value_data = value_data + i * value_stride0;
        for (int32_t j = 0; j != in_out_num_cols; ++j) {
#ifdef __CUDA_ARCH__
          atomicAdd(cur_in_out_data + j, cur_value_data[j]);
#else
          // for host thread, we assume single thread at present
          cur_in_out_data[j] += cur_value_data[j * value_stride1];
#endif
        }
      });
}

static void IndexAddWrapper(torch::Tensor index, torch::Tensor value,
                            torch::Tensor *in_out) {
  NVTX_RANGE(K2_FUNC);
  switch (value.dim()) {
    case 1:
      IndexAdd1D(index, value, in_out);
      break;
    case 2:
      IndexAdd2D(index, value, in_out);
      break;
    default:
      K2_LOG(FATAL) << "Unsupported dim: " << value.dim()
                    << ".\n Only 1-D and 2-D tensors are supported.";
      break;
  }
}

}  // namespace k2

void PybindIndexAdd(py::module &m) {
  m.def("index_add", &k2::IndexAddWrapper, py::arg("index"), py::arg("value"),
        py::arg("in_out"));
}
