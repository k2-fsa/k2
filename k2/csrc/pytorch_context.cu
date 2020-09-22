/**
 * @brief
 * pytorch_context
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <memory>

#include "c10/cuda/CUDAFunctions.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

ContextPtr GetCpuContext() { return PytorchCpuContext::Make(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  if (gpu_id < 0) gpu_id = c10::cuda::current_device();
  return std::make_shared<PytorchCudaContext>(gpu_id);
}

RegionPtr NewRegion(torch::Tensor &tensor) {
  auto ans = std::make_shared<Region>();
  if (tensor.device().type() == torch::kCPU) {
    ans->context = GetCpuContext();
  } else if (tensor.is_cuda()) {
    ans->context = GetCudaContext(tensor.device().index());
  } else {
    K2_LOG(FATAL) << "Unsupported device: " << tensor.device()
                  << "\nOnly CPU and CUDA are supported";
  }

  // NOTE: the tensor is passed from Python and we have
  // to retain it to avoid potential segmentation fault.
  //
  // It will be freed in `Context::Deallocate`.
  auto *managed_tensor = new ManagedTensor(tensor);
  ans->data = tensor.data_ptr();
  ans->deleter_context = managed_tensor;
  ans->num_bytes = tensor.nbytes();
  ans->bytes_used = ans->num_bytes;
  return ans;
}

}  // namespace k2
