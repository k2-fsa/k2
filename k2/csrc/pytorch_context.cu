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

}  // namespace k2
