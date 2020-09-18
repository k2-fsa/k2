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

#include "k2/csrc/log.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

ContextPtr GetCpuContext() { return std::make_shared<PytorchContext>(-1); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  if (gpu_id < 0) gpu_id = 0;  // TODO(fangjun): select a device
  return std::make_shared<PytorchContext>(gpu_id);
}

}  // namespace k2
