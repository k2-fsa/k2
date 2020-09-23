/**
 * @copyright
 * Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

torch::DeviceType ToTorchDeviceType(DeviceType type) {
  switch (type) {
    case kCuda:
      return torch::kCUDA;
    case kCpu:
      return torch::kCPU;
    case kUnk:  // fall-through
    default:
      K2_LOG(FATAL) << "kUnk is not supported!";
      return torch::kCPU;  // unreachable code
  }
}

}  // namespace k2
