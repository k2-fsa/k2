/**
 * @brief
 * pytorch_context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_PYTORCH_CONTEXT_H_
#define K2_CSRC_PYTORCH_CONTEXT_H_

#include <memory>

#include "k2/csrc/context.h"
#include "torch/torch.h"

namespace k2 {

class ManagedTensor {
 public:
  explicit ManagedTensor(torch::Tensor tensor) : handle_(tensor) {}

 private:
  torch::Tensor handle_;  // retain a copy of the tensor passed from Python
};

// Construct a region from a `torch::Tensor`.
//
// The resulting region shares the underlying memory with
// the given tensor.
RegionPtr NewRegion(torch::Tensor tensor);

}  // namespace k2

#endif  // K2_CSRC_PYTORCH_CONTEXT_H_
