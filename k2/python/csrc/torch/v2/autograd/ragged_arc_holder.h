/**
 * @brief A wrapper around k2::RaggedArc to support autograd.
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_RAGGED_ARC_HOLDER_H_
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_RAGGED_ARC_HOLDER_H_

#include "k2/python/csrc/torch/v2/ragged_arc.h"
#include "torch/torch.h"

namespace k2 {

// We need this wrapper so that we can save an instance
// of RaggedArc into `torch::autograd::AutogradContext *ctx`
struct RaggedArcHolder : public torch::CustomClassHolder {
  RaggedArc *fsas = nullptr;  // not owned by this class
  explicit RaggedArcHolder(RaggedArc *fsas) : fsas(fsas) {}
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_RAGGED_ARC_HOLDER_H_
