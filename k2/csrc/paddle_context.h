/**
 * Copyright      2023  PaddlePaddle    (authors: Hui Zhang)
 *
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

#ifndef K2_CSRC_PADDLE_CONTEXT_H_
#define K2_CSRC_PADDLE_CONTEXT_H_

#include <memory>

#include "k2/csrc/context.h"
#include "paddle/all.h"

namespace k2 {

class ManagedTensor {
 public:
  explicit ManagedTensor(paddle::Tensor tensor) : handle_(tensor) {}

 private:
  paddle::Tensor handle_;  // retain a copy of the tensor passed from Python
};

// Construct a region from a `paddle::Tensor`.
//
// The resulting region shares the underlying memory with
// the given tensor.
RegionPtr NewRegion(paddle::Tensor tensor);

}  // namespace k2

#endif  // K2_CSRC_PADDLE_CONTEXT_H_
