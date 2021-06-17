/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey)
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

#ifndef K2_CSRC_RAGGED_TENSOR_OPS_H_
#define K2_CSRC_RAGGED_TENSOR_OPS_H_

#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/utils.h"

namespace k2 {
// This file declares ops involving RaggedShape, Ragged<int32_t>,
// and Tensor.  They are implemented in ragged_tensor_ops.cu
// (they don't need to be in the header as Tensor doesn't have type
// information, so these functions are not templated).







}  // namespace k2


#endif  // K2_CSRC_RAGGED_TENSOR_OPS_H_
