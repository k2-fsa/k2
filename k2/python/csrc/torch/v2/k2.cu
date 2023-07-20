/**
 * @brief python wrapper for k2 2.0
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

#include "k2/python/csrc/torch/v2/any.h"
#include "k2/python/csrc/torch/v2/autograd/swoosh.h"
#include "k2/python/csrc/torch/v2/k2.h"
#include "k2/python/csrc/torch/v2/ragged_shape.h"

namespace k2 {

void PybindV2(py::module &m) {
  py::module ragged = m.def_submodule(
      "ragged", "Sub module containing operations for ragged tensors in k2");

  PybindRaggedShape(ragged);

  m.attr("RaggedShape") = ragged.attr("RaggedShape");

  PybindRaggedAny(ragged);

  PybindSwoosh(m);
}

}  // namespace k2
