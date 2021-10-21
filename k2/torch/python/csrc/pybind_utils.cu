/**
 * @copyright
 * Copyright       2021  Xiaomi Corp.       (author: Wei Kang)
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

#include <memory>

#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/pybind_utils.h"

namespace k2 {

py::object ToPyObject(torch::IValue value) {
  if (value.isCustomClass()) {
    return py::cast(ToRaggedAny(value));
  } else {
    return torch::jit::toPyObject(value);
  }
}

torch::IValue ToIValue(py::object obj) {
  auto inferred_type = torch::jit::tryToInferType(obj);
  if (inferred_type.success()) {
    return torch::jit::toIValue(obj, inferred_type.type());
  } else {
    try {
      RaggedAny ragged_tensor = obj.cast<RaggedAny>();
      return ToIValue(ragged_tensor);
    } catch (const py::cast_error &) {
      // TODO: Handle cast error
      return torch::jit::toIValue(obj, torch::jit::PyObjectType::get());
    }
  }
}

}  // namespace k2
