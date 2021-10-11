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

#ifndef K2_PYTHON_CSRC_TORCH_V2_PYBIND_UTILS_H_
#define K2_PYTHON_CSRC_TORCH_V2_PYBIND_UTILS_H_

#include <memory>

#include "k2/python/csrc/k2.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"
#include "torch/csrc/jit/python/pybind_utils.h"

namespace k2 {

py::object ToPyObject(torch::IValue value) {
  if (value.isCustomClass()) {
    torch::intrusive_ptr<RaggedAnyHolder> ragged_any_holder =
        value.toCustomClass<RaggedAnyHolder>();
    return py::cast(*(ragged_any_holder->ragged));
  } else {
    return torch::jit::toPyObject(value);
  }
  // Unreachable code
  return py::none();
}

torch::IValue ToIValue(py::object obj) {
  auto inferred_type = torch::jit::tryToInferType(obj);
  if (inferred_type.success()) {
    return torch::jit::toIValue(obj, inferred_type.type());
  } else {
    try {
      RaggedAny ragged_tensor = obj.cast<RaggedAny>();
      return torch::make_custom_class<k2::RaggedAnyHolder>(
          std::make_shared<RaggedAny>(ragged_tensor));
    } catch (const py::cast_error &) {
      // do nothing.
    }
  }
  return {};
}

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_PYBIND_UTILS_H_
