/**
 * @brief Everything related to PyTorch for k2 Python wrappers.
 *
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_PYTHON_CSRC_TORCH_H_
#define K2_PYTHON_CSRC_TORCH_H_

#include "k2/csrc/log.h"
#include "k2/python/csrc/torch.h"
#include "torch/extension.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<torch::ScalarType> {
 public:
  PYBIND11_TYPE_CASTER(torch::ScalarType, _("torch::dtype"));

  bool load(handle src, bool) {
    PyObject *obj = src.ptr();
    if (THPDtype_Check(obj)) {
      value = reinterpret_cast<THPDtype *>(obj)->scalar_type;
      return true;
    }
    return false;
  }

  static handle cast(const torch::ScalarType &src,
                     return_value_policy /* policy */, handle /* parent */) {
    auto torch = py::module::import("torch");
    py::object ans;
    switch (src) {
      case torch::kFloat32:
        ans = torch.attr("float32");
        break;
      case torch::kFloat64:
        ans = torch.attr("float64");
        break;
      case torch::kInt32:
        ans = torch.attr("int32");
        break;
      default:
        K2_LOG(FATAL) << "Unsupported scalar type: " << src;
        break;
    }
    return handle(ans.release());
  }
};

}  // namespace detail
}  // namespace pybind11

void PybindTorch(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_H_
