/**
 * @brief Wraps k2 operations
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
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
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/any.h"
#include "k2/python/csrc/torch/v2/autograd/index_select.h"
#include "k2/python/csrc/torch/v2/doc/doc.h"
#include "k2/python/csrc/torch/v2/doc/k2_ops.h"
#include "k2/python/csrc/torch/v2/k2_ops.h"
#include "k2/python/csrc/torch/v2/ops.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

namespace k2 {

void PybindK2Ops(py::module &m) {
  m.def(
      "index_select",
      [](torch::Tensor src, torch::Tensor index,
         float default_value = 0) -> torch::Tensor {
        return IndexSelectFunction::apply(src, index, default_value);
      },
      py::arg("src"), py::arg("index"), py::arg("default_value") = 0,
      kTensorIndexSelectDoc);

  m.def("simple_ragged_index_select", &k2::SimpleRaggedIndexSelect,
        py::arg("src"), py::arg("indexes"), kSimpleRaggedIndexSelectDoc);

  m.def("index_add", &k2::IndexAdd, py::arg("index"), py::arg("value"),
        py::arg("in_out"), kTensorIndexAddDoc);
}

}  // namespace k2
