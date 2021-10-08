/**
 * @brief Wrap k2 graphs
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
#include "k2/python/csrc/torch/v2/graph.h"
#include "k2/python/csrc/torch/v2/k2_graph.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

namespace k2 {

void PybindK2Graphs(py::module &m) {
  m.def("ctc_topo", &CtcTopo, py::arg("max_token"), py::arg("device"),
        py::arg("modified"));

  m.def("ctc_graph", &CtcGraphs, py::arg("symbols"), py::arg("modified"));
}

}  // namespace k2
