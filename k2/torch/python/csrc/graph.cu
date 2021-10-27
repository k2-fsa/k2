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
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/graph.h"
#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/doc/graph.h"
#include "k2/torch/python/csrc/graph.h"

namespace k2 {

void PybindGraphs(py::module &m) {
  m.def("ctc_topo", &CtcTopo, py::arg("max_token"), py::arg("modified") = false,
        py::arg("device") = py::none(), kFsaGraphCtcTopoDoc);

  m.def(
      "ctc_graph",
      [](RaggedAny &symbols, bool modified = false,
         torch::optional<torch::Device> device = {}) -> FsaClass {
        return CtcGraphs(symbols, modified);
      },
      py::arg("symbols"), py::arg("modified") = false,
      py::arg("device") = py::none(), kFsaGraphCtcGraphDoc);

  m.def(
      "ctc_graph",
      [](std::vector<std::vector<int32_t>> &symbols, bool modified = false,
         torch::optional<torch::Device> device = {}) -> FsaClass {
        return CtcGraphs(symbols, modified, device);
      },
      py::arg("symbols"), py::arg("modified") = false,
      py::arg("device") = py::none(), kFsaGraphCtcGraphDoc);

  m.def(
      "linear_fsa",
      [](RaggedAny &labels, torch::optional<torch::Device> = {}) -> FsaClass {
        return LinearFsa(labels);
      },
      py::arg("labels"), py::arg("device") = py::none(), kFsaGraphLinearFsaDoc);

  m.def(
      "linear_fsa",
      [](const std::vector<int32_t> &labels,
         torch::optional<torch::Device> device = {}) -> FsaClass {
        return LinearFsa(labels, device);
      },
      py::arg("labels"), py::arg("device") = py::none(), kFsaGraphLinearFsaDoc);

  m.def(
      "linear_fsa",
      [](const std::vector<int32_t> &labels,
         torch::optional<std::string> device = {}) -> FsaClass {
        return LinearFsa(labels, torch::Device(device.value_or("cpu")));
      },
      py::arg("labels"), py::arg("device") = py::none(), kFsaGraphLinearFsaDoc);

  m.def(
      "linear_fsa",
      [](const std::vector<std::vector<int32_t>> &labels,
         torch::optional<torch::Device> device = {}) -> FsaClass {
        return LinearFsa(labels);
      },
      py::arg("labels"), py::arg("device") = py::none(), kFsaGraphLinearFsaDoc);

  m.def(
      "linear_fsa",
      [](const std::vector<std::vector<int32_t>> &labels,
         torch::optional<std::string> device = {}) -> FsaClass {
        return LinearFsa(labels, torch::Device(device.value_or("cpu")));
      },
      py::arg("labels"), py::arg("device") = py::none(), kFsaGraphLinearFsaDoc);

  m.def(
      "levenshtein_graph",
      [](const std::vector<std::vector<int32_t>> &symbols,
         float ins_del_score = -0.501,
         torch::optional<torch::Device> device = {}) -> FsaClass {
        return LevenshteinGraphs(symbols, ins_del_score, device);
      },
      py::arg("symbols"), py::arg("ins_del_score") = -0.501,
      py::arg("device") = py::none(), kFsaGraphLevenshteinGraphDoc);

  m.def(
      "levenshtein_graph",
      [](const std::vector<std::vector<int32_t>> &symbols,
         float ins_del_score = -0.501,
         torch::optional<std::string> device = {}) -> FsaClass {
        return LevenshteinGraphs(symbols, ins_del_score,
                                 torch::Device(device.value_or("cpu")));
      },
      py::arg("symbols"), py::arg("ins_del_score") = -0.501,
      py::arg("device") = py::none(), kFsaGraphLevenshteinGraphDoc);

  m.def(
      "levenshtein_graph",
      [](RaggedAny &symbols, float ins_del_score = -0.501,
         torch::optional<torch::Device> device = {}) -> FsaClass {
        return LevenshteinGraphs(symbols, ins_del_score);
      },
      py::arg("symbols"), py::arg("ins_del_score") = -0.501,
      py::arg("device") = py::none(), kFsaGraphLevenshteinGraphDoc);
}

}  // namespace k2
