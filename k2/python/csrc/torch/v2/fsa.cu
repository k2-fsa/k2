/**
 * @brief python wrapper for Ragged<Arc>
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

#include <memory>
#include <string>
#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/doc/fsa.h"
#include "k2/python/csrc/torch/v2/fsa.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

namespace k2 {

void PybindRaggedArc(py::module &m) {
  // TODO(fangjun): Add more doc to doc/fsa.cu for k2.ragged.FSA
  py::class_<RaggedArc> fsa(m, "Fsa");

  fsa.def(
      py::init([](const std::string &s,
                  torch::optional<std::vector<std::string>> extra_label_names =
                      {}) -> std::unique_ptr<RaggedArc> {
        return std::make_unique<RaggedArc>(
            s, extra_label_names.value_or(std::vector<std::string>{}));
      }),
      py::arg("s"), py::arg("extra_label_names") = py::none());

  fsa.def("__str__", &RaggedArc::ToString);
  fsa.def("__repr__", &RaggedArc::ToString);
  fsa.def("to_str", &RaggedArc::ToString);
  fsa.def("to_str_simple", &RaggedArc::ToStringSimple);

  fsa.def("requires_grad_", &RaggedArc::SetRequiresGrad,
          py::arg("requires_grad") = true);

  fsa.def("to",
          static_cast<RaggedArc (RaggedArc::*)(torch::Device) const>(
              &RaggedArc::To),
          py::arg("device"));

  fsa.def("to",
          static_cast<RaggedArc (RaggedArc::*)(const std::string &) const>(
              &RaggedArc::To),
          py::arg("device"));

  fsa.def("arc_sort", &RaggedArc::ArcSort, kFsaAlgoArcSortDoc);
  fsa.def("connect", &RaggedArc::Connect, kFsaAlgoConnectDoc);
  fsa.def("top_sort", &RaggedArc::TopSort, kFsaAlgoTopSortDoc);
  fsa.def("add_epsilon_self_loops", &RaggedArc::AddEpsilonSelfLoops,
          kFsaAlgoAddEpsilonSelfLoopsDoc);

  fsa.def("__setattr__", (void (RaggedArc::*)(const std::string &, py::object))(
                             &RaggedArc::SetAttr));
  fsa.def("__getattr__", &RaggedArc::GetAttr);
  fsa.def("__delattr__", &RaggedArc::DeleteAttr);

  fsa.def("get_forward_scores", &RaggedArc::GetForwardScores,
          py::arg("use_double_scores"), py::arg("log_semiring"));

  fsa.def_static("from_fsas", &RaggedArc::CreateFsaVec, py::arg("fsas"));

  fsa.def_property(
      "scores", [](RaggedArc &self) -> torch::Tensor { return self.Scores(); },
      [](RaggedArc &self, torch::Tensor scores) { self.SetScores(scores); });

  fsa.def_property_readonly(
      "properties", [](RaggedArc &self) -> int { return self.Properties(); });

  fsa.def_property_readonly("shape", [](RaggedArc &self) -> py::tuple {
    if (self.fsa.NumAxes() == 2) {
      py::tuple ans(2);
      ans[0] = self.fsa.Dim0();
      ans[1] = py::none();
      return ans;
    } else {
      py::tuple ans(3);
      ans[0] = self.fsa.Dim0();
      ans[1] = py::none();
      ans[2] = py::none();
      return ans;
    }
  });

  fsa.def_property(
      "requires_grad",
      [](RaggedArc &self) -> bool {
        if (!self.scores.defined()) return false;

        return self.Scores().requires_grad();
      },
      [](RaggedArc &self, bool requires_grad) -> void {
        self.SetRequiresGrad(requires_grad);
      });

  fsa.def_property_readonly(
      "arcs", &RaggedArc::Arcs,
      "You should not modify the data of the returned arcs!");
}

}  // namespace k2
