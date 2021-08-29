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

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/v2/fsa.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

namespace k2 {

void PybindRaggedArc(py::module &m) {
  py::class_<RaggedArc> fsa(m, "Fsa");
  fsa.def(py::init<>());

  fsa.def(py::init<const std::string &, py::list>(), py::arg("s"),
          py::arg("extra_label_names") = py::none());
  fsa.def("__str__", &RaggedArc::ToString);
  fsa.def("__repr__", &RaggedArc::ToString);

  fsa.def("requires_grad_", &RaggedArc::SetRequiresGrad,
          py::arg("requires_grad") = true);

  fsa.def("arc_sort", &RaggedArc::ArcSort);

  fsa.def_property(
      "scores", [](RaggedArc &self) -> torch::Tensor { return self.Scores(); },
      [](RaggedArc &self, torch::Tensor scores) {
        self.Scores().copy_(scores);
      });

  fsa.def_property_readonly(
      "grad", [](RaggedArc &self) -> torch::optional<torch::Tensor> {
        if (!self.scores.defined()) return {};

        return self.Scores().grad();
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
}

}  // namespace k2
