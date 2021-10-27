/**
 * @brief python wrapper for Ragged<Arc>
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang,
 *                                              Wei Kang)
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
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/fsa.h"
#include "k2/torch/python/csrc/pybind_utils.h"

namespace k2 {

void PybindFsaClass(py::module &m) {
  // TODO(fangjun): Add more doc to doc/fsa.cu for k2.ragged.FSA
  py::class_<FsaClass> fsa(m, "Fsa");

  fsa.def(py::init([](const std::string &s,
                      torch::optional<std::vector<std::string>>
                          extra_label_names = {}) -> std::unique_ptr<FsaClass> {
            return std::make_unique<FsaClass>(
                s, extra_label_names.value_or(std::vector<std::string>{}));
          }),
          py::arg("s"), py::arg("extra_label_names") = py::none());

  fsa.def("__str__", &FsaClass::ToString);
  fsa.def("__repr__", &FsaClass::ToString);

  fsa.def("requires_grad_", &FsaClass::SetRequiresGrad,
          py::arg("requires_grad") = true);

  fsa.def(
      "to",
      static_cast<FsaClass (FsaClass::*)(torch::Device) const>(&FsaClass::To),
      py::arg("device"));

  fsa.def("to",
          static_cast<FsaClass (FsaClass::*)(const std::string &) const>(
              &FsaClass::To),
          py::arg("device"));

  fsa.def(
      "set_scores_stochastic_",
      [](FsaClass &self, torch::Tensor scores) -> void {
        self.SetScoresStochastic(scores);
      },
      py::arg("scores"));

  fsa.def("arc_sort", &FsaClass::ArcSort);
  fsa.def("to_str_simple", &FsaClass::ToStringSimple);

  fsa.def(
      "__getitem__",
      [](FsaClass &self, int32_t i) -> FsaClass {
        K2_CHECK_EQ(self.fsa.NumAxes(), 3);
        return self.Index(i);
      },
      py::arg("i"));

  fsa.def(
      "__setattr__",
      [](FsaClass &self, const std::string &name, py::object value) -> void {
        self.SetAttr(name, ToIValue(value));
      });

  fsa.def("__getattr__",
          [](FsaClass &self, const std::string &name) -> py::object {
            if (self.HasAttr(name)) {
              return ToPyObject(self.GetAttr(name));
            } else {
              std::ostringstream os;
              os << "No such attribute '" << name << "'";
              PyErr_SetString(PyExc_AttributeError, os.str().c_str());
              throw py::error_already_set();
            }
          });

  fsa.def("__delattr__", [](FsaClass &self, const std::string &name) -> void {
    if (self.HasAttr(name)) {
      self.DeleteAttr(name);
    } else {
      std::ostringstream os;
      os << "No such attribute '" << name << "'";
      PyErr_SetString(PyExc_AttributeError, os.str().c_str());
      throw py::error_already_set();
    }
  });

  fsa.def("get_forward_scores", &FsaClass::GetForwardScores,
          py::arg("use_double_scores"), py::arg("log_semiring"));

  fsa.def_static("from_fsas", &FsaClass::CreateFsaVec, py::arg("fsas"));

  fsa.def_property_readonly(
      "properties", [](FsaClass &self) -> int { return self.Properties(); });

  fsa.def_property_readonly("device", [](const FsaClass &self) -> py::object {
    torch::Device device = GetDevice(self.fsa.Context());
    PyObject *ptr = THPDevice_New(device);
    // takes ownership
    return py::reinterpret_steal<py::object>(ptr);
  });

  fsa.def_property_readonly("shape", [](FsaClass &self) -> py::tuple {
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

  fsa.def_property_readonly("raw_shape", [](FsaClass &self) -> RaggedShape {
    return self.fsa.shape;
  });

  fsa.def_property(
      "requires_grad",
      [](FsaClass &self) -> bool {
        if (!self.scores.defined()) return false;

        return self.Scores().requires_grad();
      },
      [](FsaClass &self, bool requires_grad) -> void {
        self.SetRequiresGrad(requires_grad);
      });

  fsa.def_property_readonly(
      "arcs", &FsaClass::Arcs,
      "You should not modify the data of the returned arcs!");
}  // namespace k2

}  // namespace k2
