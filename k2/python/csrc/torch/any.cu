/**
 * @brief Wraps Ragged<Any>
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
#include "k2/python/csrc/torch/any.h"
#include "k2/python/csrc/torch/ragged_any.h"
#include "k2/python/csrc/torch/torch_util.h"

namespace k2 {

void PybindRaggedAny(py::module &m) {
  py::module ragged = m.def_submodule(
      "ragged", "Sub module containing operations for ragged tensors in k2");

  py::class_<RaggedAny> any(ragged, "Tensor");

  //==================================================
  //      k2.ragged.Tensor methods
  //--------------------------------------------------
  any.def(py::init<>());

  any.def(
      py::init([](py::list data,
                  py::object dtype = py::none()) -> std::unique_ptr<RaggedAny> {
        return std::make_unique<RaggedAny>(data, dtype);
      }),
      py::arg("data"), py::arg("dtype") = py::none());

  any.def("__str__",
          [](const RaggedAny &self) -> std::string { return self.ToString(); });

  any.def("__repr__",
          [](const RaggedAny &self) -> std::string { return self.ToString(); });

  // o is either torch.device or torch.dtype
  any.def("to", [](const RaggedAny &self, py::object o) -> RaggedAny {
    PyObject *ptr = o.ptr();
    if (THPDevice_Check(ptr)) {
      torch::Device device = reinterpret_cast<THPDevice *>(ptr)->device;
      return self.To(device);
    }

    if (THPDtype_Check(ptr)) {
      auto scalar_type = reinterpret_cast<THPDtype *>(ptr)->scalar_type;
      return self.To(scalar_type);
    }

    K2_LOG(FATAL)
        << "Expect an instance of torch.device or torch.dtype. Given: "
        << py::str(o);

    // Unreachable code
    return {};
  });

  any.def("clone", [](const RaggedAny &self) -> RaggedAny {
    DeviceGuard guard(self.any_.Context());
    return self.Clone();
  });

  any.def(
      "__eq__",
      [](const RaggedAny &self, const RaggedAny &other) -> bool {
        DeviceGuard guard(self.any_.Context());
        Dtype t = self.any_.GetDtype();
        bool ans = false;
        FOR_REAL_AND_INT32_TYPES(t, T, {
          ans = Equal<T>(self.any_.Specialize<T>(), other.any_.Specialize<T>());
        });
        return ans;
      },
      py::arg("other"));

  any.def(
      "__ne__",
      [](const RaggedAny &self, const RaggedAny &other) -> bool {
        DeviceGuard guard(self.any_.Context());
        Dtype t = self.any_.GetDtype();
        bool ans = false;
        FOR_REAL_AND_INT32_TYPES(t, T, {
          ans =
              !Equal<T>(self.any_.Specialize<T>(), other.any_.Specialize<T>());
        });
        return ans;
      },
      py::arg("other"));

  any.def("requires_grad_", &RaggedAny::SetRequiresGrad,
          py::arg("requires_grad") = true);

  any.def("sum", &RaggedAny::Sum, py::arg("initial_value") = 0);

  //==================================================
  //      k2.ragged.Tensor properties
  //--------------------------------------------------

  any.def_property_readonly("dtype", [](const RaggedAny &self) -> py::object {
    Dtype t = self.any_.GetDtype();
    auto torch = py::module::import("torch");
    switch (t) {
      case kFloatDtype:
        return torch.attr("float32");
      case kDoubleDtype:
        return torch.attr("float64");
      case kInt32Dtype:
        return torch.attr("int32");
      default:
        K2_LOG(FATAL) << "Unsupported dtype: " << TraitsOf(t).Name();
    }

    // Unreachable code
    return py::none();
  });

  any.def_property_readonly("device", [](const RaggedAny &self) -> py::object {
    DeviceType d = self.any_.Context()->GetDeviceType();
    torch::DeviceType device_type = ToTorchDeviceType(d);

    torch::Device device(device_type, self.any_.Context()->GetDeviceId());

    PyObject *ptr = THPDevice_New(device);
    py::handle h(ptr);

    // takes ownership
    return py::reinterpret_steal<py::object>(h);
  });

  // Return the underlying memory of this tensor.
  // No data is copied. Memory is shared.
  any.def_property_readonly("data", [](RaggedAny &self) -> torch::Tensor {
    Dtype t = self.any_.GetDtype();
    FOR_REAL_AND_INT32_TYPES(
        t, T, { return ToTorch(self.any_.values.Specialize<T>()); });

    // Unreachable code
    return {};
  });

  any.def_property_readonly(
      "grad", [](RaggedAny &self) -> torch::optional<torch::Tensor> {
        if (!self.data_.defined()) return {};

        return self.Data().grad();
      });

  any.def_property(
      "requires_grad",
      [](RaggedAny &self) -> bool {
        if (!self.data_.defined()) return false;

        return self.Data().requires_grad();
      },
      [](RaggedAny &self, bool requires_grad) -> void {
        self.SetRequiresGrad(requires_grad);
      });

  //==================================================
  //      _k2.ragged.functions
  //--------------------------------------------------

  ragged.def(
      "create_tensor",
      [](py::list data, py::object dtype = py::none()) -> RaggedAny {
        return RaggedAny(data, dtype);
      },
      py::arg("data"), py::arg("dtype") = py::none());
}

}  // namespace k2
