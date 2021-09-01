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
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/any.h"
#include "k2/python/csrc/torch/v2/doc/any.h"
#include "k2/python/csrc/torch/v2/doc/doc.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

void PybindRaggedAny(py::module &m) {
  py::class_<RaggedAny> any(m, "RaggedTensor");

  //==================================================
  //      k2.ragged.Tensor methods
  //--------------------------------------------------

  any.def(
      py::init([](py::list data,
                  py::object dtype = py::none()) -> std::unique_ptr<RaggedAny> {
        return std::make_unique<RaggedAny>(data, dtype);
      }),
      py::arg("data"), py::arg("dtype") = py::none(), kRaggedAnyInitDataDoc);

  any.def(
      py::init([](const std::string &s,
                  py::object dtype = py::none()) -> std::unique_ptr<RaggedAny> {
        return std::make_unique<RaggedAny>(s, dtype);
      }),
      py::arg("s"), py::arg("dtype") = py::none(), kRaggedAnyInitStrDoc);

  any.def(
      "__str__",
      [](const RaggedAny &self) -> std::string { return self.ToString(); },
      kRaggedAnyStrDoc);

  any.def(
      "__repr__",
      [](const RaggedAny &self) -> std::string { return self.ToString(); },
      kRaggedAnyStrDoc);

  any.def(
      "__getitem__",
      [](RaggedAny &self, int32_t i) -> RaggedAny {
        return self.Index(/*axis*/ 0, i);
      },
      py::arg("i"), kRaggedAnyGetItemDoc);

  any.def("index",
          static_cast<RaggedAny (RaggedAny::*)(RaggedAny &, bool)>(
              &RaggedAny::Index),
          py::arg("indexes"), py::arg("remove_axis") = true);

  any.def("index",
          static_cast<std::pair<RaggedAny, torch::optional<torch::Tensor>> (
              RaggedAny::*)(torch::Tensor, int32_t, bool)>(&RaggedAny::Index),
          py::arg("indexes"), py::arg("axis"),
          py::arg("need_value_indexes") = false);

  any.def(
      "index",
      static_cast<RaggedAny (RaggedAny::*)(torch::Tensor)>(&RaggedAny::Index),
      py::arg("src"));

  any.def("index_and_sum", &RaggedAny::IndexAndSum, py::arg("src"));

  any.def("to",
          static_cast<RaggedAny (RaggedAny::*)(torch::Device) const>(
              &RaggedAny::To),
          py::arg("device"), kRaggedAnyToDeviceDoc);

  any.def("to",
          static_cast<RaggedAny (RaggedAny::*)(torch::ScalarType) const>(
              &RaggedAny::To),
          py::arg("dtype"), kRaggedAnyToDtypeDoc);

  any.def(
      "clone",
      [](const RaggedAny &self) -> RaggedAny {
        DeviceGuard guard(self.any.Context());
        return self.Clone();
      },
      kRaggedAnyCloneDoc);

  any.def(
      "__eq__",
      [](const RaggedAny &self, const RaggedAny &other) -> bool {
        DeviceGuard guard(self.any.Context());
        Dtype t = self.any.GetDtype();
        bool ans = false;
        FOR_REAL_AND_INT32_TYPES(t, T, {
          ans = Equal<T>(self.any.Specialize<T>(), other.any.Specialize<T>());
        });
        return ans;
      },
      py::arg("other"), kRaggedAnyEqDoc);

  any.def(
      "__ne__",
      [](const RaggedAny &self, const RaggedAny &other) -> bool {
        DeviceGuard guard(self.any.Context());
        Dtype t = self.any.GetDtype();
        bool ans = false;
        FOR_REAL_AND_INT32_TYPES(t, T, {
          ans = !Equal<T>(self.any.Specialize<T>(), other.any.Specialize<T>());
        });
        return ans;
      },
      py::arg("other"), kRaggedAnyNeDoc);

  any.def("requires_grad_", &RaggedAny::SetRequiresGrad,
          py::arg("requires_grad") = true, kRaggedAnyRequiresGradMethodDoc);

  any.def("sum", &RaggedAny::Sum, py::arg("initial_value") = 0,
          kRaggedAnySumDoc);

  any.def(
      "numel",
      [](RaggedAny &self) -> int32_t {
        DeviceGuard guard(self.any.Context());
        return self.any.NumElements();
      },
      kRaggedAnyNumelDoc);

  any.def(
      "tot_size",
      [](const RaggedAny &self, int32_t axis) -> int32_t {
        DeviceGuard guard(self.any.Context());
        return self.any.TotSize(axis);
      },
      py::arg("axis"), kRaggedAnyTotSizeDoc);

  any.def(py::pickle(
      [](const RaggedAny &self) -> py::tuple {
        DeviceGuard guard(self.any.Context());
        K2_CHECK(self.any.NumAxes() == 2 || self.any.NumAxes() == 3)
            << "Only support Ragged with NumAxes() == 2 or 3 for now, given "
            << self.any.NumAxes();
        Array1<int32_t> row_splits1 = self.any.RowSplits(1);
        Dtype t = self.any.GetDtype();

        FOR_REAL_AND_INT32_TYPES(t, T, {
          auto values = self.any.Specialize<T>().values;
          // We use "row_ids" placeholder here to make it compatible for the
          // old format file.
          if (self.any.NumAxes() == 2) {
            return py::make_tuple(ToTorch(row_splits1), "row_ids1",
                                  ToTorch(values));
          } else {
            Array1<int32_t> row_splits2 = self.any.RowSplits(2);
            return py::make_tuple(ToTorch(row_splits1), "row_ids1",
                                  ToTorch(row_splits2), "row_ids2",
                                  ToTorch(values));
          }
        });
        // Unreachable code
        return py::none();
      },
      [](const py::tuple &t) -> RaggedAny {
        K2_CHECK(t.size() == 3 || t.size() == 5)
            << "Invalid state. "
            << "Expect a size of 3 or 5. Given: " << t.size();

        torch::Tensor row_splits1_tensor = t[0].cast<torch::Tensor>();
        DeviceGuard guard(GetContext(row_splits1_tensor));
        Array1<int32_t> row_splits1 = FromTorch<int32_t>(row_splits1_tensor);

        RaggedShape shape;
        if (t.size() == 3) {
          auto values_tensor = t[2].cast<torch::Tensor>();
          Dtype t = ScalarTypeToDtype(values_tensor.scalar_type());
          FOR_REAL_AND_INT32_TYPES(t, T, {
            auto values = FromTorch<T>(values_tensor);
            shape = RaggedShape2(&row_splits1, nullptr, values.Dim());
            Ragged<T> any(shape, values);
            return RaggedAny(any.Generic());
          });
        } else if (t.size() == 5) {
          torch::Tensor row_splits2_tensor = t[2].cast<torch::Tensor>();
          Array1<int32_t> row_splits2 = FromTorch<int32_t>(row_splits2_tensor);

          auto values_tensor = t[4].cast<torch::Tensor>();
          Dtype t = ScalarTypeToDtype(values_tensor.scalar_type());

          FOR_REAL_AND_INT32_TYPES(t, T, {
            auto values = FromTorch<T>(values_tensor);
            shape = RaggedShape3(&row_splits1, nullptr, -1, &row_splits2,
                                 nullptr, values.Dim());
            Ragged<T> any(shape, values);
            return RaggedAny(any.Generic());
          });
        } else {
          K2_LOG(FATAL) << "Invalid size : " << t.size();
        }

        // Unreachable code
        return {};
      }));
  SetMethodDoc(&any, "__getstate__", kRaggedAnyGetStateDoc);
  SetMethodDoc(&any, "__setstate__", kRaggedAnySetStateDoc);

  any.def("remove_axis", &RaggedAny::RemoveAxis, py::arg("axis"));
  any.def("arange", &RaggedAny::Arange, py::arg("axis"), py::arg("begin"),
          py::arg("end"));
  any.def("remove_values_leq", &RaggedAny::RemoveValuesLeq, py::arg("cutoff"));
  any.def("remove_values_eq", &RaggedAny::RemoveValuesEq, py::arg("target"));
  any.def("argmax", &RaggedAny::ArgMax, py::arg("initial_value"));
  any.def("max", &RaggedAny::Max, py::arg("initial_value"));
  any.def("min", &RaggedAny::Min, py::arg("initial_value"));

  any.def_static("cat", &RaggedAny::Cat, py::arg("srcs"), py::arg("axis"));
  m.attr("cat") = any.attr("cat");

  any.def("unique", &RaggedAny::Unique, py::arg("need_num_repeats") = false,
          py::arg("need_new2old_indexes") = false);

  any.def("normalize", &RaggedAny::Normalize, py::arg("use_log"));

  any.def("pad", &RaggedAny::Pad, py::arg("mode"), py::arg("padding_value"));
  any.def("tolist", &RaggedAny::ToList);
  any.def("sort", &RaggedAny::Sort, py::arg("descending") = false,
          py::arg("need_new2old_indexes") = false);

  //==================================================
  //      k2.ragged.Tensor properties
  //--------------------------------------------------

  any.def_property_readonly(
      "dtype",
      [](const RaggedAny &self) -> py::object {
        Dtype t = self.any.GetDtype();
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
      },
      kRaggedAnyDtypeDoc);

  any.def_property_readonly(
      "device",
      [](const RaggedAny &self) -> py::object {
        DeviceType d = self.any.Context()->GetDeviceType();
        torch::DeviceType device_type = ToTorchDeviceType(d);

        torch::Device device(device_type, self.any.Context()->GetDeviceId());

        PyObject *ptr = THPDevice_New(device);

        // takes ownership
        return py::reinterpret_steal<py::object>(ptr);
      },
      kRaggedAnyDeviceDoc);

  // Return the underlying memory of this tensor.
  // No data is copied. Memory is shared.
  any.def_property_readonly(
      "data",
      [](RaggedAny &self) -> torch::Tensor {
        Dtype t = self.any.GetDtype();
        FOR_REAL_AND_INT32_TYPES(
            t, T, { return ToTorch(self.any.values.Specialize<T>()); });

        // Unreachable code
        return {};
      },
      kRaggedAnyDataDoc);

  any.def_property_readonly(
      "shape", [](RaggedAny &self) -> RaggedShape { return self.any.shape; },
      "Return the ``Shape`` of this tensor.");

  any.def_property_readonly(
      "grad",
      [](RaggedAny &self) -> torch::optional<torch::Tensor> {
        if (!self.data.defined()) return {};

        return self.Data().grad();
      },
      kRaggedAnyGradPropDoc);

  any.def_property(
      "requires_grad",
      [](RaggedAny &self) -> bool {
        if (!self.data.defined()) return false;

        return self.Data().requires_grad();
      },
      [](RaggedAny &self, bool requires_grad) -> void {
        self.SetRequiresGrad(requires_grad);
      },
      kRaggedAnyRequiresGradPropDoc);

  any.def_property_readonly(
      "is_cuda",
      [](RaggedAny &self) -> bool {
        return self.any.Context()->GetDeviceType() == kCuda;
      },
      kRaggedAnyIsCudaDoc);

  // NumAxes() does not access GPU memory
  any.def_property_readonly(
      "num_axes",
      [](const RaggedAny &self) -> int32_t { return self.any.NumAxes(); },
      kRaggedAnyNumAxesDoc);

  // Dim0() does not access GPU memory
  any.def_property_readonly(
      "dim0", [](const RaggedAny &self) -> int32_t { return self.any.Dim0(); },
      kRaggedAnyDim0Doc);

  //==================================================
  //      _k2.ragged.functions
  //--------------------------------------------------

  // TODO: change the function name from "create_tensor" to "tensor"
  m.def(
      "create_ragged_tensor",
      [](py::list data, py::object dtype = py::none()) -> RaggedAny {
        return RaggedAny(data, dtype);
      },
      py::arg("data"), py::arg("dtype") = py::none(), kRaggedAnyInitDataDoc);

  m.def(
      "create_ragged_tensor",
      [](const std::string &s, py::object dtype = py::none()) -> RaggedAny {
        return RaggedAny(s, dtype);
      },
      py::arg("s"), py::arg("dtype") = py::none(), kRaggedAnyInitStrDoc);
}

}  // namespace k2
