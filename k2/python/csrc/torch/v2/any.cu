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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/csrc/torch_util.h"
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

  any.def(py::init([](py::list data, py::object dtype = py::none(),
                      py::object device =
                          py::str("cpu")) -> std::unique_ptr<RaggedAny> {
            std::string device_str = device.is_none() ? "cpu" : py::str(device);
            return std::make_unique<RaggedAny>(data, dtype,
                                               torch::Device(device_str));
          }),
          py::arg("data"), py::arg("dtype") = py::none(),
          py::arg("device") = py::str("cpu"), kRaggedAnyInitDataDeviceDoc);

  any.def(py::init<py::list, py::object, const std::string &>(),
          py::arg("data"), py::arg("dtype") = py::none(),
          py::arg("device") = "cpu", kRaggedAnyInitDataDeviceDoc);

  any.def(py::init([](const std::string &s, py::object dtype = py::none(),
                      py::object device =
                          py::str("cpu")) -> std::unique_ptr<RaggedAny> {
            std::string device_str = device.is_none() ? "cpu" : py::str(device);
            return std::make_unique<RaggedAny>(s, dtype, device_str);
          }),
          py::arg("s"), py::arg("dtype") = py::none(),
          py::arg("device") = py::str("cpu"), kRaggedAnyInitStrDeviceDoc);

  any.def(py::init<const std::string &, py::object, const std::string &>(),
          py::arg("s"), py::arg("dtype") = py::none(),
          py::arg("device") = "cpu", kRaggedAnyInitStrDeviceDoc);

  any.def(py::init<const RaggedShape &, torch::Tensor>(), py::arg("shape"),
          py::arg("value"), kRaggedInitFromShapeAndTensorDoc);

  any.def(py::init<torch::Tensor>(), py::arg("tensor"),
          kRaggedAnyInitTensorDoc);

  any.def(
      "__str__",
      [](const RaggedAny &self) -> std::string { return self.ToString(); },
      kRaggedAnyStrDoc);

  any.def(
      "to_str_simple",
      [](const RaggedAny &self) -> std::string {
        return self.ToString(/*compact*/ true);
      },
      kRaggedAnyToStrSimpleDoc);

  any.def(
      "__repr__",
      [](const RaggedAny &self) -> std::string { return self.ToString(); },
      kRaggedAnyStrDoc);

  any.def(
      "__getitem__",
      [](RaggedAny &self, int32_t i) -> py::object {
        if (self.any.NumAxes() > 2) {
          RaggedAny ragged = self.Index(/*axis*/ 0, i);
          return py::cast(ragged);
        } else {
          DeviceGuard guard(self.any.Context());
          K2_CHECK_EQ(self.any.NumAxes(), 2);
          Array1<int32_t> row_split = self.any.RowSplits(1).To(GetCpuContext());
          const int32_t *row_split_data = row_split.Data();
          int32_t begin = row_split_data[i], end = row_split_data[i + 1];
          Dtype t = self.any.GetDtype();
          FOR_REAL_AND_INT32_TYPES(t, T, {
            Array1<T> array =
                self.any.Specialize<T>().values.Arange(begin, end);
            torch::Tensor tensor = ToTorch(array);
            return py::cast(tensor);
          });
        }
        // Unreachable code
        return py::none();
      },
      py::arg("i"), kRaggedAnyGetItemDoc);

  any.def(
      "__getitem__",
      [](RaggedAny &self, const py::slice &slice) -> RaggedAny {
        py::size_t start = 0, stop = 0, step = 0, slicelength = 0;
        if (!slice.compute(self.any.Dim0(), &start, &stop, &step, &slicelength))
          throw py::error_already_set();
        int32_t istart = static_cast<int32_t>(start);
        int32_t istop = static_cast<int32_t>(stop);
        int32_t istep = static_cast<int32_t>(step);
        K2_CHECK_EQ(istep, 1)
            << "Only support slicing with step 1, given : " << istep;

        return self.Arange(/*axis*/ 0, istart, istop);
      },
      py::arg("key"), kRaggedAnyGetItemSliceDoc);

  any.def(
      "__getitem__",
      [](RaggedAny &self, torch::Tensor key) -> RaggedAny {
        // key is a 1-d torch tensor with dtype torch.int32
        DeviceGuard guard(self.any.Context());
        Array1<int32_t> indexes = FromTorch<int32_t>(key);
        Dtype t = self.any.GetDtype();
        FOR_REAL_AND_INT32_TYPES(t, T, {
          Ragged<T> ans =
              k2::Index<T>(self.any.Specialize<T>(), /*axis*/ 0, indexes,
                           /*value_indexes*/ nullptr);

          return RaggedAny(ans.Generic());
        });
        // Unreachable code
        return {};
      },
      py::arg("key"), kRaggedAnyGetItem1DTensorDoc);

  any.def("index",
          static_cast<RaggedAny (RaggedAny::*)(RaggedAny &)>(&RaggedAny::Index),
          py::arg("indexes"), kRaggedAnyRaggedIndexDoc);

  any.def("index",
          static_cast<std::pair<RaggedAny, torch::optional<torch::Tensor>> (
              RaggedAny::*)(torch::Tensor, int32_t, bool)>(&RaggedAny::Index),
          py::arg("indexes"), py::arg("axis"),
          py::arg("need_value_indexes") = false, kRaggedAnyTensorIndexDoc);

  m.def(
      "index",
      [](torch::Tensor src, RaggedAny &indexes,
         py::object default_value = py::none()) -> RaggedAny {
        return indexes.Index(src, default_value);
      },
      py::arg("src"), py::arg("indexes"), py::arg("default_value") = py::none(),
      kRaggedAnyIndexTensorWithRaggedDoc);

  m.def(
      "index_and_sum",
      [](torch::Tensor src, RaggedAny &indexes) -> torch::Tensor {
        return indexes.IndexAndSum(src);
      },
      py::arg("src"), py::arg("indexes"), kRaggedAnyIndexAndSumDoc);

  any.def(
      "to",
      [](RaggedAny &self, py::object device) -> RaggedAny {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        return self.To(torch::Device(device_str));
      },
      py::arg("device"), kRaggedAnyToDeviceDoc);

  any.def("to",
          static_cast<RaggedAny (RaggedAny::*)(const std::string &) const>(
              &RaggedAny::To),
          py::arg("device"), kRaggedAnyToDeviceStrDoc);

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

  any.def("logsumexp", &RaggedAny::LogSumExp,
          py::arg("initial_value") = -std::numeric_limits<float>::infinity(),
          kRaggedAnyLogSumExpDoc);

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
            << "Only support Ragged with NumAxes() == 2 or 3 for now, "
               "given "
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

  any.def("remove_axis", &RaggedAny::RemoveAxis, py::arg("axis"),
          kRaggedAnyRemoveAxisDoc);

  any.def("arange", &RaggedAny::Arange, py::arg("axis"), py::arg("begin"),
          py::arg("end"), kRaggedAnyArangeDoc);

  any.def("remove_values_leq", &RaggedAny::RemoveValuesLeq, py::arg("cutoff"),
          kRaggedAnyRemoveValuesLeqDoc);

  any.def("remove_values_eq", &RaggedAny::RemoveValuesEq, py::arg("target"),
          kRaggedAnyRemoveValuesEqDoc);

  any.def("argmax", &RaggedAny::ArgMax, py::arg("initial_value") = py::none(),
          kRaggedAnyArgMaxDoc);

  any.def("max", &RaggedAny::Max, py::arg("initial_value") = py::none(),
          kRaggedAnyMaxDoc);

  any.def("min", &RaggedAny::Min, py::arg("initial_value") = py::none(),
          kRaggedAnyMinDoc);

  any.def_static("cat", &RaggedAny::Cat, py::arg("srcs"), py::arg("axis"),
                 kRaggedCatDoc);
  m.attr("cat") = any.attr("cat");

  any.def("unique", &RaggedAny::Unique, py::arg("need_num_repeats") = false,
          py::arg("need_new2old_indexes") = false, kRaggedAnyUniqueDoc);

  any.def("normalize", &RaggedAny::Normalize, py::arg("use_log"),
          kRaggedAnyNormalizeDoc);

  any.def("add", &RaggedAny::Add, py::arg("value"), py::arg("alpha"),
          kRaggedAnyAddDoc);

  any.def("pad", &RaggedAny::Pad, py::arg("mode"), py::arg("padding_value"),
          kRaggedAnyPadDoc);

  any.def("tolist", &RaggedAny::ToList, kRaggedAnyToListDoc);

  any.def("sort_", &RaggedAny::Sort, py::arg("descending") = false,
          py::arg("need_new2old_indexes") = false, kRaggedAnySortDoc);

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

        auto torch_device = py::module::import("torch").attr("device");
        return torch_device(device.str());
      },
      kRaggedAnyDeviceDoc);

  // Return the underlying memory of this tensor.
  // No data is copied. Memory is shared.
  any.def_property_readonly(
      "values", [](RaggedAny &self) -> torch::Tensor { return self.Data(); },
      kRaggedAnyValuesDoc);

  any.def_property_readonly(
      "shape", [](RaggedAny &self) -> RaggedShape { return self.any.shape; },
      kRaggedAnyShapeDoc);

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

  m.def(
      "create_ragged_tensor",
      [](py::list data, py::object dtype = py::none(),
         py::object device = py::str("cpu")) -> RaggedAny {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        return RaggedAny(data, dtype, torch::Device(device_str));
      },
      py::arg("data"), py::arg("dtype") = py::none(),
      py::arg("device") = py::str("cpu"), kCreateRaggedTensorDataDoc);

  m.def(
      "create_ragged_tensor",
      [](py::list data, py::object dtype = py::none(),
         const std::string &device = "cpu") -> RaggedAny {
        return RaggedAny(data, dtype, device);
      },
      py::arg("data"), py::arg("dtype") = py::none(), py::arg("device") = "cpu",
      kCreateRaggedTensorDataDoc);

  m.def(
      "create_ragged_tensor",
      [](const std::string &s, py::object dtype = py::none(),
         py::object device = py::str("cpu")) -> RaggedAny {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        return RaggedAny(s, dtype, torch::Device(device_str));
      },
      py::arg("s"), py::arg("dtype") = py::none(),
      py::arg("device") = py::str("cpu"), kCreateRaggedTensorStrDoc);

  m.def(
      "create_ragged_tensor",
      [](const std::string &s, py::object dtype = py::none(),
         const std::string &device = "cpu") -> RaggedAny {
        return RaggedAny(s, dtype, device);
      },
      py::arg("s"), py::arg("dtype") = py::none(), py::arg("device") = "cpu",
      kCreateRaggedTensorStrDoc);

  m.def(
      "create_ragged_tensor",
      [](torch::Tensor tensor) -> RaggedAny { return RaggedAny(tensor); },
      py::arg("tensor"), kCreateRaggedTensorTensorDoc);
}

}  // namespace k2
