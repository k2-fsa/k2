/**
 * @brief Wraps Ragged<Any>
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang
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
#include <utility>
#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/any.h"
#include "k2/torch/python/csrc/doc/any.h"
#include "k2/torch/python/csrc/doc/doc.h"
#include "k2/torch/python/csrc/pybind_utils.h"

namespace k2 {

/** One iteration of RaggedAnyFromList.

  @param data It is a list or a list-of sublist(s).
  @param cur_level It is the level of a sublist. The root has a level 0.
  @param deepest_level values appear at this level.
  @param row_splits  It contains row_splits of different levels, indexed
                     by `cur_level`.
  @param elems  It contains the elements read so far.
 */
template <typename T>
static void RaggedAnyFromListIter(py::list data, int32_t *cur_level,
                                  int32_t *deepest_level,
                                  std::vector<std::vector<int32_t>> *row_splits,
                                  std::vector<T> *elems) {
  // We encounter a new sublist, so increase the level number
  *cur_level += 1;

  if (static_cast<size_t>(*cur_level) > row_splits->size()) {
    // This is a deeper level that has not been seen, so
    // we need to allocate a row_split for this level
    row_splits->resize(*cur_level, std::vector<int32_t>(1, 0));
  }

  if (data.size() > 0 && py::isinstance<py::list>(data[0])) {
    // If `data` is not empty and it contains sublist
    for (auto &d : data) {
      if (!py::isinstance<py::list>(d)) {
        throw std::runtime_error("Expect an instance of list");
      }
      RaggedAnyFromListIter(d.cast<py::list>(), cur_level, deepest_level,
                            row_splits, elems);
    }
  } else {
    if (*deepest_level == -1) {
      *deepest_level = *cur_level;
    } else if (data.size() > 0 && *deepest_level != *cur_level) {
      // Handle the case for [ [2], [[1]] ]
      //
      // Note: [ [], [[1]] ] is valid
      throw std::runtime_error("Make sure sublists are properly nested");
    }

    if (data.size() > 0 &&
        static_cast<size_t>(*cur_level) != row_splits->size()) {
      // Handle cases like the following string:
      // [ [[1]], [2, 3] ]
      // The sublist [2, 3] should be [[2, 3]], i.e., has the same
      // level as [[1]]
      //
      // Note: [ [], [[1]] ] is valid
      throw std::runtime_error("Expect a [");
    }

    auto tmp = data.cast<std::vector<T>>();
    elems->insert(elems->end(), tmp.begin(), tmp.end());
  }

  *cur_level -= 1;

  (*row_splits)[*cur_level].push_back(
      (*cur_level + 1 >= (int32_t)row_splits->size())
          ? static_cast<int32_t>(elems->size())
          : ((*row_splits)[*cur_level + 1].size() - 1));
}

/** Construct a Ragged<T> from a list of sublist(s) of integers
   or real numbers.

  @param data  A list of sublist(s).
  @return Return a Ragged<T> constructed from the given `data`.
 */
template <typename T>
static Ragged<T> RaggedAnyFromList(py::list data) {
  std::vector<std::vector<int32_t>> row_splits;
  std::vector<T> elems;
  int32_t cur_level = 0;
  int32_t deepest_level = -1;  // values appear at this level
  for (auto &d : data) {
    if (!py::isinstance<py::list>(d)) {
      throw std::runtime_error("Expect a list");
    }
    RaggedAnyFromListIter(d.cast<py::list>(), &cur_level, &deepest_level,
                          &row_splits, &elems);
  }

  if (row_splits.empty()) {
    // Assume 2 axes even though the num-axes is ambiguous from the input `[ ]`
    // row_splits is [ 0 ].
    row_splits.push_back(std::vector<int32_t>(1, 0));
  }

  std::vector<RaggedShapeLayer> axes(row_splits.size());
  ContextPtr c = GetCpuContext();
  for (size_t i = 0; i != row_splits.size(); ++i) {
    axes[i].row_splits = Array1<int32_t>(c, row_splits[i]);
    axes[i].cached_tot_size = row_splits[i].back();
  }
  Ragged<T> ans;
  ans.shape = RaggedShape(axes);
  ans.values = Array1<T>(c, elems);
  if (ans.values.Dim() != ans.shape.NumElements()) {
    throw std::runtime_error("Invalid format of a ragged tensor");
  }
  return ans;
}

static RaggedAny RaggedAnyFromList(py::list data, py::object dtype = py::none(),
                                   torch::Device device = torch::kCPU) {
  if (!dtype.is_none() && !THPDtype_Check(dtype.ptr())) {
    K2_LOG(FATAL) << "Expect an instance of torch.dtype. "
                  << "Given: " << py::str(dtype);
  }

  ContextPtr context = GetContext(device);
  DeviceGuard guard(context);

  if (dtype.is_none()) {
    try {
      // We try int first; if it fails, use float
      auto any = RaggedAnyFromList<int32_t>(data).To(context).Generic();
      return RaggedAny(any);
    } catch (const std::exception &) {
      // Use float. If it fails again, another exception
      // is thrown and it is propagated to the user
      auto any = RaggedAnyFromList<float>(data).To(context).Generic();
      return RaggedAny(any);
    }
  }

  auto scalar_type = reinterpret_cast<THPDtype *>(dtype.ptr())->scalar_type;

  Dtype t = ScalarTypeToDtype(scalar_type);

  FOR_REAL_AND_INT32_TYPES(t, T, {
    auto any = RaggedAnyFromList<T>(data).To(context).Generic();
    return RaggedAny(any);
  });

  K2_LOG(FATAL) << "Unsupported dtype: " << scalar_type
                << ". Supported dtypes are: torch.int32, torch.float32, "
                << "and torch.float64";
  // Unreachable code
  return RaggedAny();
}

template <typename T>
static py::list ToList(Ragged<T> &src, int32_t axis, int32_t begin,
                       int32_t end) {
  // assuming src is on CPU
  int32_t num_axes = src.NumAxes();

  K2_CHECK_GE(axis, 0);
  K2_CHECK_LT(axis, num_axes);
  K2_CHECK_LE(begin, end);
  K2_CHECK_LE(end, src.TotSize(axis));

  py::list ans(end - begin);
  if (axis == num_axes - 1) {
    const T *data = src.values.Data();
    // recursion ends here
    for (int32_t i = begin; i != end; ++i) {
      ans[i - begin] = data[i];
    }
  } else {
    const int32_t *data = src.RowSplits(axis + 1).Data();
    for (int32_t i = begin; i != end; ++i) {
      ans[i - begin] = ToList(src, axis + 1, data[i], data[i + 1]);
    }
  }
  return ans;
}

static py::list RaggedAnyToList(RaggedAny &src) {
  Dtype t = src.any.GetDtype();
  FOR_REAL_AND_INT32_TYPES(t, T, {
    return ToList(src.any.Specialize<T>(), /*axis*/ 0, /*begin*/ 0,
                  /*end*/ src.any.Dim0());
  });
  // Unreachable code
  return py::none();
}

void PybindRaggedAny(py::module &m) {
  py::class_<RaggedAny> any(m, "RaggedTensor");

  //==================================================
  //      k2.ragged.Tensor methods
  //--------------------------------------------------

  any.def(py::init([](py::list data, py::object dtype = py::none(),
                      torch::Device device = torch::Device(torch::kCPU)) {
            auto ragged_any = RaggedAnyFromList(data, dtype, device);
            return std::make_unique<RaggedAny>(ragged_any);
          }),
          py::arg("data"), py::arg("dtype") = py::none(),
          py::arg("device") = torch::Device(torch::kCPU),
          kRaggedAnyInitDataDeviceDoc);

  any.def(py::init([](py::list data, py::object dtype = py::none(),
                      const std::string &device = "cpu") {
            auto ragged_any =
                RaggedAnyFromList(data, dtype, torch::Device(device));
            return std::make_unique<RaggedAny>(ragged_any);
          }),
          py::arg("data"), py::arg("dtype") = py::none(),
          py::arg("device") = "cpu", kRaggedAnyInitDataDeviceDoc);

  any.def(py::init<const std::string &, torch::optional<torch::ScalarType>,
                   torch::Device>(),
          py::arg("s"), py::arg("dtype") = torch::optional<torch::ScalarType>(),
          py::arg("device") = torch::Device(torch::kCPU),
          kRaggedAnyInitStrDeviceDoc);

  any.def(py::init<const std::string &, torch::optional<torch::ScalarType>,
                   const std::string &>(),
          py::arg("s"), py::arg("dtype") = torch::optional<torch::ScalarType>(),
          py::arg("device") = torch::Device(torch::kCPU),
          kRaggedAnyInitStrDeviceDoc);

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
        py::ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
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
        return indexes.Index(src, ToIValue(default_value));
      },
      py::arg("src"), py::arg("indexes"), py::arg("default_value") = py::none(),
      kRaggedAnyIndexTensorWithRaggedDoc);

  m.def(
      "index_and_sum",
      [](torch::Tensor src, RaggedAny &indexes) -> torch::Tensor {
        return indexes.IndexAndSum(src);
      },
      py::arg("src"), py::arg("indexes"), kRaggedAnyIndexAndSumDoc);

  any.def("to",
          static_cast<RaggedAny (RaggedAny::*)(torch::Device) const>(
              &RaggedAny::To),
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

  any.def("remove_axis", &RaggedAny::RemoveAxis, py::arg("axis"),
          kRaggedAnyRemoveAxisDoc);

  any.def("arange", &RaggedAny::Arange, py::arg("axis"), py::arg("begin"),
          py::arg("end"), kRaggedAnyArangeDoc);

  any.def(
      "remove_values_leq",
      [](RaggedAny &self, py::object cutoff) -> RaggedAny {
        return self.RemoveValuesLeq(ToIValue(cutoff));
      },
      py::arg("cutoff"), kRaggedAnyRemoveValuesLeqDoc);

  any.def(
      "remove_values_eq",
      [](RaggedAny &self, py::object target) -> RaggedAny {
        return self.RemoveValuesEq(ToIValue(target));
      },
      py::arg("target"), kRaggedAnyRemoveValuesEqDoc);

  any.def(
      "argmax",
      [](RaggedAny &self, py::object initial_value = py::none())
          -> torch::Tensor { return self.ArgMax(ToIValue(initial_value)); },
      py::arg("initial_value") = py::none(), kRaggedAnyArgMaxDoc);

  any.def(
      "max",
      [](RaggedAny &self, py::object initial_value = py::none())
          -> torch::Tensor { return self.Max(ToIValue(initial_value)); },
      py::arg("initial_value") = py::none(), kRaggedAnyMaxDoc);

  any.def(
      "min",
      [](RaggedAny &self, py::object initial_value = py::none())
          -> torch::Tensor { return self.Min(ToIValue(initial_value)); },
      py::arg("initial_value") = py::none(), kRaggedAnyMinDoc);

  any.def_static("cat", &RaggedAny::Cat, py::arg("srcs"), py::arg("axis"),
                 kRaggedCatDoc);
  m.attr("cat") = any.attr("cat");

  any.def("unique", &RaggedAny::Unique, py::arg("need_num_repeats") = false,
          py::arg("need_new2old_indexes") = false, kRaggedAnyUniqueDoc);

  any.def("normalize", &RaggedAny::Normalize, py::arg("use_log"),
          kRaggedAnyNormalizeDoc);

  any.def(
      "pad",
      [](RaggedAny &self, std::string mode, py::object padding_value)
          -> torch::Tensor { return self.Pad(mode, ToIValue(padding_value)); },
      py::arg("mode"), py::arg("padding_value"), kRaggedAnyPadDoc);

  any.def(
      "tolist",
      [](RaggedAny &self) -> py::list {
        if (self.any.Context()->GetDeviceType() != kCpu) {
          auto tmp = self.To("cpu");
          return RaggedAnyToList(tmp);
        }
        return RaggedAnyToList(self);
      },
      kRaggedAnyToListDoc);

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

        PyObject *ptr = THPDevice_New(device);

        // takes ownership
        return py::reinterpret_steal<py::object>(ptr);
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
         torch::Device device = torch::kCPU) -> RaggedAny {
        return RaggedAnyFromList(data, dtype, device);
      },
      py::arg("data"), py::arg("dtype") = py::none(),
      py::arg("device") = torch::Device(torch::kCPU),
      kCreateRaggedTensorDataDoc);

  m.def(
      "create_ragged_tensor",
      [](py::list data, py::object dtype = py::none(),
         const std::string &device = "cpu") -> RaggedAny {
        return RaggedAnyFromList(data, dtype, device);
      },
      py::arg("data"), py::arg("dtype") = py::none(), py::arg("device") = "cpu",
      kCreateRaggedTensorDataDoc);

  m.def(
      "create_ragged_tensor",
      [](const std::string &s,
         torch::optional<torch::Dtype> dtype = torch::optional<torch::Dtype>(),
         torch::Device device = torch::kCPU) -> RaggedAny {
        return RaggedAny(s, dtype, device);
      },
      py::arg("s"), py::arg("dtype") = torch::optional<torch::Dtype>(),
      py::arg("device") = torch::Device(torch::kCPU),
      kCreateRaggedTensorStrDoc);

  m.def(
      "create_ragged_tensor",
      [](const std::string &s,
         torch::optional<torch::Dtype> dtype = torch::optional<torch::Dtype>(),
         const std::string &device = "cpu") -> RaggedAny {
        return RaggedAny(s, dtype, device);
      },
      py::arg("s"), py::arg("dtype") = torch::optional<torch::Dtype>(),
      py::arg("device") = "cpu", kCreateRaggedTensorStrDoc);

  m.def(
      "create_ragged_tensor",
      [](torch::Tensor tensor) -> RaggedAny { return RaggedAny(tensor); },
      py::arg("tensor"), kCreateRaggedTensorTensorDoc);
}

}  // namespace k2
