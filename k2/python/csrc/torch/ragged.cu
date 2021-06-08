/**
 * @brief python wrappers for Ragged<T>.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang, Liyong Guo)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/device_guard.h"
#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/ragged.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {
template <typename T>
static void PybindRaggedTpl(py::module &m, const char *name) {
  using PyClass = Ragged<T>;
  py::class_<PyClass> pyclass(m, name);

  pyclass.def(py::init([](const std::string &s) -> std::unique_ptr<PyClass> {
    return std::make_unique<PyClass>(s);
  }));

  pyclass.def(py::init([](const RaggedShape &shape,
                          torch::Tensor values) -> std::unique_ptr<PyClass> {
                DeviceGuard guard(shape.Context());
                K2_CHECK_EQ(shape.NumElements(), values.sizes()[0]);
                return std::make_unique<PyClass>(shape, FromTorch<T>(values));
              }),
              py::arg("shape"), py::arg("values"));
  pyclass.def(
      "to",
      [](const PyClass &self, py::object device) -> PyClass {
        DeviceGuard guard(self.Context());
        return To(self, device);
      },
      py::arg("device"));

  pyclass.def("cpu", [](const PyClass &self) -> PyClass {
    DeviceGuard guard(self.Context());
    return self.To(GetCpuContext());
  });

  pyclass.def("clone", [](const PyClass &self) -> PyClass {
    DeviceGuard guard(self.Context());
    return self.Clone();
  });

  pyclass.def("is_cpu", [](const PyClass &self) -> bool {
    return self.Context()->GetDeviceType() == kCpu;
  });

  pyclass.def("is_cuda", [](const PyClass &self) -> bool {
    return self.Context()->GetDeviceType() == kCuda;
  });

  pyclass.def("values", [](PyClass &self) -> torch::Tensor {
    DeviceGuard guard(self.Context());
    Array1<T> &values = self.values;
    return ToTorch(values);
  });

  pyclass.def("num_elements", [](PyClass &self) -> int32_t {
    DeviceGuard guard(self.Context());
    return self.NumElements();
  });

  pyclass.def("shape", [](PyClass &self) -> RaggedShape { return self.shape; });

  pyclass.def(
      "row_splits",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        DeviceGuard guard(self.Context());
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTorch(row_splits);
      },
      py::arg("axis"));

  pyclass.def(
      "row_ids",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        DeviceGuard guard(self.Context());
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTorch(row_ids);
      },
      py::arg("axis"));

  pyclass.def(
      "tot_size",
      [](PyClass &self, int32_t axis) -> int32_t {
        DeviceGuard guard(self.Context());
        return self.TotSize(axis);
      },
      py::arg("axis"));

  // Dim0() does not access GPU memory
  pyclass.def("dim0", &PyClass::Dim0);

  // NumAxes() does not access GPU memory
  pyclass.def("num_axes", &PyClass::NumAxes);

  pyclass.def(
      "index",
      [](PyClass &self, int32_t axis,
         int32_t i) -> std::pair<PyClass, int32_t> {
        DeviceGuard guard(self.Context());
        Ragged<T> ans = self.Index(axis, i);
        int32_t offset = ans.values.Data() - self.values.Data();
        return std::make_pair(ans, offset);
      },
      py::arg("axis"), py::arg("i"));

  pyclass.def("__eq__", [](const PyClass &self, const PyClass &other) -> bool {
    DeviceGuard guard(self.Context());
    return Equal(self, other);
    }, py::arg("other"));

  pyclass.def("__ne__", [](const PyClass &self, const PyClass &other) -> bool {
    DeviceGuard guard(self.Context());
    return !Equal(self, other);
    }, py::arg("other"));

  pyclass.def("__str__", [](const PyClass &self) -> std::string {
    DeviceGuard guard(self.Context());
    std::ostringstream os;
    os << self;
    return os.str();
  });

  pyclass.def("tot_sizes", [](const PyClass &self) -> std::vector<int32_t> {
    DeviceGuard guard(self.Context());
    int32_t num_axes = self.NumAxes();
    std::vector<int32_t> ans(num_axes);
    for (int32_t i = 0; i != num_axes; ++i) ans[i] = self.TotSize(i);
    return ans;
  });

  pyclass.def(py::pickle(
      [](const PyClass &obj) {
        DeviceGuard guard(obj.Context());
        K2_CHECK_EQ(obj.NumAxes(), 2)
            << "Only support Ragged with NumAxes() == 2 for now";
        Array1<int32_t> row_splits1 = obj.RowSplits(1);
        Array1<int32_t> row_ids1 = obj.RowIds(1);
        Array1<T> values = obj.values;
        return py::make_tuple(ToTorch(row_splits1), ToTorch(row_ids1),
                              ToTorch(values));
      },
      [](py::tuple t) {
        K2_CHECK_EQ(t.size(), 3) << "Invalid state";
        torch::Tensor row_splits1_tensor = t[0].cast<torch::Tensor>();
        DeviceGuard guard(GetContext(row_splits1_tensor));
        Array1<int32_t> row_splits1 = FromTorch<int32_t>(row_splits1_tensor);
        torch::Tensor row_ids1_tensor = t[1].cast<torch::Tensor>();
        Array1<int32_t> row_ids1 = FromTorch<int32_t>(row_ids1_tensor);
        torch::Tensor values_tensor = t[2].cast<torch::Tensor>();
        Array1<T> values = FromTorch<T>(values_tensor);
        RaggedShape shape = RaggedShape2(&row_splits1, &row_ids1, -1);
        PyClass obj(shape, values);
        return obj;
      }));

  // Return a pair:
  // - Ragged<T>
  // - value_indexes_out
  //     a 1-D torch::Tensor of dtype torch.int32 if need_value_indexes_out
  //     == true, None if need_value_indexes_out == false
  m.def(
      "index",
      [](PyClass &src, int32_t axis, torch::Tensor indexes,
         bool need_value_indexes =
             true) -> std::pair<PyClass, torch::optional<torch::Tensor>> {
        DeviceGuard guard(src.Context());
        Array1<int32_t> indexes_array = FromTorch<int32_t>(indexes);
        Array1<int32_t> value_indexes;

        Ragged<T> ans = Index(src, axis, indexes_array,
                              need_value_indexes ? &value_indexes : nullptr);
        torch::optional<torch::Tensor> value_indexes_tensor;

        if (need_value_indexes) value_indexes_tensor = ToTorch(value_indexes);

        return std::make_pair(ans, value_indexes_tensor);
      },
      py::arg("src"), py::arg("axis"), py::arg("indexes"),
      py::arg("need_value_indexes") = true);

  m.def(
      "index",
      [](Ragged<T> &src, Ragged<int32_t> &indexes) -> Ragged<T> {
        DeviceGuard guard(src.Context());
        bool remove_axis = true;
        return Index(src, indexes, remove_axis);
      },
      py::arg("src"), py::arg("indexes"));
}

static void PybindRaggedImpl(py::module &m) {
  PybindRaggedTpl<Arc>(m, "RaggedArc");
  PybindRaggedTpl<int32_t>(m, "RaggedInt");
  PybindRaggedTpl<float>(m, "RaggedFloat");

  m.def(
      "index",
      [](torch::Tensor src, Ragged<int32_t> &indexes) -> Ragged<int32_t> {
        DeviceGuard guard(GetContext(src));
        K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
        Array1<int32_t> src_array = FromTorch<int32_t>(src);
        return Index(src_array, indexes);
      },
      py::arg("src"), py::arg("indexes"));

  m.def(
      "index_and_sum",
      [](torch::Tensor src, Ragged<int32_t> &indexes) -> torch::Tensor {
        DeviceGuard guard(GetContext(src));
        K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
        Array1<float> src_array = FromTorch<float>(src);
        Ragged<float> ragged = Index(src_array, indexes);
        Array1<float> ans_array(ragged.Context(), ragged.Dim0());
        SumPerSublist<float>(ragged, 0, &ans_array);
        return ToTorch(ans_array);
      },
      py::arg("src"), py::arg("indexes"));
}

static void PybindRaggedShape(py::module &m) {
  using PyClass = RaggedShape;
  py::class_<PyClass> pyclass(m, "RaggedShape");

  pyclass.def(py::init<const std::string &>(), py::arg("src"));

  pyclass.def("dim0", &PyClass::Dim0);

  pyclass.def(
      "max_size",
      [](PyClass &self, int32_t axis) -> int32_t {
        DeviceGuard guard(self.Context());
        return self.MaxSize(axis);
      },
      py::arg("axis"));

  pyclass.def("num_axes", &PyClass::NumAxes);

  pyclass.def("num_elements", [](PyClass &self) -> int32_t {
    DeviceGuard guard(self.Context());
    return self.NumElements();
  });

  pyclass.def(
      "tot_size",
      [](PyClass &self, int32_t axis) -> int32_t {
        DeviceGuard guard(self.Context());
        return self.TotSize(axis);
      },
      py::arg("axis"));

  pyclass.def(
      "to",
      [](const PyClass &self, py::object device) -> PyClass {
        DeviceGuard guard(self.Context());
        return To(self, device);
      },
      py::arg("device"));

  pyclass.def(
      "row_ids",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        DeviceGuard guard(self.Context());
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTorch(row_ids);
      },
      py::arg("axis"));

  pyclass.def(
      "row_splits",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        DeviceGuard guard(self.Context());
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTorch(row_splits);
      },
      py::arg("axis"));

  pyclass.def("tot_sizes", [](const PyClass &self) -> std::vector<int32_t> {
    DeviceGuard guard(self.Context());
    int32_t num_axes = self.NumAxes();
    std::vector<int32_t> ans(num_axes);
    for (int32_t i = 0; i != num_axes; ++i) ans[i] = self.TotSize(i);
    return ans;
  });

  pyclass.def("__eq__", [](const PyClass &self, const PyClass &other) -> bool {
    DeviceGuard guard(self.Context());
    return Equal(self, other);
    }, py::arg("other"));

  pyclass.def("__ne__", [](const PyClass &self, const PyClass &other) -> bool {
    DeviceGuard guard(self.Context());
    return !Equal(self, other);
  }, py::arg("other"));

  pyclass.def("__str__", [](const PyClass &self) -> std::string {
    DeviceGuard guard(self.Context());
    std::ostringstream os;
    os << self;
    return os.str();
  });

  pyclass.def(
      "index",
      [](PyClass &self, int32_t axis,
         int32_t i) -> std::pair<PyClass, int32_t> {
        DeviceGuard guard(self.Context());
        int32_t value_offset;
        RaggedShape ans = self.Index(axis, i, &value_offset);
        return std::make_pair(ans, value_offset);
      },
      py::arg("axis"), py::arg("i"));
}

static void PybindRaggedShapeUtils(py::module &m) {
  m.def("random_ragged_shape", &RandomRaggedShape, "RandomRaggedShape",
        py::arg("set_row_ids") = false, py::arg("min_num_axes") = 2,
        py::arg("max_num_axes") = 4, py::arg("min_num_elements") = 0,
        py::arg("max_num_elements") = 2000);
  m.def(
      "create_ragged_shape2",
      [](torch::optional<torch::Tensor> row_splits,
         torch::optional<torch::Tensor> row_ids,
         int32_t cached_tot_size = -1) -> RaggedShape {
        if (!row_splits.has_value() && !row_ids.has_value())
          K2_LOG(FATAL) << "Both row_splits and row_ids are None";

        int32_t device_id = -1;
        if (row_splits.has_value() && row_splits->is_cuda()) {
          device_id = row_splits->device().index();
        } else if (row_ids.has_value() && row_ids->is_cuda()) {
          device_id = row_ids->device().index();
        }

        DeviceGuard guard(device_id);

        Array1<int32_t> array_row_splits;
        if (row_splits.has_value())
          array_row_splits = FromTorch<int32_t>(row_splits.value());
        Array1<int32_t> array_row_ids;
        if (row_ids.has_value())
          array_row_ids = FromTorch<int32_t>(row_ids.value());
        return RaggedShape2(
            row_splits.has_value() ? &array_row_splits : nullptr,
            row_ids.has_value() ? &array_row_ids : nullptr, cached_tot_size);
      },
      py::arg("row_splits"), py::arg("row_ids"),
      py::arg("cached_tot_size") = -1);

  m.def(
      "compose_ragged_shapes",
      [](const RaggedShape &a, const RaggedShape &b) -> RaggedShape {
        DeviceGuard guard(a.Context());
        return ComposeRaggedShapes(a, b);
      },
      py::arg("a"), py::arg("b"));

  m.def(
      "remove_axis",
      [](RaggedShape &src, int32_t axis) -> RaggedShape {
        DeviceGuard guard(src.Context());
        return RemoveAxis(src, axis);
      },
      py::arg("src"), py::arg("axis"));
}

}  // namespace k2

void PybindRagged(py::module &m) {
  k2::PybindRaggedImpl(m);
  k2::PybindRaggedShape(m);
  k2::PybindRaggedShapeUtils(m);
}
