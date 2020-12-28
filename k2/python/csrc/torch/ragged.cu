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
                K2_CHECK_EQ(shape.NumElements(), values.sizes()[0]);
                return std::make_unique<PyClass>(shape, FromTensor<T>(values));
              }),
              py::arg("shape"), py::arg("values"));
  pyclass.def(
      "to",
      [](const PyClass &self, py::object device) -> PyClass {
        return To(self, device);
      },
      py::arg("device"));

  pyclass.def("to_cpu", [](const PyClass &self) -> PyClass {
    return self.To(GetCpuContext());
  });

  pyclass.def("clone",
              [](const PyClass &self) -> PyClass { return self.Clone(); });

  pyclass.def("is_cpu", [](const PyClass &self) -> bool {
    return self.Context()->GetDeviceType() == kCpu;
  });

  pyclass.def(
      "to_cuda",
      [](const PyClass &self, int32_t device_id) -> PyClass {
        ContextPtr c = GetCudaContext(device_id);
        return self.To(c);
      },
      py::arg("device_id"));

  pyclass.def("is_cuda", [](const PyClass &self) -> bool {
    return self.Context()->GetDeviceType() == kCuda;
  });

  pyclass.def("values", [](PyClass &self) -> torch::Tensor {
    Array1<T> &values = self.values;
    return ToTensor(values);
  });

  pyclass.def("num_elements", &PyClass::NumElements);

  pyclass.def("shape", [](PyClass &self) -> RaggedShape { return self.shape; });

  pyclass.def(
      "row_splits",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTensor(row_splits);
      },
      py::arg("axis"));

  pyclass.def(
      "row_ids",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTensor(row_ids);
      },
      py::arg("axis"));

  pyclass.def("tot_size", &PyClass::TotSize, py::arg("axis"));
  pyclass.def("dim0", &PyClass::Dim0);
  pyclass.def("num_axes", &PyClass::NumAxes);
  pyclass.def(
      "index",
      [](PyClass &self, int32_t axis,
         int32_t i) -> std::pair<PyClass, int32_t> {
        Ragged<T> ans = self.Index(axis, i);
        int32_t offset = ans.values.Data() - self.values.Data();
        return std::make_pair(ans, offset);
      },
      py::arg("axis"), py::arg("i"));

  pyclass.def("__str__", [](const PyClass &self) -> std::string {
    std::ostringstream os;
    os << self;
    return os.str();
  });

  pyclass.def("tot_sizes", [](const PyClass &self) -> py::list {
    int32_t num_axes = self.NumAxes();
    py::list ans(num_axes);
    for (int32_t i = 0; i < self.NumAxes(); i++) ans[i] = self.TotSize(i);
    return ans;
  });

  pyclass.def(py::pickle(
      [](const PyClass &obj) {
        K2_CHECK_EQ(obj.NumAxes(), 2)
            << "Only support Ragged with NumAxes() == 2 for now";
        Array1<int32_t> row_splits1 = obj.RowSplits(1);
        Array1<int32_t> row_ids1 = obj.RowIds(1);
        Array1<T> values = obj.values;
        return py::make_tuple(ToTensor(row_splits1), ToTensor(row_ids1),
                              ToTensor(values));
      },
      [](py::tuple t) {
        K2_CHECK_EQ(t.size(), 3) << "Invalid state";
        torch::Tensor row_splits1_tensor = t[0].cast<torch::Tensor>();
        Array1<int32_t> row_splits1 = FromTensor<int32_t>(row_splits1_tensor);
        torch::Tensor row_ids1_tensor = t[1].cast<torch::Tensor>();
        Array1<int32_t> row_ids1 = FromTensor<int32_t>(row_ids1_tensor);
        torch::Tensor values_tensor = t[2].cast<torch::Tensor>();
        Array1<T> values = FromTensor<T>(values_tensor);
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
      [](PyClass &src, torch::Tensor indexes, bool need_value_indexes = true)
          -> std::pair<PyClass, torch::optional<torch::Tensor>> {
        Array1<int32_t> indexes_array = FromTensor<int32_t>(indexes);
        Array1<int32_t> value_indexes;

        Ragged<T> ans = Index(src, indexes_array,
                              need_value_indexes ? &value_indexes : nullptr);
        torch::optional<torch::Tensor> value_indexes_tensor;

        if (need_value_indexes) value_indexes_tensor = ToTensor(value_indexes);

        return std::make_pair(ans, value_indexes_tensor);
      },
      py::arg("src"), py::arg("indexes"), py::arg("need_value_indexes") = true);

  m.def(
      "index",
      [](Ragged<T> &src, Ragged<int32_t> &indexes) -> Ragged<T> {
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
        K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
        Array1<int32_t> src_array = FromTensor<int32_t>(src);
        return Index(src_array, indexes);
      },
      py::arg("src"), py::arg("indexes"));
}

static void PybindRaggedShape(py::module &m) {
  using PyClass = RaggedShape;
  py::class_<PyClass> pyclass(m, "RaggedShape");
  pyclass.def(py::init<const std::string &>(), py::arg("src"));
  pyclass.def("dim0", &PyClass::Dim0);
  pyclass.def("max_size", &PyClass::MaxSize, py::arg("axis"));
  pyclass.def("num_axes", &PyClass::NumAxes);
  pyclass.def("num_elements", &PyClass::NumElements);
  pyclass.def("tot_size", &PyClass::TotSize, py::arg("axis"));

  pyclass.def(
      "to",
      [](const PyClass &self, py::object device) -> PyClass {
        return To(self, device);
      },
      py::arg("device"));

  pyclass.def(
      "row_ids",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTensor(row_ids);
      },
      py::arg("axis"));

  pyclass.def(
      "row_splits",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTensor(row_splits);
      },
      py::arg("axis"));

  pyclass.def("tot_sizes", [](const PyClass &self) -> py::list {
    int32_t num_axes = self.NumAxes();
    py::list ans(num_axes);
    for (int32_t i = 0; i < self.NumAxes(); i++) ans[i] = self.TotSize(i);
    return ans;
  });

  pyclass.def("__str__", [](const PyClass &self) -> std::string {
    std::ostringstream os;
    os << self;
    return os.str();
  });

  pyclass.def(
      "index",
      [](PyClass &self, int32_t axis,
         int32_t i) -> std::pair<PyClass, int32_t> {
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
        Array1<int32_t> array_row_splits;
        if (row_splits.has_value())
          array_row_splits = FromTensor<int32_t>(row_splits.value());
        Array1<int32_t> array_row_ids;
        if (row_ids.has_value())
          array_row_ids = FromTensor<int32_t>(row_ids.value());
        return RaggedShape2(
            row_splits.has_value() ? &array_row_splits : nullptr,
            row_ids.has_value() ? &array_row_ids : nullptr, cached_tot_size);
      },
      py::arg("row_splits"), py::arg("row_ids"),
      py::arg("cached_tot_size") = -1);
  m.def("compose_ragged_shapes", ComposeRaggedShapes, py::arg("a"),
        py::arg("b"));

  m.def(
      "ragged_shape_remove_axis",
      [](RaggedShape &src, int32_t axis) -> RaggedShape {
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
