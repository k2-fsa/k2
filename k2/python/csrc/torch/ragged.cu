/**
 * @brief python wrappers for Ragged<T>.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

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

  pyclass.def(
      "to",
      [](const PyClass &self, py::object device) -> PyClass {
        std::string device_type = static_cast<py::str>(device.attr("type"));
        K2_CHECK(device_type == "cpu" || device_type == "cuda")
            << "Unsupported device type: " << device_type;

        ContextPtr &context = self.Context();
        if (device_type == "cpu") {
          if (context->GetDeviceType() == kCpu) return self;
          return self.To(GetCpuContext());
        }

        auto index_attr = static_cast<py::object>(device.attr("index"));
        int32_t device_index = 0;
        if (!index_attr.is_none())
          device_index = static_cast<py::int_>(index_attr);

        if (context->GetDeviceType() == kCuda &&
            context->GetDeviceId() == device_index)
          return self;

        return self.To(GetCudaContext(device_index));
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

  // Return a pair:
  // - Ragged<T>
  // - value_indexes_out
  //     a 1-D torch::Tensor of dtype torch.int32 if need_value_indexes_out ==
  //     true, None if need_value_indexes_out == false
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
}

static void PybindRaggedImpl(py::module &m) {
  PybindRaggedTpl<Arc>(m, "RaggedArc");
  PybindRaggedTpl<int32_t>(m, "RaggedInt");
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
        std::string device_type = static_cast<py::str>(device.attr("type"));
        K2_CHECK(device_type == "cpu" || device_type == "cuda")
            << "Unsupported device type: " << device_type;

        ContextPtr &context = self.Context();
        if (device_type == "cpu") {
          if (context->GetDeviceType() == kCpu) return self;
          return self.To(GetCpuContext());
        }

        int32_t device_index = static_cast<py::int_>(device.attr("index"));
        if (context->GetDeviceType() == kCuda &&
            context->GetDeviceId() == device_index)
          return self;

        return self.To(GetCudaContext(device_index));
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

  pyclass.def("__str__", [](const PyClass &self) -> std::string {
    std::ostringstream os;
    os << self;
    return os.str();
  });
}

static void PybindRandomRaggedShape(py::module &m) {
  m.def("random_ragged_shape", &RandomRaggedShape, "RandomRaggedShape",
        py::arg("set_row_ids") = false, py::arg("min_num_axes") = 2,
        py::arg("max_num_axes") = 4, py::arg("min_num_elements") = 0,
        py::arg("max_num_elements") = 2000);
}

}  // namespace k2

void PybindRagged(py::module &m) {
  k2::PybindRaggedImpl(m);
  k2::PybindRaggedShape(m);
  k2::PybindRandomRaggedShape(m);
}
