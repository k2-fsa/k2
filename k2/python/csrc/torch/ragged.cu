/**
 * @brief python wrappers for Ragged<T>.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

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

  pyclass.def("to_cpu", [](const PyClass &self) -> PyClass {
    return self.To(GetCpuContext());
  });

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

}  // namespace k2

void PybindRagged(py::module &m) { k2::PybindRaggedImpl(m); }
