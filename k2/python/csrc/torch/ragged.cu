/**
 * @brief python wrappers for Ragged<T>.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/ragged.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

static void PybindRaggedShape(py::module &m) {
  using PyClass = RaggedShape;
  py::class_<PyClass> pyclass(m, "RaggedShape");

  // NOTE: we do not provide constructors of RaggedShape
  // in Python, so the only way to get an instance of
  // RaggedShape is from Ragged<T>.

  pyclass.def("dim0", &PyClass::Dim0);
  pyclass.def("total_size", &PyClass::TotSize, py::arg("axis"));

  // TODO(fangjun): Append has not been implemented yet
  // pyclass.def("append", &PyClass::Append, py::arg("other"));

  pyclass.def("numel", &PyClass::NumElements);

  pyclass.def(
      "row_splits",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTensor(row_splits);
      },
      py::keep_alive<0, 1>());

  pyclass.def(
      "row_ids",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTensor(row_ids);
      },
      py::keep_alive<0, 1>());

  pyclass.def("num_axes", &PyClass::NumAxes);
  pyclass.def("max_size", &PyClass::MaxSize, py::arg("axis"));
  pyclass.def("index", &PyClass::Index, py::arg("axis"), py::arg("i"));
  pyclass.def(
      "get_offset",
      [](PyClass &self, const std::vector<int32_t> &indexes) -> int32_t {
        return self[indexes];
      });
}

template <typename T>
static void PybindRaggedTpl(py::module &m, const char *name) {
  using PyClass = Ragged<T>;
  py::class_<PyClass> pyclass(m, name);

  pyclass.def_readwrite("shape", &PyClass::shape);
  pyclass.def_readwrite("values", &PyClass::values);

  pyclass.def("num_axes", &PyClass::NumAxes);
  pyclass.def("index", &PyClass::Index, py::arg("axis"), py::arg("i"));
  pyclass.def("remove_axis", &PyClass::RemoveAxis, py::arg("axis"));

  pyclass.def(
      "cuda",
      [](const PyClass &self, int32_t gpu_id = -1) -> PyClass {
        auto context = GetCudaContext(gpu_id);
        return self.To(context);
      },
      py::arg("gpu_id") = -1);

  pyclass.def("cpu", [](const PyClass &self) -> PyClass {
    auto context = GetCpuContext();
    return self.To(context);
  });
}

static void PybindRaggedImpl(py::module &m) { PybindRaggedTpl<Arc>(m, "_Fsa"); }

}  // namespace k2

void PybindRagged(py::module &m) {
  k2::PybindRaggedShape(m);
  k2::PybindRaggedImpl(m);
}
