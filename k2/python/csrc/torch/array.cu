/**
 * @brief python wrappers for Array.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <type_traits>

#include "c10/core/ScalarType.h"
#include "k2/csrc/array.h"
#include "k2/csrc/pytorch_context.h"
#include "k2/python/csrc/torch/array.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

template <typename T>
static void PybindArray1Tpl(py::module &m, const char *name) {
  using PyClass = Array1<T>;
  py::class_<PyClass> pyclass(m, name);
  pyclass.def(py::init<>());
  pyclass.def("tensor", [](PyClass &self) { return ToTensor(self); });

  pyclass.def_static(
      "from_tensor",
      [](torch::Tensor &tensor) { return FromTensor<T>(tensor); },
      py::arg("tensor"));

  // the following functions are for testing only
  pyclass.def(
      "get", [](const PyClass &self, int32_t i) { return self[i]; },
      py::arg("i"));
  pyclass.def("__str__", [](const PyClass &self) {
    std::ostringstream os;
    os << self;
    return os.str();
  });
}

static void PybindArrayImpl(py::module &m) {
  // users should not use classes with prefix `_` in Python.
  PybindArray1Tpl<float>(m, "_FloatArray1");
  PybindArray1Tpl<int>(m, "_Int32Array1");

  // the following functions are for testing purposes
  // and they can be removed later.
  m.def("get_cpu_float_array1", []() {
    return Array1<float>(GetCpuContext(), {1, 2, 3, 4});
  });

  m.def("get_cpu_int_array1", []() {
    return Array1<int32_t>(GetCpuContext(), {1, 2, 3, 4});
  });

  m.def(
      "get_cuda_float_array1",
      [](int32_t gpu_id = -1) {
        return Array1<float>(GetCudaContext(gpu_id), {0, 1, 2, 3});
      },
      py::arg("gpu_id") = -1);

  m.def(
      "get_cuda_int_array1",
      [](int32_t gpu_id = -1) {
        return Array1<int32_t>(GetCudaContext(gpu_id), {0, 1, 2, 3});
      },
      py::arg("gpu_id") = -1);
}

}  // namespace k2

void PybindArray(py::module &m) { k2::PybindArrayImpl(m); }
