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
#include <vector>

#include "c10/core/ScalarType.h"
#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/pytorch_context.h"
#include "k2/python/csrc/torch/array.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

template <typename T>
static void PybindArray2Tpl(py::module &m, const char *name) {
  using PyClass = Array2<T>;
  py::class_<PyClass> pyclass(m, name);
  pyclass.def("tensor",
              [](PyClass &self) -> torch::Tensor { return ToTensor(self); });

  pyclass.def_static(
      "from_tensor",
      [](torch::Tensor &tensor) -> PyClass {
        return FromTensor<T>(tensor, Array2Tag{});
      },
      py::arg("tensor"));

  // the following functions are for testing only
  pyclass.def(
      "get", [](PyClass &self, int32_t i) -> Array1<T> { return self[i]; },
      py::arg("i"));

  pyclass.def("__str__", [](const PyClass &self) {
    std::ostringstream os;
    os << self;
    return os.str();
  });
}

template <typename T>
static void PybindArray1Tpl(py::module &m, const char *name) {
  using PyClass = Array1<T>;
  py::class_<PyClass> pyclass(m, name);
  pyclass.def("tensor",
              [](PyClass &self) -> torch::Tensor { return ToTensor(self); });

  pyclass.def_static(
      "from_tensor",
      [](torch::Tensor &tensor) -> PyClass { return FromTensor<T>(tensor); },
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
  PybindArray1Tpl<Arc>(m, "_ArcArray1");

  PybindArray2Tpl<float>(m, "_FloatArray2");
  PybindArray2Tpl<int>(m, "_Int32Array2");

  // the following functions are for testing purposes
  // and they can be removed later.
  m.def("get_cpu_float_array1", []() -> Array1<float> {
    return Array1<float>(GetCpuContext(), {1, 2, 3, 4});
  });

  m.def("get_cpu_int_array1", []() -> Array1<int32_t> {
    return Array1<int32_t>(GetCpuContext(), {1, 2, 3, 4});
  });

  m.def(
      "get_cuda_float_array1",
      [](int32_t gpu_id = -1) -> Array1<float> {
        return Array1<float>(GetCudaContext(gpu_id), {0, 1, 2, 3});
      },
      py::arg("gpu_id") = -1);

  m.def(
      "get_cuda_int_array1",
      [](int32_t gpu_id = -1) -> Array1<int32_t> {
        return Array1<int32_t>(GetCudaContext(gpu_id), {0, 1, 2, 3});
      },
      py::arg("gpu_id") = -1);

  m.def("get_cpu_arc_array1", []() -> Array1<Arc> {
    std::vector<Arc> arcs = {
        {1, 2, 3, 1.5},
        {10, 20, 30, 2.5},
    };
    return Array1<Arc>(GetCpuContext(), arcs);
  });

  m.def(
      "get_cuda_arc_array1",
      [](int32_t gpu_id = -1) -> Array1<Arc> {
        std::vector<Arc> arcs = {
            {1, 2, 3, 1.5},
            {10, 20, 30, 2.5},
        };
        return Array1<Arc>(GetCudaContext(gpu_id), arcs);
      },
      py::arg("gpu_id") = -1);

  m.def("get_cpu_int_array2", []() -> Array2<int32_t> {
    Array1<int32_t> array1(GetCpuContext(), {1, 2, 3, 4, 5, 6});
    return Array2<int32_t>(array1, 2, 3);
  });

  m.def(
      "get_cuda_float_array2",
      [](int32_t gpu_id = -1) -> Array2<float> {
        Array1<float> array1(GetCudaContext(gpu_id), {1, 2, 3, 4, 5, 6});
        return Array2<float>(array1, 2, 3);
      },
      py::arg("gpu_id") = -1);
}

}  // namespace k2

void PybindArray(py::module &m) { k2::PybindArrayImpl(m); }
