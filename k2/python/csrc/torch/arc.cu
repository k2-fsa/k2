/**
 * @brief python wrappers for Arc.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <string>

#include "k2/csrc/fsa.h"
#include "k2/python/csrc/torch/arc.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

static void PybindArcImpl(py::module &m) {
  using PyClass = Arc;
  py::class_<PyClass> pyclass(m, "Arc");
  pyclass.def(py::init<>());
  pyclass.def(py::init<int32_t, int32_t, int32_t, float>(),
              py::arg("src_state"), py::arg("dest_state"), py::arg("label"),
              py::arg("score"));

  pyclass.def_readwrite("src_state", &PyClass::src_state)
      .def_readwrite("dest_state", &PyClass::dest_state)
      .def_readwrite("label", &PyClass::label)
      .def_readwrite("score", &PyClass::score);

  pyclass.def("__str__", [](const PyClass &self) -> std::string {
    std::ostringstream os;
    os << self;
    return os.str();
  });

  m.def("float_as_int",
        [](float f) -> int32_t { return *reinterpret_cast<int32_t *>(&f); });

  m.def("int_as_float",
        [](int32_t i) -> float { return *reinterpret_cast<float *>(&i); });

  m.def("as_int", [](torch::Tensor tensor) -> torch::Tensor {
    auto scalar_type = ToScalarType<int32_t>::value;
    if (tensor.numel() == 0)
      return torch::empty(tensor.sizes(), tensor.options().dtype(scalar_type));
    return torch::from_blob(
        tensor.data_ptr(), tensor.sizes(), tensor.strides(),
        [tensor](void *p) {}, tensor.options().dtype(scalar_type));
  });

  m.def("as_float", [](torch::Tensor tensor) -> torch::Tensor {
    auto scalar_type = ToScalarType<float>::value;
    if (tensor.numel() == 0)
      return torch::empty(tensor.sizes(), tensor.options().dtype(scalar_type));
    return torch::from_blob(
        tensor.data_ptr(), tensor.sizes(), tensor.strides(),
        [tensor](void *p) {}, tensor.options().dtype(scalar_type));
  });
}

}  // namespace k2

void PybindArc(py::module &m) { k2::PybindArcImpl(m); }
