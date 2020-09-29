/**
 * @brief python wrappers for FSA.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

static void PybindFsaImpl(py::module &m) {
  using PyClass = Fsa;
  py::class_<PyClass> pyclass(m, "Fsa");
  pyclass.def(py::init<>());
  pyclass.def(py::init([](torch::Tensor &arcs) {
    Array1<Arc> array = FromTensor<Arc>(arcs);
    bool error = true;
    Fsa fsa = FsaFromArray1(array, &error);
    K2_CHECK_EQ(error, false);
    return new Fsa(fsa);  // python takes the ownership
  }));

  pyclass.def(
      "arcs",
      [](PyClass &self) -> torch::Tensor { return ToTensor(self.values); },
      py::keep_alive<0, 1>());

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

static void PybindFsaUtil(py::module &m) {
  m.def("fsa_to_str", &FsaToString, py::arg("fsa"),
        py::arg("negate_scores") = false, py::arg("aux_lables") = nullptr);

  m.def("str_to_fsa", &FsaFromString, py::arg("s"),
        py::arg("negate_scores") = false, py::arg("aux_labels") = nullptr);
}

}  // namespace k2

void PybindFsa(py::module &m) {
  k2::PybindFsaImpl(m);
  k2::PybindFsaUtil(m);
}
