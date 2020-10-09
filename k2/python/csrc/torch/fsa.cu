/**
 * @brief python wrappers for FSA.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <string>
#include <utility>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

static void PybindDenseFsa(py::module &m) {
  using PyClass = DenseFsaVec;
  py::class_<PyClass> pyclass(m, "_DenseFsaVec");

  pyclass.def_readwrite("shape", &PyClass::shape);
  pyclass.def_readwrite("scores", &PyClass::scores);
  pyclass.def("num_arcs", &PyClass::NumArcs);
}

static void PybindFsaUtil(py::module &m) {
  m.def("_fsa_from_tensor", [](torch::Tensor tensor) -> Fsa {
    Array1<Arc> array = FromTensor<Arc>(tensor);
    bool error = true;
    Fsa fsa = FsaFromArray1(array, &error);
    K2_CHECK(!error);
    return fsa;
  });

  m.def(
      "_fsa_to_str",
      [](Fsa &fsa, bool openfst = false,
         torch::optional<torch::Tensor> aux_labels = {}) -> std::string {
        Array1<int32_t> array;
        if (aux_labels.has_value())
          array = FromTensor<int32_t>(aux_labels.value());
        return FsaToString(fsa, openfst, aux_labels ? &array : nullptr);
      },
      py::arg("fsa"), py::arg("openfst") = false,
      py::arg("aux_labels") = py::none());

  m.def(
      "_fsa_from_str",
      [](const std::string &s, bool acceptor = true, bool openfst = false)
          -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        Array1<int32_t> aux_labels;
        Fsa fsa = FsaFromString(s, openfst, acceptor ? nullptr : &aux_labels);
        torch::optional<torch::Tensor> tensor;
        if (aux_labels.Dim() > 0) tensor = ToTensor(aux_labels);
        return std::make_pair(fsa, tensor);
      },
      py::arg("s"), py::arg("acceptor") = true, py::arg("openfst") = false,
      "It returns a tuple with two elements. Element 0 is the FSA; element 1 "
      "is a 1-D tensor of dtype torch.int32 containing the aux_labels if the "
      "returned FSA is a transducer; element 1 is None if the "
      "returned FSA is an acceptor");
}

}  // namespace k2

void PybindFsa(py::module &m) {
  k2::PybindFsaUtil(m);
  k2::PybindDenseFsa(m);
}
