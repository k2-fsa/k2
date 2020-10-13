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

static void PybindFsaUtil(py::module &m) {
  m.def("_fsa_from_tensor", [](torch::Tensor tensor) -> Fsa {
    Array1<Arc> array = FromTensor<Arc>(tensor);
    bool error = true;
    Fsa fsa = FsaFromArray1(array, &error);
    // TODO(fangjun): implement FsaVecFromArray1
    // if (error == true) fsa = FsaVecFromArray1(array, &error);
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

static void PybindDenseFsaVec(py::module &m) {
  using PyClass = DenseFsaVec;
  py::class_<PyClass> pyclass(m, "DenseFsaVec");
  // We do not need to access its members in Python

  // TODO(fangjun): add docstring for this funciton
  pyclass.def(py::init(
      [](torch::Tensor scores, torch::Tensor row_splits) -> DenseFsaVec * {
        // remove the contiguous check once the following comment
        // https://github.com/k2-fsa/k2/commit/60b8e97b1838033b45b83cc88a58ec91912ce91e#r43174753
        // is resolved.
        K2_CHECK(scores.is_contiguous());
        Array1<int32_t> _row_splits = FromTensor<int32_t>(row_splits);
        DenseFsaVec *dense_fsa = new DenseFsaVec;  // will be freed by Python
        dense_fsa->shape = RaggedShape2(&_row_splits, nullptr, -1);
        dense_fsa->scores = FromTensor<float>(scores, Array2Tag{});

        K2_CHECK(IsCompatible(dense_fsa->shape, dense_fsa->scores));

        return dense_fsa;  // Python takes the ownership
      }));

  // the `to_str` method is for debugging only
  pyclass.def("to_str", [](PyClass &self) {
    std::ostringstream os;
    os << "num_axes: " << self.shape.NumAxes() << '\n';
    os << "device_type: " << self.shape.Context()->GetDeviceType() << '\n';
    os << "row_splits1: " << self.shape.RowSplits(1) << '\n';
    os << "row_ids1: " << self.shape.RowIds(1) << '\n';
    os << "scores:" << self.scores << '\n';
    return os.str();
  });
}

}  // namespace k2

void PybindFsa(py::module &m) {
  k2::PybindFsaUtil(m);
  k2::PybindDenseFsaVec(m);
}
