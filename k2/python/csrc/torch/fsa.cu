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
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

static void PybindFsaBasicProperties(py::module &m) {
  m.def("fsa_properties_as_str", &FsaPropertiesAsString);

  m.def("get_fsa_basic_properties", &GetFsaBasicProperties);

  m.def("is_arc_sorted", [](int32_t properties) -> bool {
    return (properties & kFsaPropertiesArcSorted) == kFsaPropertiesArcSorted;
  });

  m.def("is_accessible", [](int32_t properties) -> bool {
    return (properties & kFsaPropertiesMaybeAccessible) ==
           kFsaPropertiesMaybeAccessible;
  });

  m.def("is_coaccessible", [](int32_t properties) -> bool {
    return (properties & kFsaPropertiesMaybeCoaccessible) ==
           kFsaPropertiesMaybeCoaccessible;
  });
}

static void PybindFsaUtil(py::module &m) {
  // TODO(fangjun): add docstring in Python describing
  // the format of the input tensor when it is a FsaVec.
  m.def(
      "_fsa_from_tensor",
      [](torch::Tensor tensor) -> FsaOrVec {
        auto k2_tensor = FromTensor(tensor, TensorTag{});
        bool error = true;
        Fsa fsa;
        if (tensor.dim() == 2)
          fsa = FsaFromTensor(k2_tensor, &error);
        else if (tensor.dim() == 1)
          fsa = FsaVecFromTensor(k2_tensor, &error);
        else
          K2_LOG(FATAL)
              << "Expect dim: 2 (a single FSA) or 1 (a vector of FSAs). "
              << "Given: " << tensor.dim();

        K2_CHECK(!error);
        return fsa;
      },
      py::arg("tensor"));

  m.def(
      "_fsa_to_tensor",
      [](const FsaOrVec &fsa) -> torch::Tensor {
        if (fsa.NumAxes() == 2) {
          Tensor tensor = FsaToTensor(fsa);
          return ToTensor(tensor);
        } else if (fsa.NumAxes() == 3) {
          Tensor tensor = FsaVecToTensor(fsa);
          return ToTensor(tensor);
        } else {
          K2_LOG(FATAL) << "Unsupported num_axes: " << fsa.NumAxes();
          return {};
        }
      },
      py::arg("fsa"));

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

  // the following methods are for debugging only
  m.def("_fsa_to_fsa_vec", &FsaToFsaVec, py::arg("fsa"));
  m.def("_get_fsa_vec_element", &GetFsaVecElement, py::arg("vec"),
        py::arg("i"));
  m.def("_create_fsa_vec", [](std::vector<Fsa *> &fsas) -> FsaVec {
    return CreateFsaVec(fsas.size(), fsas.data());
  });
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
  k2::PybindFsaBasicProperties(m);
}
