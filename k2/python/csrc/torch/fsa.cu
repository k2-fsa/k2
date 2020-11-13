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

#include <memory>
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
  m.def("fsa_properties_as_str", &FsaPropertiesAsString, py::arg("properties"));

  m.def("get_fsa_basic_properties", &GetFsaBasicProperties, py::arg("fsa"));

  m.def(
      "get_fsa_vec_basic_properties",
      [](FsaVec &fsa_vec) -> int32_t {
        int32_t tot_properties;
        Array1<int32_t> properties;
        GetFsaVecBasicProperties(fsa_vec, &properties, &tot_properties);
        return tot_properties;
      },
      py::arg("fsa_vec"));

  m.def(
      "is_arc_sorted",
      [](int32_t properties) -> bool {
        return (properties & kFsaPropertiesArcSorted) ==
               kFsaPropertiesArcSorted;
      },
      py::arg("properties"));

  m.def(
      "is_accessible",
      [](int32_t properties) -> bool {
        return (properties & kFsaPropertiesMaybeAccessible) ==
               kFsaPropertiesMaybeAccessible;
      },
      py::arg("properties"));

  m.def(
      "is_coaccessible",
      [](int32_t properties) -> bool {
        return (properties & kFsaPropertiesMaybeCoaccessible) ==
               kFsaPropertiesMaybeCoaccessible;
      },
      py::arg("properties"));
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
         torch::optional<torch::Tensor> aux_labels =
             torch::nullopt) -> std::string {
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

  m.def(
      "_create_fsa_vec",
      [](std::vector<Fsa *> &fsas) -> FsaVec {
        return CreateFsaVec(fsas.size(), fsas.data());
      },
      py::arg("fsas"));

  // returns Ragged<int32_t>
  m.def("_get_state_batches", &GetStateBatches, py::arg("fsas"),
        py::arg("transpose") = true);

  m.def(
      "_get_dest_states",
      [](FsaVec &fsas, bool as_idx01) -> torch::Tensor {
        Array1<int32_t> ans = GetDestStates(fsas, as_idx01);
        return ToTensor(ans);
      },
      py::arg("fsas"), py::arg("as_idx01"));

  m.def(
      "_get_incoming_arcs",
      [](FsaVec &fsas, torch::Tensor dest_states) -> Ragged<int32_t> {
        Array1<int32_t> dest_states_array = FromTensor<int32_t>(dest_states);
        return GetIncomingArcs(fsas, dest_states_array);
      },
      py::arg("fsas"), py::arg("dest_states"));

  // returns Ragged<int32_t>
  m.def("_get_entering_arc_index_batches", &GetEnteringArcIndexBatches,
        py::arg("fsas"), py::arg("incoming_arcs"), py::arg("state_batches"));

  // returns Ragged<int32_t>
  m.def("_get_leaving_arc_index_batches", &GetLeavingArcIndexBatches,
        py::arg("fsas"), py::arg("state_batches"));
}

template <typename T>
static void PybindGetForwardScores(py::module &m, const char *name) {
  // Return a std::pair
  //   - forward_scores, a torch::Tensor of dtype torch.float32 or torch.float64
  //   (depending on T) containing the scores
  //
  //   - entering_arcs (optional)
  //     - if log_semiring is true, it is None
  //     - else it is a torch::Tensor of dtype torch.int32
  m.def(
      name,
      [](FsaVec &fsas, Ragged<int32_t> &state_batches,
         Ragged<int32_t> &entering_arc_batches, bool log_semiring)
          -> std::pair<torch::Tensor, torch::optional<torch::Tensor>> {
        Array1<int32_t> entering_arcs;
        Array1<T> scores = GetForwardScores<T>(
            fsas, state_batches, entering_arc_batches, log_semiring,
            log_semiring ? nullptr : &entering_arcs);

        torch::optional<torch::Tensor> entering_arcs_tensor;
        if (!log_semiring) entering_arcs_tensor = ToTensor(entering_arcs);

        return std::make_pair(ToTensor(scores), entering_arcs_tensor);
      },
      py::arg("fsas"), py::arg("state_batches"),
      py::arg("entering_arc_batches"), py::arg("log_semiring"));
}

template <typename T>
static void PybindGetBackwardScores(py::module &m, const char *name) {
  m.def(
      name,
      [](FsaVec &fsas, Ragged<int32_t> &state_batches,
         Ragged<int32_t> &leaving_arc_batches,
         torch::optional<torch::Tensor> tot_scores = torch::nullopt,
         bool log_semiring = true) -> torch::Tensor {
        if (tot_scores.has_value()) {
          const Array1<T> tot_scores_array = FromTensor<T>(tot_scores.value());
          Array1<T> ans =
              GetBackwardScores<T>(fsas, state_batches, leaving_arc_batches,
                                   &tot_scores_array, log_semiring);
          return ToTensor(ans);
        } else {
          Array1<T> ans = GetBackwardScores<T>(
              fsas, state_batches, leaving_arc_batches, nullptr, log_semiring);
          return ToTensor(ans);
        }
      },
      py::arg("fsas"), py::arg("state_batches"), py::arg("leaving_arc_batches"),
      py::arg("tot_scores") = py::none(), py::arg("log_semiring") = true);
}

template <typename T>
static void PybindGetTotScores(py::module &m, const char *name) {
  m.def(
      name,
      [](FsaVec &fsas, torch::Tensor forward_scores) -> torch::Tensor {
        Array1<T> forward_scores_array = FromTensor<T>(forward_scores);
        Array1<T> tot_scores = GetTotScores(fsas, forward_scores_array);
        return ToTensor(tot_scores);
      },
      py::arg("fsas"), py::arg("forward_scores"));
}

static void PybindDenseFsaVec(py::module &m) {
  using PyClass = DenseFsaVec;
  py::class_<PyClass> pyclass(m, "DenseFsaVec");
  // We do not need to access its members in Python

  // TODO(fangjun): add docstring for this funciton
  pyclass.def(
      py::init([](torch::Tensor scores,
                  torch::Tensor row_splits) -> std::unique_ptr<DenseFsaVec> {
        // remove the contiguous check once the following comment
        // https://github.com/k2-fsa/k2/commit/60b8e97b1838033b45b83cc88a58ec91912ce91e#r43174753
        // is resolved.
        K2_CHECK(scores.is_contiguous());
        Array1<int32_t> row_splits_array = FromTensor<int32_t>(row_splits);

        RaggedShape shape = RaggedShape2(&row_splits_array, nullptr, -1);
        Array2<float> scores_array = FromTensor<float>(scores, Array2Tag{});

        return std::make_unique<DenseFsaVec>(shape, scores_array);
      }),
      py::arg("scores"), py::arg("row_splits"));

  pyclass.def(
      "dim0", [](PyClass &self) -> int32_t { return self.shape.Dim0(); },
      "Returns number of supervisions contained in it");

  // the `to_str` method is for debugging only
  pyclass.def("to_str", [](PyClass &self) -> std::string {
    std::ostringstream os;
    os << "num_axes: " << self.shape.NumAxes() << '\n';
    os << "device_type: " << self.shape.Context()->GetDeviceType() << '\n';
    os << "row_splits1: " << self.shape.RowSplits(1) << '\n';
    os << "row_ids1: " << self.shape.RowIds(1) << '\n';
    os << "scores:" << self.scores << '\n';
    return os.str();
  });
}

static void PybindConvertDenseToFsaVec(py::module &m) {
  m.def(
      "convert_dense_to_fsa_vec",
      [](DenseFsaVec &dense_fsa_vec) -> FsaVec {
        return ConvertDenseToFsaVec(dense_fsa_vec);
      },
      py::arg("dense_fsa_vec"));
}

template <typename T>
static void PybindGetArcScores(py::module &m, const char *name) {
  m.def(
      name,
      [](FsaVec &fsas, torch::Tensor forward_scores,
         torch::Tensor backward_scores) -> torch::Tensor {
        Array1<T> forward_scores_array = FromTensor<T>(forward_scores);
        Array1<T> backward_scores_array = FromTensor<T>(backward_scores);
        Array1<T> arc_scores =
            GetArcScores<T>(fsas, forward_scores_array, backward_scores_array);
        return ToTensor(arc_scores);
      },
      py::arg("fsas"), py::arg("forward_scores"), py::arg("backward_scores"));
}

}  // namespace k2

void PybindFsa(py::module &m) {
  k2::PybindFsaUtil(m);
  k2::PybindDenseFsaVec(m);
  k2::PybindConvertDenseToFsaVec(m);
  k2::PybindFsaBasicProperties(m);
  k2::PybindGetForwardScores<float>(m, "_get_forward_scores_float");
  k2::PybindGetForwardScores<double>(m, "_get_forward_scores_double");
  k2::PybindGetBackwardScores<float>(m, "_get_backward_scores_float");
  k2::PybindGetBackwardScores<double>(m, "_get_backward_scores_double");
  k2::PybindGetTotScores<float>(m, "_get_tot_scores_float");
  k2::PybindGetTotScores<double>(m, "_get_tot_scores_double");
  k2::PybindGetArcScores<float>(m, "_get_arc_scores_float");
  k2::PybindGetArcScores<double>(m, "_get_arc_scores_double");
}
