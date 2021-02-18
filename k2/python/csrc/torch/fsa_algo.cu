/**
 * @brief python wrappers for fsa_algo.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/rm_epsilon.h"
#include "k2/python/csrc/torch/fsa_algo.h"
#include "k2/python/csrc/torch/torch_util.h"

namespace k2 {

static void PybindTopSort(py::module &m) {
  // TODO(fangjun): add docstring for this function
  //
  // if need_arc_map is true, it returns (sorted_fsa_vec, arc_map);
  // otherwise, it returns (sorted_fsa_vec, None)
  m.def(
      "top_sort",
      [](FsaVec &src, bool need_arc_map = true)
          -> std::pair<FsaVec, torch::optional<torch::Tensor>> {
        Array1<int32_t> arc_map;
        FsaVec sorted;
        TopSort(src, &sorted, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTensor(arc_map);
        return std::make_pair(sorted, tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindLinearFsa(py::module &m) {
  m.def(
      "linear_fsa",
      [](const std::vector<int32_t> &labels, int32_t gpu_id = -1) -> Fsa {
        ContextPtr context;
        if (gpu_id < 0)
          context = GetCpuContext();
        else
          context = GetCudaContext(gpu_id);
        Array1<int32_t> array(context, labels);
        return LinearFsa(array);  //
      },
      py::arg("labels"), py::arg("gpu_id") = -1,
      R"(
  If gpu_id is -1, the returned FSA is on CPU.
  If gpu_id >= 0, the returned FSA is on the specified GPU.
  )");

  m.def(
      "linear_fsa",
      [](const std::vector<std::vector<int32_t>> &labels,
         int32_t gpu_id = -1) -> FsaVec {
        ContextPtr context;
        if (gpu_id < 0)
          context = GetCpuContext();
        else
          context = GetCudaContext(gpu_id);

        Ragged<int32_t> ragged = CreateRagged2<int32_t>(labels).To(context);
        return LinearFsas(ragged);
      },
      py::arg("labels"), py::arg("gpu_id") = -1,
      R"(
  If gpu_id is -1, the returned FsaVec is on CPU.
  If gpu_id >= 0, the returned FsaVec is on the specified GPU.
      )");

  m.def(
      "linear_fsa",
      [](const Ragged<int32_t> &labels, int32_t /*unused_gpu_id*/) -> FsaVec {
        return LinearFsas(labels);
      },
      py::arg("labels"), py::arg("gpu_id"));
}

static void PybindIntersect(py::module &m) {
  // It runs on CUDA if and only if
  //  - a_fsas is on GPU
  //  - b_fsas is on GPU
  //  - treat_epsilons_specially is False
  //
  // Otherwise, it is run on CPU.
  m.def(
      "intersect",
      [](FsaOrVec &a_fsas, int32_t properties_a, FsaOrVec &b_fsas,
         int32_t properties_b, bool treat_epsilons_specially = true,
         bool need_arc_map =
             true) -> std::tuple<FsaOrVec, torch::optional<torch::Tensor>,
                                 torch::optional<torch::Tensor>> {
        Array1<int32_t> a_arc_map;
        Array1<int32_t> b_arc_map;
        FsaVec out;
        if (!treat_epsilons_specially &&
            a_fsas.Context()->GetDeviceType() == kCuda) {
          FsaVec a_fsa_vec = FsaToFsaVec(a_fsas);
          FsaVec b_fsa_vec = FsaToFsaVec(b_fsas);
          std::vector<int32_t> tmp_b_to_a_map(b_fsa_vec.Dim0());
          if (a_fsa_vec.Dim0() == 1) {
            std::fill(tmp_b_to_a_map.begin(), tmp_b_to_a_map.end(), 0);
          } else {
            std::iota(tmp_b_to_a_map.begin(), tmp_b_to_a_map.end(), 0);
          }
          Array1<int32_t> b_to_a_map(a_fsa_vec.Context(), tmp_b_to_a_map);

          out =
              IntersectDevice(a_fsa_vec, properties_a, b_fsa_vec, properties_b,
                              b_to_a_map, need_arc_map ? &a_arc_map : nullptr,
                              need_arc_map ? &b_arc_map : nullptr);
        } else {
          Intersect(a_fsas, properties_a, b_fsas, properties_b,
                    treat_epsilons_specially, &out,
                    need_arc_map ? &a_arc_map : nullptr,
                    need_arc_map ? &b_arc_map : nullptr);
        }
        FsaOrVec ans;
        if (a_fsas.NumAxes() == 2 && b_fsas.NumAxes() == 2)
          ans = GetFsaVecElement(out, 0);
        else
          ans = out;
        torch::optional<torch::Tensor> a_tensor;
        torch::optional<torch::Tensor> b_tensor;
        if (need_arc_map) {
          a_tensor = ToTensor(a_arc_map);
          b_tensor = ToTensor(b_arc_map);
        }
        return std::make_tuple(ans, a_tensor, b_tensor);
      },
      py::arg("a_fsas"), py::arg("properties_a"), py::arg("b_fsas"),
      py::arg("properties_b"), py::arg("treat_epsilons_specially") = true,
      py::arg("need_arc_map") = true,
      R"(
      If treat_epsilons_specially it will treat epsilons as epsilons; otherwise
      it will treat them as a real symbol.

      If need_arc_map is true, it returns a tuple (fsa_vec, a_arc_map, b_arc_map);
      If need_arc_map is false, it returns a tuple (fsa_vec, None, None).

      a_arc_map maps arc indexes of the returned fsa to the input a_fsas.
      )");
}

static void PybindIntersectDevice(py::module &m) {
  // It works on both GPU and CPU.
  // But it is super slow on CPU.
  // Do not use this one for CPU; use `Intersect` for CPU.
  m.def(
      "intersect_device",
      [](FsaVec &a_fsas, int32_t properties_a, FsaVec &b_fsas,
         int32_t properties_b, torch::Tensor b_to_a_map,
         bool need_arc_map =
             true) -> std::tuple<FsaVec, torch::optional<torch::Tensor>,
                                 torch::optional<torch::Tensor>> {
        Array1<int32_t> a_arc_map;
        Array1<int32_t> b_arc_map;
        Array1<int32_t> b_to_a_map_array = FromTensor<int32_t>(b_to_a_map);

        FsaVec ans = IntersectDevice(a_fsas, properties_a, b_fsas, properties_b,
                                     b_to_a_map_array,
                                     need_arc_map ? &a_arc_map : nullptr,
                                     need_arc_map ? &b_arc_map : nullptr);
        torch::optional<torch::Tensor> a_tensor;
        torch::optional<torch::Tensor> b_tensor;
        if (need_arc_map) {
          a_tensor = ToTensor(a_arc_map);
          b_tensor = ToTensor(b_arc_map);
        }
        return std::make_tuple(ans, a_tensor, b_tensor);
      },
      py::arg("a_fsas"), py::arg("properties_a"), py::arg("b_fsas"),
      py::arg("properties_b"), py::arg("b_to_a_map"),
      py::arg("need_arc_map") = true);
}

static void PybindIntersectDensePruned(py::module &m) {
  m.def(
      "intersect_dense_pruned",
      [](FsaVec &a_fsas, DenseFsaVec &b_fsas, float search_beam,
         float output_beam, int32_t min_active_states,
         int32_t max_active_states)
          -> std::tuple<FsaVec, torch::Tensor, torch::Tensor> {
        Array1<int32_t> arc_map_a;
        Array1<int32_t> arc_map_b;
        FsaVec out;

        IntersectDensePruned(a_fsas, b_fsas, search_beam, output_beam,
                             min_active_states, max_active_states, &out,
                             &arc_map_a, &arc_map_b);
        return std::make_tuple(out, ToTensor(arc_map_a), ToTensor(arc_map_b));
      },
      py::arg("a_fsas"), py::arg("b_fsas"), py::arg("search_beam"),
      py::arg("output_beam"), py::arg("min_active_states"),
      py::arg("max_active_states"));
}

static void PybindIntersectDense(py::module &m) {
  m.def(
      "intersect_dense",
      [](FsaVec &a_fsas, DenseFsaVec &b_fsas,
         torch::optional<torch::Tensor> a_to_b_map, float output_beam)
          -> std::tuple<FsaVec, torch::Tensor, torch::Tensor> {
        Array1<int32_t> arc_map_a;
        Array1<int32_t> arc_map_b;
        FsaVec out;

        // the following is in case a_fsas had 2 not 3 axes.  It happens in some
        // test code, and IntersectDense() used to support it.
        FsaVec a_fsa_vec = FsaToFsaVec(a_fsas);

        Array1<int32_t> a_to_b_map_array;
        if (a_to_b_map.has_value()) {
          a_to_b_map_array = FromTensor<int32_t>(a_to_b_map.value());
        } else {
          a_to_b_map_array = Arange(a_fsa_vec.Context(), 0, a_fsa_vec.Dim0());
        }
        IntersectDense(a_fsa_vec, b_fsas, &a_to_b_map_array, output_beam, &out,
                       &arc_map_a, &arc_map_b);
        return std::make_tuple(out, ToTensor(arc_map_a), ToTensor(arc_map_b));
      },
      py::arg("a_fsas"), py::arg("b_fsas"), py::arg("a_to_b_map"),
      py::arg("output_beam"));
}

static void PybindConnect(py::module &m) {
  m.def(
      "connect",
      [](Fsa &src, bool need_arc_map =
                       true) -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        Array1<int32_t> arc_map;
        Fsa out;
        Connect(src, &out, need_arc_map ? &arc_map : nullptr);

        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTensor(arc_map);
        return std::make_pair(out, tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindArcSort(py::module &m) {
  m.def(
      "arc_sort",
      [](FsaOrVec &src, bool need_arc_map = true)
          -> std::pair<FsaOrVec, torch::optional<torch::Tensor>> {
        Array1<int32_t> arc_map;
        FsaOrVec out;
        ArcSort(src, &out, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTensor(arc_map);
        return std::make_pair(out, tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindShortestPath(py::module &m) {
  // returns a std::pair containing the following entries (listed in order):
  //  - FsaVec
  //      contains linear FSAs of the best path of every FSA
  //  - best_path_arc_indexes
  //      a RaggedInt containing the arc indexes of the best paths
  m.def(
      "shortest_path",
      [](FsaVec &fsas,
         torch::Tensor entering_arcs) -> std::pair<Fsa, Ragged<int32_t>> {
        Array1<int32_t> entering_arcs_array =
            FromTensor<int32_t>(entering_arcs);

        Ragged<int32_t> best_path_arc_indexes =
            ShortestPath(fsas, entering_arcs_array);

        FsaVec out = FsaVecFromArcIndexes(fsas, best_path_arc_indexes);
        return std::make_pair(out, best_path_arc_indexes);
      },
      py::arg("fsas"), py::arg("entering_arcs"));
}

static void PybindAddEpsilonSelfLoops(py::module &m) {
  // Return a pair containing:
  // - FsaOrVec
  //     the output FSA
  // - arc_map
  //     a 1-D torch::Tensor of dtype torch.int32;
  //     None if `need_arc_map` is false
  m.def(
      "add_epsilon_self_loops",
      [](FsaOrVec &src, bool need_arc_map = true)
          -> std::pair<FsaOrVec, torch::optional<torch::Tensor>> {
        Array1<int32_t> arc_map;
        FsaOrVec out;
        AddEpsilonSelfLoops(src, &out, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTensor(arc_map);
        return std::make_pair(out, arc_map_tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindUnion(py::module &m) {
  m.def(
      "union",
      [](FsaVec &fsas, bool need_arc_map = true)
          -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        Array1<int32_t> arc_map;
        Fsa out = Union(fsas, need_arc_map ? &arc_map : nullptr);

        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTensor(arc_map);
        return std::make_pair(out, arc_map_tensor);
      },
      py::arg("fsas"), py::arg("need_arc_map") = true);
}

static void PybindRemoveEpsilon(py::module &m) {
  m.def(
      "remove_epsilon_host",
      [](FsaOrVec &src) -> std::pair<FsaOrVec, Ragged<int32_t>> {
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilonHost(src, &dest, &arc_map);
        return std::make_pair(dest, arc_map);
      },
      py::arg("src"));
  m.def(
      "remove_epsilon_device",
      [](FsaOrVec &src) -> std::pair<FsaOrVec, Ragged<int32_t>> {
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilonDevice(src, &dest, &arc_map);
        return std::make_pair(dest, arc_map);
      },
      py::arg("src"));
  m.def(
      "remove_epsilon",
      [](FsaOrVec &src,
         int32_t properties) -> std::pair<FsaOrVec, Ragged<int32_t>> {
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilon(src, properties, &dest, &arc_map);
        return std::make_pair(dest, arc_map);
      },
      py::arg("src"), py::arg("properties"));
  m.def(
      "remove_epsilon_and_add_self_loops",
      [](FsaOrVec &src,
         int32_t properties) -> std::pair<FsaOrVec, Ragged<int32_t>> {
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilonAndAddSelfLoops(src, properties, &dest, &arc_map);
        return std::make_pair(dest, arc_map);
      },
      py::arg("src"), py::arg("properties"));
}

static void PybindDeterminize(py::module &m) {
  m.def(
      "determinize",
      [](FsaOrVec &src) -> std::pair<FsaOrVec, Ragged<int32_t>> {
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        Determinize(src, &dest, &arc_map);
        return std::make_pair(dest, arc_map);
      },
      py::arg("src"));
}

static void PybindClosure(py::module &m) {
  m.def(
      "closure",
      [](Fsa &src, bool need_arc_map =
                       true) -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        Array1<int32_t> arc_map;
        Fsa out = Closure(src, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTensor(arc_map);
        return std::make_pair(out, arc_map_tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindInvert(py::module &m) {
  m.def(
      "invert",
      [](FsaOrVec &src, Ragged<int32_t> &src_aux_labels,
         bool need_arc_map =
             true) -> std::tuple<FsaOrVec, Ragged<int32_t>,
                                 torch::optional<torch::Tensor>> {
        FsaOrVec dest;
        Ragged<int32_t> dest_aux_labels;
        Array1<int32_t> arc_map;
        Invert(src, src_aux_labels, &dest, &dest_aux_labels,
               need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTensor(arc_map);
        return std::make_tuple(dest, dest_aux_labels, arc_map_tensor);
      },
      py::arg("src"), py::arg("src_aux_labels"), py::arg("need_arc_map"));
}

}  // namespace k2

void PybindFsaAlgo(py::module &m) {
  k2::PybindLinearFsa(m);
  k2::PybindTopSort(m);
  k2::PybindIntersect(m);
  k2::PybindIntersectDevice(m);
  k2::PybindIntersectDensePruned(m);
  k2::PybindIntersectDense(m);
  k2::PybindConnect(m);
  k2::PybindArcSort(m);
  k2::PybindShortestPath(m);
  k2::PybindAddEpsilonSelfLoops(m);
  k2::PybindUnion(m);
  k2::PybindRemoveEpsilon(m);
  k2::PybindDeterminize(m);
  k2::PybindClosure(m);
  k2::PybindInvert(m);
}
