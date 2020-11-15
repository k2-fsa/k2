/**
 * @brief python wrappers for fsa_algo.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
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
  // TODO(fangjun): Replace std::vector<int32_t> with torch::Tensor
  m.def(
      "linear_fsa",
      [](const std::vector<int32_t> &symbols, int32_t gpu_id = -1) -> Fsa {
        ContextPtr context;
        if (gpu_id < 0)
          context = GetCpuContext();
        else
          context = GetCudaContext(gpu_id);
        Array1<int32_t> array(context, symbols);
        return LinearFsa(array);  //
      },
      py::arg("symbols"), py::arg("device_id") = -1,
      R"(
  If gpu_id is -1, the returned FSA is on CPU.
  If gpu_id >= 0, the returned FSA is on the specified GPU.
  )");

  m.def(
      "linear_fsa",
      [](const std::vector<std::vector<int32_t>> &symbols,
         int32_t gpu_id = -1) -> FsaVec {
        ContextPtr context;
        if (gpu_id < 0)
          context = GetCpuContext();
        else
          context = GetCudaContext(gpu_id);

        auto n = static_cast<int32_t>(symbols.size());
        std::vector<int32_t> sizes;
        sizes.reserve(n);

        std::vector<int32_t> flatten;
        for (const auto &s : symbols) {
          sizes.push_back(static_cast<int32_t>(s.size()));
          flatten.insert(flatten.end(), s.begin(), s.end());
        }
        sizes.push_back(0);  // an extra element for exclusive sum

        Array1<int32_t> row_splits(context, sizes);
        ExclusiveSum(row_splits, &row_splits);
        RaggedShape shape = RaggedShape2(&row_splits, nullptr, flatten.size());
        Array1<int32_t> values(context, flatten);
        Ragged<int32_t> ragged(shape, values);

        return LinearFsas(ragged);
      },
      py::arg("symbols"), py::arg("gpu_id") = -1,
      R"(
  If gpu_id is -1, the returned FsaVec is on CPU.
  If gpu_id >= 0, the returned FsaVec is on the specified GPU.
      )");
}

static void PybindIntersect(py::module &m) {
  m.def(
      "intersect",  // works only on CPU
      [](FsaOrVec &a_fsas, FsaOrVec &b_fsas,
         bool treat_epsilons_specially = true,
         bool need_arc_map =
             true) -> std::tuple<FsaOrVec, torch::optional<torch::Tensor>,
                                 torch::optional<torch::Tensor>> {
        Array1<int32_t> a_arc_map;
        Array1<int32_t> b_arc_map;
        FsaVec out;
        Intersect(a_fsas, b_fsas, treat_epsilons_specially, &out,
                  need_arc_map ? &a_arc_map : nullptr,
                  need_arc_map ? &b_arc_map : nullptr);
        FsaOrVec ans;
        if (out.Dim0() == 1)
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
      py::arg("a_fsas"), py::arg("b_fsas"),
      py::arg("treat_epsilons_specially") = true,
      py::arg("need_arc_map") = true,
      R"(
      If treat_epsilons_specially it will treat epsilons as epsilons; otherwise
      it will treat them as a real symbol.

      If need_arc_map is true, it returns a tuple (fsa_vec, a_arc_map, b_arc_map);
      If need_arc_map is false, it returns a tuple (fsa_vec, None, None).

      a_arc_map maps arc indexes of the returned fsa to the input a_fsas.
      )");
}

static void PybindIntersectDensePruned(py::module &m) {
  m.def(
      "intersect_dense_pruned",
      [](FsaVec &a_fsas, DenseFsaVec &b_fsas, float beam,
         int32_t max_active_states, int32_t min_active_states)
          -> std::tuple<FsaVec, torch::Tensor, torch::Tensor> {
        Array1<int32_t> arc_map_a;
        Array1<int32_t> arc_map_b;
        FsaVec out;

        IntersectDensePruned(a_fsas, b_fsas, beam, max_active_states,
                             min_active_states, &out, &arc_map_a, &arc_map_b);
        return std::make_tuple(out, ToTensor(arc_map_a), ToTensor(arc_map_b));
      },
      py::arg("a_fsas"), py::arg("b_fsas"), py::arg("beam"),
      py::arg("max_active_states"), py::arg("min_active_states"));
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

}  // namespace k2

void PybindFsaAlgo(py::module &m) {
  k2::PybindLinearFsa(m);
  k2::PybindTopSort(m);
  k2::PybindIntersect(m);
  k2::PybindIntersectDensePruned(m);
  k2::PybindConnect(m);
  k2::PybindArcSort(m);
  k2::PybindShortestPath(m);
  k2::PybindAddEpsilonSelfLoops(m);
  k2::PybindUnion(m);
}
