/**
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu, Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/rm_epsilon.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/fsa_algo.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

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
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        FsaVec sorted;
        TopSort(src, &sorted, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTorch(arc_map);
        return std::make_pair(sorted, tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindLinearFsa(py::module &m) {
  m.def(
      "linear_fsa",
      [](RaggedAny &labels, py::object = py::none()) -> FsaVec {
        DeviceGuard guard(labels.any.Context());
        return LinearFsas(labels.any.Specialize<int32_t>());
      },
      py::arg("labels"), py::arg("device") = py::none());

  m.def(
      "linear_fsa",
      [](const std::vector<int32_t> &labels,
         py::object device = py::str("cpu")) -> Fsa {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        ContextPtr context = GetContext(torch::Device(device_str));
        DeviceGuard guard(context);
        Array1<int32_t> array(context, labels);
        return LinearFsa(array);  //
      },
      py::arg("labels"), py::arg("device") = py::str("cpu"));

  m.def(
      "linear_fsa",
      [](const std::vector<std::vector<int32_t>> &labels,
         py::object device = py::str("cpu")) -> FsaVec {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        ContextPtr context = GetContext(torch::Device(device_str));
        DeviceGuard guard(context);
        Ragged<int32_t> ragged = CreateRagged2<int32_t>(labels).To(context);
        return LinearFsas(ragged);
      },
      py::arg("labels"), py::arg("device") = py::str("cpu"));
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
        DeviceGuard guard(a_fsas.Context());
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

          // TODO: should perhaps just always make this false, for
          // predictability, and let the user call intersect_device
          // if they want to use sorted matching?
          bool sorted_match_a = ((properties_a & kFsaPropertiesArcSorted) != 0);
          out = IntersectDevice(
              a_fsa_vec, properties_a, b_fsa_vec, properties_b, b_to_a_map,
              need_arc_map ? &a_arc_map : nullptr,
              need_arc_map ? &b_arc_map : nullptr, sorted_match_a);
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
          a_tensor = ToTorch(a_arc_map);
          b_tensor = ToTorch(b_arc_map);
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
         bool need_arc_map = true,
         bool sorted_match_a =
             false) -> std::tuple<FsaVec, torch::optional<torch::Tensor>,
                                  torch::optional<torch::Tensor>> {
        DeviceGuard guard(a_fsas.Context());
        Array1<int32_t> a_arc_map;
        Array1<int32_t> b_arc_map;
        Array1<int32_t> b_to_a_map_array = FromTorch<int32_t>(b_to_a_map);

        FsaVec ans = IntersectDevice(
            a_fsas, properties_a, b_fsas, properties_b, b_to_a_map_array,
            need_arc_map ? &a_arc_map : nullptr,
            need_arc_map ? &b_arc_map : nullptr, sorted_match_a);
        torch::optional<torch::Tensor> a_tensor;
        torch::optional<torch::Tensor> b_tensor;
        if (need_arc_map) {
          a_tensor = ToTorch(a_arc_map);
          b_tensor = ToTorch(b_arc_map);
        }
        return std::make_tuple(ans, a_tensor, b_tensor);
      },
      py::arg("a_fsas"), py::arg("properties_a"), py::arg("b_fsas"),
      py::arg("properties_b"), py::arg("b_to_a_map"),
      py::arg("need_arc_map") = true, py::arg("sorted_match_a") = false);
}

static void PybindIntersectDensePruned(py::module &m) {
  m.def(
      "intersect_dense_pruned",
      [](FsaVec &a_fsas, DenseFsaVec &b_fsas, float search_beam,
         float output_beam, int32_t min_active_states,
         int32_t max_active_states)
          -> std::tuple<FsaVec, torch::Tensor, torch::Tensor> {
        DeviceGuard guard(a_fsas.Context());
        Array1<int32_t> arc_map_a;
        Array1<int32_t> arc_map_b;
        FsaVec out;

        IntersectDensePruned(a_fsas, b_fsas, search_beam, output_beam,
                             min_active_states, max_active_states, &out,
                             &arc_map_a, &arc_map_b);
        return std::make_tuple(out, ToTorch(arc_map_a), ToTorch(arc_map_b));
      },
      py::arg("a_fsas"), py::arg("b_fsas"), py::arg("search_beam"),
      py::arg("output_beam"), py::arg("min_active_states"),
      py::arg("max_active_states"));
}

static void PybindIntersectDense(py::module &m) {
  m.def(
      "intersect_dense",
      [](FsaVec &a_fsas, DenseFsaVec &b_fsas,
         torch::optional<torch::Tensor> a_to_b_map, float output_beam,
         int32_t max_states,
         int32_t max_arcs) -> std::tuple<FsaVec, torch::Tensor, torch::Tensor> {
        DeviceGuard guard(a_fsas.Context());
        Array1<int32_t> arc_map_a;
        Array1<int32_t> arc_map_b;
        FsaVec out;

        // the following is in case a_fsas had 2 not 3 axes.  It happens in some
        // test code, and IntersectDense() used to support it.
        FsaVec a_fsa_vec = FsaToFsaVec(a_fsas);

        Array1<int32_t> a_to_b_map_array;
        if (a_to_b_map.has_value()) {
          a_to_b_map_array = FromTorch<int32_t>(a_to_b_map.value());
        } else {
          a_to_b_map_array = Arange(a_fsa_vec.Context(), 0, a_fsa_vec.Dim0());
        }
        IntersectDense(a_fsa_vec, b_fsas, &a_to_b_map_array, output_beam,
                       max_states, max_arcs, &out, &arc_map_a, &arc_map_b);
        return std::make_tuple(out, ToTorch(arc_map_a), ToTorch(arc_map_b));
      },
      py::arg("a_fsas"), py::arg("b_fsas"), py::arg("a_to_b_map"),
      py::arg("output_beam"), py::arg("max_states") = 15000000,
      py::arg("max_arcs") = 1073741824 /* 2^30 */);
}

static void PybindConnect(py::module &m) {
  m.def(
      "connect",
      [](Fsa &src, bool need_arc_map =
                       true) -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        Fsa out;
        Connect(src, &out, need_arc_map ? &arc_map : nullptr);

        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTorch(arc_map);
        return std::make_pair(out, tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindArcSort(py::module &m) {
  m.def(
      "arc_sort",
      [](FsaOrVec &src, bool need_arc_map = true)
          -> std::pair<FsaOrVec, torch::optional<torch::Tensor>> {
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        FsaOrVec out;
        ArcSort(src, &out, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTorch(arc_map);
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
         torch::Tensor entering_arcs) -> std::pair<Fsa, RaggedAny> {
        DeviceGuard guard(fsas.Context());
        Array1<int32_t> entering_arcs_array = FromTorch<int32_t>(entering_arcs);

        Ragged<int32_t> best_path_arc_indexes =
            ShortestPath(fsas, entering_arcs_array);

        FsaVec out = FsaVecFromArcIndexes(fsas, best_path_arc_indexes);
        return std::make_pair(out, RaggedAny(best_path_arc_indexes.Generic()));
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
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        FsaOrVec out;
        AddEpsilonSelfLoops(src, &out, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTorch(arc_map);
        return std::make_pair(out, arc_map_tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindUnion(py::module &m) {
  m.def(
      "union",
      [](FsaVec &fsas, bool need_arc_map = true)
          -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        DeviceGuard guard(fsas.Context());
        Array1<int32_t> arc_map;
        Fsa out = Union(fsas, need_arc_map ? &arc_map : nullptr);

        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTorch(arc_map);
        return std::make_pair(out, arc_map_tensor);
      },
      py::arg("fsas"), py::arg("need_arc_map") = true);
}

static void PybindRemoveEpsilon(py::module &m) {
  m.def(
      "remove_epsilon_host",
      [](FsaOrVec &src) -> std::pair<FsaOrVec, RaggedAny> {
        DeviceGuard guard(src.Context());
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilonHost(src, &dest, &arc_map);
        return std::make_pair(dest, RaggedAny(arc_map.Generic()));
      },
      py::arg("src"));
  m.def(
      "remove_epsilon_device",
      [](FsaOrVec &src) -> std::pair<FsaOrVec, RaggedAny> {
        DeviceGuard guard(src.Context());
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilonDevice(src, &dest, &arc_map);
        return std::make_pair(dest, RaggedAny(arc_map.Generic()));
      },
      py::arg("src"));
  m.def(
      "remove_epsilon",
      [](FsaOrVec &src, int32_t properties) -> std::pair<FsaOrVec, RaggedAny> {
        DeviceGuard guard(src.Context());
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilon(src, properties, &dest, &arc_map);
        return std::make_pair(dest, RaggedAny(arc_map.Generic()));
      },
      py::arg("src"), py::arg("properties"));
  m.def(
      "remove_epsilon_and_add_self_loops",
      [](FsaOrVec &src, int32_t properties) -> std::pair<FsaOrVec, RaggedAny> {
        DeviceGuard guard(src.Context());
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        RemoveEpsilonAndAddSelfLoops(src, properties, &dest, &arc_map);
        return std::make_pair(dest, RaggedAny(arc_map.Generic()));
      },
      py::arg("src"), py::arg("properties"));
}

static void PybindDeterminize(py::module &m) {
  py::enum_<DeterminizeWeightPushingType>(m, "DeterminizeWeightPushingType",
                                          py::arithmetic())
      .value("kTropicalWeightPushing",
             DeterminizeWeightPushingType::kTropicalWeightPushing)
      .value("kLogWeightPushing",
             DeterminizeWeightPushingType::kLogWeightPushing)
      .value("kNoWeightPushing",
             DeterminizeWeightPushingType::kNoWeightPushing);

  m.def(
      "determinize",
      [](FsaOrVec &src, DeterminizeWeightPushingType weight_pushing_type)
          -> std::pair<FsaOrVec, RaggedAny> {
        DeviceGuard guard(src.Context());
        FsaOrVec dest;
        Ragged<int32_t> arc_map;
        Determinize(src, weight_pushing_type, &dest, &arc_map);
        return std::make_pair(dest, RaggedAny(arc_map.Generic()));
      },
      py::arg("src"), py::arg("weight_pushing_type"));
}

static void PybindClosure(py::module &m) {
  m.def(
      "closure",
      [](Fsa &src, bool need_arc_map =
                       true) -> std::pair<Fsa, torch::optional<torch::Tensor>> {
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        Fsa out = Closure(src, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTorch(arc_map);
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
        DeviceGuard guard(src.Context());
        FsaOrVec dest;
        Ragged<int32_t> dest_aux_labels;
        Array1<int32_t> arc_map;
        Invert(src, src_aux_labels, &dest, &dest_aux_labels,
               need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTorch(arc_map);
        return std::make_tuple(dest, dest_aux_labels, arc_map_tensor);
      },
      py::arg("src"), py::arg("src_aux_labels"), py::arg("need_arc_map"));
}

static void PybindRemoveEpsilonSelfLoops(py::module &m) {
  m.def(
      "remove_epsilon_self_loops",
      [](FsaOrVec &src, bool need_arc_map = true)
          -> std::pair<FsaOrVec, torch::optional<torch::Tensor>> {
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        FsaOrVec ans =
            RemoveEpsilonSelfLoops(src, need_arc_map ? &arc_map : nullptr);

        torch::optional<torch::Tensor> arc_map_tensor;
        if (need_arc_map) arc_map_tensor = ToTorch(arc_map);
        return std::make_pair(ans, arc_map_tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

/*static*/ void PybindExpandArcs(py::module &m) {
  // See doc-string below.
  m.def(
      "expand_arcs",
      [](FsaOrVec &fsas, std::vector<RaggedAny> &ragged)
          -> std::tuple<FsaOrVec, std::vector<torch::Tensor>, torch::Tensor> {
        DeviceGuard guard(fsas.Context());
        std::vector<Ragged<int32_t>> ragged_labels(ragged.size());
        int32_t ragged_labels_size = ragged_labels.size();
        for (int32_t i = 0; i != ragged_labels_size; ++i) {
          ragged_labels[i] = ragged[i].any.Specialize<int32_t>();
        }

        K2_CHECK_NE(ragged_labels_size, 0);
        K2_CHECK_LE(ragged_labels_size, 6);  // see SmallVec<...,6> below.
        ContextPtr c = fsas.Context();
        int32_t num_arcs = fsas.NumElements();
        SmallVec<int32_t *, 6> ragged_labels_row_splits, ragged_labels_data;
        for (int32_t r = 0; r < ragged_labels_size; r++) {
          K2_CHECK_EQ(ragged_labels[r].NumAxes(), 2);
          K2_CHECK_EQ(ragged_labels[r].Dim0(), num_arcs);
          ragged_labels_row_splits.data[r] =
              ragged_labels[r].RowSplits(1).Data();
          ragged_labels_data.data[r] = ragged_labels[r].values.Data();
        }

        // we'll be using the labels on the arcs of `fsas` (i.e. whether they
        // are -1 or not) to determine whether arcs are final or not.  The
        // assumption is that `fsas` is valid.
        const Arc *fsas_arcs = fsas.values.Data();

        // will be set to the maximum of 1 and the length of the i'th sub-list
        // of any of the lists in `ragged_labels` (for final-arcs where the last
        // element of a sub-list was not -1, we imagine that there was an extra
        // element of the sub-list with the value of -1).
        Array1<int32_t> combined_size(c, num_arcs + 1);
        int32_t *combined_size_data = combined_size.Data();
        K2_EVAL(
            c, num_arcs, lambda_get_combined_size, (int32_t arc_idx)->void {
              int32_t fsa_label = fsas_arcs[arc_idx].label;
              bool arc_is_final = (fsa_label == -1);
              int32_t max_num_elems = 1;
              for (int32_t r = 0; r < ragged_labels_size; r++) {
                int32_t this_label_idx0x =
                            ragged_labels_row_splits.data[r][arc_idx],
                        next_label_idx0x =
                            ragged_labels_row_splits.data[r][arc_idx + 1];
                int32_t size = next_label_idx0x - this_label_idx0x;

                // Adds an extra place for the final-arc's -1 if this is a
                // final-arc and the ragged label list did not have a -1 as its
                // last element.  We don't do a memory fetch until we know that
                // it would make a difference to the result.
                if (arc_is_final && size >= max_num_elems &&
                    ragged_labels_data.data[r][next_label_idx0x - 1] != -1)
                  max_num_elems = size + 1;
                else if (size > max_num_elems)
                  max_num_elems = size;
              }
              combined_size_data[arc_idx] = max_num_elems;
            });
        ExclusiveSum(combined_size, &combined_size);
        RaggedShape combined_shape = RaggedShape2(&combined_size, nullptr, -1);

        Array1<int32_t> fsas_arc_map, labels_arc_map;
        FsaOrVec ans =
            ExpandArcs(fsas, combined_shape, &fsas_arc_map, &labels_arc_map);

        int32_t ans_num_arcs = ans.NumElements();
        Array2<int32_t> labels(c, ragged_labels_size, ans_num_arcs);
        auto labels_acc = labels.Accessor();

        // we'll be using the labels on the returned arcs (i.e. whether they are
        // -1 or not) to determine whether arcs are final or not.  The
        // assumption is that the answer is valid; since we'll likely be
        // constructing an Fsa (i.e. a python-level Fsa) from it, the properties
        // should be checked, so if this assumption is false we'll find out
        // sooner or later.
        const Arc *ans_arcs = ans.values.Data();

        K2_CHECK_EQ(labels_arc_map.Dim(), ans_num_arcs);
        const int32_t *labels_arc_map_data = labels_arc_map.Data();
        const int32_t *combined_shape_row_ids_data =
                          combined_shape.RowIds(1).Data(),
                      *combined_shape_row_splits_data =
                          combined_shape.RowSplits(1).Data();
        K2_EVAL2(
            c, ragged_labels_size, ans_num_arcs, lambda_linearize_labels,
            (int32_t r, int32_t arc_idx)->void {
              int32_t fsa_label = ans_arcs[arc_idx].label;
              bool arc_is_final = (fsa_label == -1);
              int32_t combined_shape_idx01 = labels_arc_map_data[arc_idx];
              // The reason we can assert the following is that `combined_size`
              // has no empty sub-lists because we initialized `max_num_elems =
              // 1` when we set up those sizes.
              K2_CHECK_GE(combined_shape_idx01, 0);
              // combined_shape_idx0 is also an arc_idx012 into the *original*
              // fsas; combined_shape_idx1 is the index into the sequence of
              // ragged labels attached to that arc.
              int32_t combined_shape_idx0 =
                          combined_shape_row_ids_data[combined_shape_idx01],
                      combined_shape_idx0x =
                          combined_shape_row_splits_data[combined_shape_idx0],
                      combined_shape_idx1 =
                          combined_shape_idx01 - combined_shape_idx0x;
              K2_CHECK_GE(combined_shape_idx1, 0);
              int32_t src_idx0x =
                          ragged_labels_row_splits.data[r][combined_shape_idx0],
                      src_idx0x_next = ragged_labels_row_splits
                                           .data[r][combined_shape_idx0 + 1],
                      src_idx01 = src_idx0x + combined_shape_idx1;
              int32_t this_label;
              if (src_idx01 >= src_idx0x_next) {
                // We were past the end of the source sub-list of ragged labels.
                this_label = 0;
              } else {
                this_label = ragged_labels_data.data[r][src_idx01];
              }
              if (this_label == -1 || this_label == 0)
                this_label = (arc_is_final ? -1 : 0);

              if (arc_is_final) {
                // In positions where the source FSA has label -1 (which should
                // be final-arcs), the ragged labels should have label -1.  If
                // this fails it will be because final-arcs had labels that were
                // neither -1 or 0.  If this becomes a problem in future we may
                // have to revisit this.
                K2_CHECK_EQ(this_label, fsa_label);
              }
              labels_acc(r, arc_idx) = this_label;
            });

        std::vector<torch::Tensor> ans_labels(ragged_labels_size);
        for (int32_t r = 0; r < ragged_labels_size; r++) {
          Array1<int32_t> labels_row = labels.Row(r);
          ans_labels[r] = ToTorch(labels_row);
        }

        return std::make_tuple(ans, ans_labels, ToTorch(fsas_arc_map));
      },
      py::arg("fsas"), py::arg("ragged_labels"),
      R"(
    This function expands the arcs in an Fsa or FsaVec so that we can
    turn a list of attributes stored as ragged tensors into normal, linear
    tensors.  It does this by expanding arcs into linear chains of arcs.

   Args:
       fsas:   The Fsa or FsaVec (ragged tensor of arcs with 2 or 3 axes)
           whose structure we want to copy and possibly expand chains of arcs
       ragged_labels:   A list of at least one ragged tensor of
           ints; must satisfy ragged_labels[i].NumAxes() == 2
           and ragged_labels[i].Dim0() == fsas.NumElements(),
           i.e. one sub-list per arc in the input FSAs
   Returns:   A triplet (ans_fsas, ans_label_list, arc_map), where:
             ans_fsas is the possibly-modified arcs,
             ans_label_list is a list of torch::Tensor representing
             the linearized form of `ragged_labels`
             arc_map is the map from arcs in `ans_fsas` to arcs
             in `fsas` where the score came from, or -1 in positions
             for newly-created arcs

    Caution: the behavior of this function w.r.t. final-arcs and -1's in ragged
    labels is a little complicated.  We ensure that in the linearized labels,
    all final-arcs have a label of -1 (we turn final-arcs into longer sequences
    if necessary to ensure this); and we ensure that no other arcs have -1's
    (we turn -1's into 0 to ensure this).
     )");
}

static void PybindFixFinalLabels(py::module &m) {
  // See doc-string below.
  m.def(
      "fix_final_labels",
      [](FsaOrVec &fsas, torch::optional<torch::Tensor> labels) -> void {
        DeviceGuard guard(fsas.Context());
        if (labels.has_value()) {
          Array1<int32_t> labels_array = FromTorch<int32_t>(labels.value());
          K2_CHECK_EQ(labels_array.Dim(), fsas.NumElements());
          K2_CHECK(fsas.Context()->IsCompatible(*labels_array.Context()));
          FixFinalLabels(fsas, labels_array.Data(), 1);
        } else {
          // `label` is the 3rd field of struct Arc.
          FixFinalLabels(
              fsas, reinterpret_cast<int32_t *>(fsas.values.Data()) + 2, 4);
        }
      },
      py::arg("fsas"), py::arg("labels"),
      R"(
       This function modifies, in-place, labels attached to arcs, so
      that they satisfy constraints on the placement of -1's: namely,
      that arcs to final-states must have -1's as their label, and
      that no other arcs can have -1 as their label.

       fsas: the FSA whose labels we want to modify
      labels: if supplied, must be a tensor of int32 with shape
            equal to (fsas.Dim0(),); and in this case, these labels
            will be modified.  If not supplied, the labels on the arcs
            of `fsas` will be modified.
     )");
}

static void PybindReplaceFsa(py::module &m) {
  m.def(
      "replace_fsa",
      [](FsaVec &src, FsaOrVec &index, int32_t symbol_begin_range)
          -> std::tuple<FsaOrVec, torch::optional<torch::Tensor>,
                        torch::optional<torch::Tensor>> {
        DeviceGuard guard(index.Context());
        Array1<int32_t> arc_map_src, arc_map_index;
        FsaOrVec out = ReplaceFsa(src, index, symbol_begin_range, &arc_map_src,
                                  &arc_map_index);
        torch::optional<torch::Tensor> src_map_tensor, index_map_tensor;
        src_map_tensor = ToTorch(arc_map_src);
        index_map_tensor = ToTorch(arc_map_index);
        return std::make_tuple(out, src_map_tensor, index_map_tensor);
      },
      py::arg("src"), py::arg("index"), py::arg("symbol_begin_range"));
}

static void PybindCtcGraph(py::module &m) {
  m.def(
      "ctc_graph",
      [](RaggedAny &symbols,
         bool modified = false) -> std::pair<FsaVec, torch::Tensor> {
        DeviceGuard guard(symbols.any.Context());
        Array1<int32_t> aux_labels;
        FsaVec graph =
            CtcGraphs(symbols.any.Specialize<int32_t>(), modified, &aux_labels);
        torch::Tensor tensor = ToTorch(aux_labels);
        return std::make_pair(graph, tensor);
      },
      py::arg("symbols"), py::arg("modified") = false);
}

static void PybindCtcTopo(py::module &m) {
  m.def(
      "ctc_topo",
      [](int32_t max_token, py::object device = py::str("cpu"),
         bool modified = false) -> std::pair<Fsa, torch::Tensor> {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        ContextPtr context = GetContext(torch::Device(device_str));
        DeviceGuard guard(context);
        Array1<int32_t> aux_labels;
        Fsa fsa = CtcTopo(context, max_token, modified, &aux_labels);
        torch::Tensor tensor = ToTorch(aux_labels);
        return std::make_pair(fsa, tensor);
      },
      py::arg("max_token"), py::arg("device") = py::str("cpu"),
      py::arg("modified") = false);
}

static void PybindTrivialGraph(py::module &m) {
  m.def(
      "trivial_graph",
      [](int32_t max_token,
         py::object device = py::str("cpu")) -> std::pair<Fsa, torch::Tensor> {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        ContextPtr context = GetContext(torch::Device(device_str));
        DeviceGuard guard(context);
        Array1<int32_t> aux_labels;
        Fsa fsa = TrivialGraph(context, max_token, &aux_labels);
        torch::Tensor tensor = ToTorch(aux_labels);
        return std::make_pair(fsa, tensor);
      },
      py::arg("max_token"), py::arg("device") = py::str("cpu"));
}

static void PybindLevenshteinGraph(py::module &m) {
  m.def(
      "levenshtein_graph",
      [](RaggedAny &symbols, float ins_del_score = -0.501,
         bool need_score_offset = true)
          -> std::tuple<FsaVec, torch::Tensor, torch::optional<torch::Tensor>> {
        DeviceGuard guard(symbols.any.Context());
        Array1<int32_t> aux_labels;
        Array1<float> score_offsets;
        FsaVec graph = LevenshteinGraphs(
            symbols.any.Specialize<int32_t>(), ins_del_score, &aux_labels,
            need_score_offset ? &score_offsets : nullptr);
        torch::Tensor aux_labels_tensor = ToTorch(aux_labels);
        torch::optional<torch::Tensor> score_offsets_tensor;
        if (need_score_offset) score_offsets_tensor = ToTorch(score_offsets);
        return std::make_tuple(graph, aux_labels_tensor, score_offsets_tensor);
      },
      py::arg("symbols"), py::arg("ins_del_score") = -0.501,
      py::arg("need_score_offset") = true);
}

static void PybindReverse(py::module &m) {
  // if need arc_map is true, it returns (reversed_fsa_vec, arc_map);
  // otherwise, it returns (reversed_fsa_vec, None).
  m.def(
      "reverse",
      [](FsaVec &src, bool need_arc_map = true)
          -> std::pair<FsaVec, torch::optional<torch::Tensor>> {
        DeviceGuard guard(src.Context());
        Array1<int32_t> arc_map;
        FsaVec reversed;
        Reverse(src, &reversed, need_arc_map ? &arc_map : nullptr);
        torch::optional<torch::Tensor> tensor;
        if (need_arc_map) tensor = ToTorch(arc_map);
        return std::make_pair(reversed, tensor);
      },
      py::arg("src"), py::arg("need_arc_map") = true);
}

static void PybindGenerateDenominatorLattice(py::module &m) {
  m.def(
      "generate_denominator_lattice",
      [](RaggedAny &sampled_paths, RaggedAny &frame_ids,
         RaggedAny &left_symbols, RaggedAny &sampling_probs,
         int32_t vocab_size, int32_t context_size)
      -> std::pair<FsaVec, torch::Tensor> {
        DeviceGuard guard(sampled_paths.any.Context());
        Array1<int32_t> arc_map;
        FsaVec lattice = GenerateDenominatorLattice(
            sampled_paths.any.Specialize<int32_t>(),
            frame_ids.any.Specialize<int32_t>(),
            left_symbols.any.Specialize<int32_t>(),
            sampling_probs.any.Specialize<float>(),
            vocab_size, context_size, &arc_map);
        auto arc_map_tensor = ToTorch(arc_map);
        return std::make_pair(lattice, arc_map_tensor);
      },
      py::arg("sampled_paths"), py::arg("frame_ids"), py::arg("left_symbols"),
      py::arg("sampling_probs"), py::arg("vocab_size"),
      py::arg("context_size"));
}

}  // namespace k2

void PybindFsaAlgo(py::module &m) {
  k2::PybindAddEpsilonSelfLoops(m);
  k2::PybindArcSort(m);
  k2::PybindClosure(m);
  k2::PybindConnect(m);
  k2::PybindCtcGraph(m);
  k2::PybindCtcTopo(m);
  k2::PybindDeterminize(m);
  k2::PybindExpandArcs(m);
  k2::PybindFixFinalLabels(m);
  k2::PybindGenerateDenominatorLattice(m);
  k2::PybindIntersect(m);
  k2::PybindIntersectDense(m);
  k2::PybindIntersectDensePruned(m);
  k2::PybindIntersectDevice(m);
  k2::PybindInvert(m);
  k2::PybindLevenshteinGraph(m);
  k2::PybindLinearFsa(m);
  k2::PybindRemoveEpsilon(m);
  k2::PybindRemoveEpsilonSelfLoops(m);
  k2::PybindReplaceFsa(m);
  k2::PybindReverse(m);
  k2::PybindShortestPath(m);
  k2::PybindTopSort(m);
  k2::PybindTrivialGraph(m);
  k2::PybindUnion(m);
}
