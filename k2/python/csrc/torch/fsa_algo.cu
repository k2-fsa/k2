/**
 * @brief python wrappers for fsa_algo.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
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

}  // namespace k2

void PybindFsaAlgo(py::module &m) { k2::PybindTopSort(m); }
