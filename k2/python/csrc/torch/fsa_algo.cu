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
      "intersect",
      [](FsaOrVec &a_fsas, FsaOrVec &b_fsas, bool need_arc_map = true)
          -> std::tuple<FsaOrVec, torch::optional<torch::Tensor>,
                        torch::optional<torch::Tensor>> {
        Array1<int32_t> a_arc_map;
        Array1<int32_t> b_arc_map;
        FsaVec out;
        Intersect(a_fsas, b_fsas, &out, need_arc_map ? &a_arc_map : nullptr,
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
      py::arg("a_fsas"), py::arg("b_fsas"), py::arg("need_arc_map") = true,
      R"(
      If need_arc_map is true, it returns a tuple (fsa_vec, a_arc_map, b_arc_map);
      If need_arc_map is false, it returns a tuple (fsa_vec, None, None).

      a_arc_map maps arc indexes of the returned fsa to the input a_fsas.
      )");
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

}  // namespace k2

void PybindFsaAlgo(py::module &m) {
  k2::PybindLinearFsa(m);
  k2::PybindTopSort(m);
  k2::PybindIntersect(m);
  k2::PybindConnect(m);
}
