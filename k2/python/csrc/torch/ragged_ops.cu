/**
 * @brief python wrappers for ragged_ops.h
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/ragged_ops.h"
#include "k2/python/csrc/torch/ragged_ops.h"

namespace k2 {


template <typename T>
static void PybindRemoveAxis(py::module &m, const char *name) {
  m.def(name, &RemoveAxis<T>, py::arg("src"), py::arg("axis"));
}

template <typename T>
static void PybindRemoveValuesLeq(py::module &m, const char *name) {
  m.def(name, &RemoveValuesLeq<T>, py::arg("src"), py::arg("cutoff"));
}

template <typename T>
static void PybindRemoveValuesEq(py::module &m, const char *name) {
  m.def(name, &RemoveValuesEq<T>, py::arg("src"), py::arg("target"));
}


// Recursive implementation function used inside PybindToLists().
// Returns a list containing elements `begin` through `end-1` on
// axis `axis` of `r`, with 0 <= axis < r.NumAxes(), and
// 0 <= begin <= end <= r.TotSize(axis).
py::list RaggedInt32ToList(Ragged<int32_t> &r, int32_t axis,
                           int32_t begin, int32_t end) {
  K2_CHECK_LT(static_cast<uint32_t>(axis),
              static_cast<uint32_t>(r.NumAxes()));
  K2_CHECK_LE(end, r.TotSize(axis));
  py::list ans(end - begin);
  int32_t num_axes = r.NumAxes();
  int32_t *data;
  if (axis == num_axes - 1) data = r.values.Data();
  else data = r.RowSplits(axis + 1).Data();
  for (int32_t i = begin; i < end; i++) {
    if (axis == num_axes - 1) {
      ans[i - begin] = data[i];
    } else {
      int32_t row_begin = data[i],
                row_end = data[i + 1];
      ans[i - begin] = RaggedInt32ToList(r, axis + 1, row_begin, row_end);
    }
  }
  return ans;
};

static void PybindToLists(py::module &m, const char *name) {
  m.def(name, [](Ragged<int32_t> &src) -> py::list {
      Ragged<int32_t> r = src.To(GetCpuContext());
      return RaggedInt32ToList(r, 0, 0, r.Dim0());
    }, py::arg("src"));
}


static void PybindRaggedOpsImpl(py::module &m) {
  PybindRemoveAxis<int32_t>(m, "remove_axis");
  PybindRemoveValuesLeq<int32_t>(m, "remove_values_leq");
  PybindRemoveValuesEq<int32_t>(m, "remove_values_eq");
  PybindToLists(m, "to_list");
}

}  // namespace k2

void PybindRaggedOps(py::module &m) { k2::PybindRaggedOpsImpl(m); }
