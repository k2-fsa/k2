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
static void PybindRemoveValuesLeq(py::module &m, const char *name) {
  m.def(name, &RemoveValuesLeq<T>, py::arg("src"), py::arg("cutoff"));
}

template <typename T>
static void PybindRemoveValuesEqual(py::module &m, const char *name) {
  m.def(name, &RemoveValuesEqual<T>, py::arg("src"), py::arg("target"));
}

static void PybindRaggedOpsImpl(py::module &m) {
  PybindRemoveValuesLeq<int32_t>(m, "remove_values_leq");
  PybindRemoveValuesEqual<int32_t>(m, "remove_values_equal");
}

}  // namespace k2

void PybindRaggedOps(py::module &m) { k2::PybindRaggedOpsImpl(m); }
