// k2/python/csrc/properties.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/properties.h"

#include <vector>

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/properties.h"
#include "k2/python/host/csrc/array.h"

// We would never pass `order` parameter to k2host::IsAcyclic in Python code.
// We can make it accept `None` with `std::optional` in pybind11, but
// that will require C++17, so we here choose to write a version without
// `order`.
static bool IsAcyclic(const k2host::Fsa &fsa) {
  return k2host::IsAcyclic(fsa /*, std::vector<int32_t>* order = nullptr*/);
}

void PybindProperties(py::module &m) {
  m.def("_is_valid", &k2host::IsValid, py::arg("fsa"));
  m.def("_is_top_sorted", &k2host::IsTopSorted, py::arg("fsa"));
  m.def("_is_arc_sorted", &k2host::IsArcSorted, py::arg("fsa"));
  m.def("_has_self_loops", &k2host::HasSelfLoops, py::arg("fsa"));
  m.def("_is_acyclic", &IsAcyclic, py::arg("fsa"));
  m.def("_is_deterministic", &k2host::IsDeterministic, py::arg("fsa"));
  m.def("_is_epsilon_free", &k2host::IsEpsilonFree, py::arg("fsa"));
  m.def("_is_connected", &k2host::IsConnected, py::arg("fsa"));
  m.def("_is_empty", &k2host::IsEmpty, py::arg("fsa"));
}
