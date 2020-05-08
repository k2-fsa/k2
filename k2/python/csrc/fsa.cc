// k2/python/csrc/fsa.cc

// Copyright (c)  2020  Fangjun Kuang

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa.h"

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"

void PybindFsa(py::module &m) {
  using k2::Arc;
  using k2::Fsa;

  py::class_<Arc>(m, "Arc")
      .def(py::init<>())
      .def_readwrite("src_state", &Arc::src_state)
      .def_readwrite("dest_state", &Arc::dest_state)
      .def_readwrite("label", &Arc::label)
      .def("__eq__",
           [](const Arc &self, const Arc &other) { return self == other; })
      .def("__lt__",
           [](const Arc &self, const Arc &other) { return self < other; });

  py::class_<Fsa>(m, "Fsa")
      .def(py::init<>())
      .def("num_states", &Fsa::NumStates)
      .def("__str__", [](const Fsa &self) { return FsaToString(self); })
      .def_readwrite("arc_indexes", &Fsa::arc_indexes)
      .def_readwrite("arcs", &Fsa::arcs);
}
