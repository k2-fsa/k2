// k2/python/csrc/fsa.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa.h"

#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"

void PybindFsa(py::module &m) {
  using k2::Arc;
  using k2::Fsa;

  py::class_<Arc>(m, "Arc")
      .def(py::init<>())
      .def(py::init<int32_t, int32_t, int32_t>(), py::arg("src_state"),
           py::arg("dest_state"), py::arg("label"))
      .def_readwrite("src_state", &Arc::src_state)
      .def_readwrite("dest_state", &Arc::dest_state)
      .def_readwrite("label", &Arc::label)
      .def("__str__", [](const Arc &self) {
        std::ostringstream os;
        os << self;
        return os.str();
      });

  py::class_<Fsa>(m, "Fsa")
      .def(py::init<>())
      .def("num_states", &Fsa::NumStates)
      .def("final_state", &Fsa::FinalState)
      .def("__str__", [](const Fsa &self) { return FsaToString(self); })
      .def_readwrite("arc_indexes", &Fsa::arc_indexes)
      .def_readwrite("arcs", &Fsa::arcs);

  py::class_<std::vector<Fsa>>(m, "FsaVec")
      .def(py::init<>())
      .def("clear", &std::vector<Fsa>::clear)
      .def("__len__", [](const std::vector<Fsa> &self) { return self.size(); })
      .def("push_back",
           [](std::vector<Fsa> *self, const Fsa &fsa) { self->push_back(fsa); })
      .def("__iter__",
           [](const std::vector<Fsa> &self) {
             return py::make_iterator(self.begin(), self.end());
           },
           py::keep_alive<0, 1>());
  // py::keep_alive<Nurse, Patient>
  // 0 is the return value and 1 is the first argument.
  // Keep the patient (i.e., `self`) alive as long as the Nurse (i.e., the
  // return value) is not freed.

  py::class_<std::vector<Arc>>(m, "ArcVec")
      .def(py::init<>())
      .def("clear", &std::vector<Arc>::clear)
      .def("__len__", [](const std::vector<Arc> &self) { return self.size(); })
      .def("__iter__",
           [](const std::vector<Arc> &self) {
             return py::make_iterator(self.begin(), self.end());
           },
           py::keep_alive<0, 1>());

  using k2::Cfsa;
  py::class_<Cfsa>(m, "Cfsa")
      .def(py::init<>())
      .def(py::init<const Fsa &>(), py::arg("fsa"))
      .def("num_states", &Cfsa::NumStates)
      .def("num_arcs", &Cfsa::NumArcs)
      .def("arc",
           [](Cfsa *self, int s) {
             DCHECK_GE(s, 0);
             DCHECK_LT(s, self->NumStates());
             auto begin = self->arc_indexes[s];
             auto end = self->arc_indexes[s + 1];
             return py::make_iterator(self->arcs + begin, self->arcs + end);
           },
           py::keep_alive<0, 1>())
      .def("__str__", [](const Cfsa &self) {
        std::ostringstream os;
        os << self;
        return os.str();
      });
}
