// k2/python/csrc/fsa_algo.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_algo.h"

#include <memory>
#include <utility>

#include "k2/csrc/arcsort.h"
#include "k2/csrc/array.h"
#include "k2/python/csrc/array.h"

namespace k2 {}  // namespace k2

void PyBindArcSort(py::module &m) {
  using PyClass = k2::ArcSorter;
  py::class_<PyClass>(m, "_ArcSorter")
      .def(py::init<const k2::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<int32_t *> *arc_map = nullptr) {
            self.GetOutput(fsa_out,
                           arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"),
          py::arg("arc_map") = (k2::Array1<int32_t *> *)nullptr);

  m.def(
      "_arc_sort",
      [](k2::Fsa *fsa, k2::Array1<int32_t *> *arc_map = nullptr) {
        k2::ArcSort(fsa, arc_map == nullptr ? nullptr : arc_map->data);
      },
      "in-place version of ArcSorter", py::arg("fsa"),
      py::arg("arc_map") = (k2::Array1<int32_t *> *)nullptr);
}

void PybindFsaAlgo(py::module &m) { PyBindArcSort(m); }
