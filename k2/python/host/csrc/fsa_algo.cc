// k2/python/host/csrc/fsa_algo.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/fsa_algo.h"

#include <memory>
#include <utility>

#include "k2/csrc/host/arcsort.h"
#include "k2/csrc/host/array.h"
#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/determinize.h"
#include "k2/csrc/host/determinize_impl.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/intersect.h"
#include "k2/csrc/host/rmepsilon.h"
#include "k2/csrc/host/topsort.h"
#include "k2/csrc/host/weights.h"
#include "k2/python/host/csrc/array.h"

void PyBindArcSort(py::module &m) {
  using PyClass = k2host::ArcSorter;
  py::class_<PyClass>(m, "_ArcSorter")
      .def(py::init<const k2host::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array1<int32_t *> *arc_map = nullptr) {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));

  m.def(
      "_arc_sort",
      [](k2host::Fsa *fsa, k2host::Array1<int32_t *> *arc_map = nullptr) {
        return k2host::ArcSort(fsa,
                               arc_map == nullptr ? nullptr : arc_map->data);
      },
      "in-place version of ArcSorter", py::arg("fsa"),
      py::arg("arc_map").none(true));
}

void PyBindTopSort(py::module &m) {
  using PyClass = k2host::TopSorter;
  py::class_<PyClass>(m, "_TopSorter")
      .def(py::init<const k2host::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array1<int32_t *> *arc_map = nullptr) -> bool {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));
}

void PyBindConnect(py::module &m) {
  using PyClass = k2host::Connection;
  py::class_<PyClass>(m, "_Connection")
      .def(py::init<const k2host::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array1<int32_t *> *arc_map = nullptr) -> bool {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));
}

void PyBindIntersect(py::module &m) {
  using PyClass = k2host::Intersection;
  py::class_<PyClass>(m, "_Intersection")
      .def(py::init<const k2host::Fsa &, const k2host::Fsa &>(),
           py::arg("fsa_a"), py::arg("fsa_b"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array1<int32_t *> *arc_map_a = nullptr,
             k2host::Array1<int32_t *> *arc_map_b = nullptr) -> bool {
            return self.GetOutput(
                fsa_out, arc_map_a == nullptr ? nullptr : arc_map_a->data,
                arc_map_b == nullptr ? nullptr : arc_map_b->data);
          },
          py::arg("fsa_out"), py::arg("arc_map_a").none(true),
          py::arg("arc_map_b").none(true));
}

template <typename TracebackState>
void PybindDeterminizerPrunedTpl(py::module &m, const char *name) {
  using PyClass = k2host::DeterminizerPruned<TracebackState>;
  py::class_<PyClass>(m, name)
      .def(py::init<const k2host::WfsaWithFbWeights &, float, int64_t>(),
           py::arg("fsa_in"), py::arg("beam"), py::arg("max_step"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"),
           py::arg("arc_derivs_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array2<typename TracebackState::DerivType *> *arc_derivs)
              -> float { return self.GetOutput(fsa_out, arc_derivs); },
          py::arg("fsa_out"), py::arg("arc_derivs"));
}

template <typename TracebackState>
void PybindEpsilonsRemoverPrunedTpl(py::module &m, const char *name) {
  using PyClass = k2host::EpsilonsRemoverPruned<TracebackState>;
  py::class_<PyClass>(m, name)
      .def(py::init<const k2host::WfsaWithFbWeights &, float>(),
           py::arg("fsa_in"), py::arg("beam"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"),
           py::arg("arc_derivs_size"))
      .def(
          "get_output",
          [](PyClass &self, k2host::Fsa *fsa_out,
             k2host::Array2<typename TracebackState::DerivType *> *arc_derivs)
              -> void { return self.GetOutput(fsa_out, arc_derivs); },
          py::arg("fsa_out"), py::arg("arc_derivs"));
}

void PybindFsaAlgo(py::module &m) {
  PyBindArcSort(m);
  PyBindTopSort(m);
  PyBindConnect(m);
  PyBindIntersect(m);

  PybindDeterminizerPrunedTpl<k2host::MaxTracebackState>(
      m, "_DeterminizerPrunedMax");
  PybindDeterminizerPrunedTpl<k2host::LogSumTracebackState>(
      m, "_DeterminizerPrunedLogSum");

  PybindEpsilonsRemoverPrunedTpl<k2host::MaxTracebackState>(
      m, "_EpsilonsRemoverPrunedMax");
  PybindEpsilonsRemoverPrunedTpl<k2host::LogSumTracebackState>(
      m, "_EpsilonsRemoverPrunedLogSum");
}
