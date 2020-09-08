// k2/python/csrc/fsa_algo.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_algo.h"

#include <memory>
#include <utility>

#include "k2/csrc/old/arcsort.h"
#include "k2/csrc/old/array.h"
#include "k2/csrc/old/connect.h"
#include "k2/csrc/old/determinize.h"
#include "k2/csrc/old/determinize_impl.h"
#include "k2/csrc/old/fsa.h"
#include "k2/csrc/old/intersect.h"
#include "k2/csrc/old/rmepsilon.h"
#include "k2/csrc/old/topsort.h"
#include "k2/csrc/old/weights.h"
#include "k2/python/csrc/array.h"

void PyBindArcSort(py::module &m) {
  using PyClass = k2::ArcSorter;
  py::class_<PyClass>(m, "_ArcSorter")
      .def(py::init<const k2::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<int32_t *> *arc_map = nullptr) {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));

  m.def(
      "_arc_sort",
      [](k2::Fsa *fsa, k2::Array1<int32_t *> *arc_map = nullptr) {
        return k2::ArcSort(fsa, arc_map == nullptr ? nullptr : arc_map->data);
      },
      "in-place version of ArcSorter", py::arg("fsa"),
      py::arg("arc_map").none(true));
}

void PyBindTopSort(py::module &m) {
  using PyClass = k2::TopSorter;
  py::class_<PyClass>(m, "_TopSorter")
      .def(py::init<const k2::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<int32_t *> *state_map = nullptr) -> bool {
            return self.GetOutput(
                fsa_out, state_map == nullptr ? nullptr : state_map->data);
          },
          py::arg("fsa_out"), py::arg("state_map").none(true));
}

void PyBindConnect(py::module &m) {
  using PyClass = k2::Connection;
  py::class_<PyClass>(m, "_Connection")
      .def(py::init<const k2::Fsa &>(), py::arg("fsa_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<int32_t *> *arc_map = nullptr) -> bool {
            return self.GetOutput(fsa_out,
                                  arc_map == nullptr ? nullptr : arc_map->data);
          },
          py::arg("fsa_out"), py::arg("arc_map").none(true));
}

void PyBindIntersect(py::module &m) {
  using PyClass = k2::Intersection;
  py::class_<PyClass>(m, "_Intersection")
      .def(py::init<const k2::Fsa &, const k2::Fsa &>(), py::arg("fsa_a"),
           py::arg("fsa_b"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<int32_t *> *arc_map_a = nullptr,
             k2::Array1<int32_t *> *arc_map_b = nullptr) -> bool {
            return self.GetOutput(
                fsa_out, arc_map_a == nullptr ? nullptr : arc_map_a->data,
                arc_map_b == nullptr ? nullptr : arc_map_b->data);
          },
          py::arg("fsa_out"), py::arg("arc_map_a").none(true),
          py::arg("arc_map_b").none(true));
}

template <typename TracebackState>
void PybindDeterminizerTpl(py::module &m, const char *name) {
  using PyClass = k2::Determinizer<TracebackState>;
  py::class_<PyClass>(m, name)
      .def(py::init<const k2::WfsaWithFbWeights &, float, int64_t>(),
           py::arg("fsa_in"), py::arg("beam"), py::arg("max_step"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"),
           py::arg("arc_derivs_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<float *> *arc_weights_out,
             k2::Array2<typename TracebackState::DerivType *> *arc_derivs)
              -> float {
            return self.GetOutput(fsa_out, arc_weights_out->data, arc_derivs);
          },
          py::arg("fsa_out"), py::arg("arc_weights_out"),
          py::arg("arc_derivs"));
}

template <typename TracebackState>
void PybindEpsilonsRemoverTpl(py::module &m, const char *name) {
  using PyClass = k2::EpsilonsRemover<TracebackState>;
  py::class_<PyClass>(m, name)
      .def(py::init<const k2::WfsaWithFbWeights &, float>(), py::arg("fsa_in"),
           py::arg("beam"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"),
           py::arg("arc_derivs_size"))
      .def(
          "get_output",
          [](PyClass &self, k2::Fsa *fsa_out,
             k2::Array1<float *> *arc_weights_out,
             k2::Array2<typename TracebackState::DerivType *> *arc_derivs)
              -> void {
            return self.GetOutput(fsa_out, arc_weights_out->data, arc_derivs);
          },
          py::arg("fsa_out"), py::arg("arc_weights_out"),
          py::arg("arc_derivs"));
}

void PybindFsaAlgo(py::module &m) {
  PyBindArcSort(m);
  PyBindTopSort(m);
  PyBindConnect(m);
  PyBindIntersect(m);

  PybindDeterminizerTpl<k2::MaxTracebackState>(m, "_DeterminizerMax");
  PybindDeterminizerTpl<k2::LogSumTracebackState>(m, "_DeterminizerLogSum");

  PybindEpsilonsRemoverTpl<k2::MaxTracebackState>(m, "_EpsilonsRemoverMax");
  PybindEpsilonsRemoverTpl<k2::LogSumTracebackState>(m,
                                                     "_EpsilonsRemoverLogSum");
}
