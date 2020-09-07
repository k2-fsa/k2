// k2/python/csrc/aux_labels.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/aux_labels.h"

#include "k2/csrc/host/aux_labels.h"

void PyBindAuxLabels1Mapper(py::module &m) {
  using PyClass = k2::AuxLabels1Mapper;
  py::class_<PyClass>(m, "_AuxLabels1Mapper")
      .def(py::init<const k2::AuxLabels &, const k2::Array1<int32_t *> &>(),
           py::arg("labels_in"), py::arg("arc_map"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("aux_size"))
      .def("get_output", &PyClass::GetOutput, py::arg("labels_out"));
}

void PyBindAuxLabels2Mapper(py::module &m) {
  using PyClass = k2::AuxLabels2Mapper;
  py::class_<PyClass>(m, "_AuxLabels2Mapper")
      .def(py::init<const k2::AuxLabels &, const k2::Array2<int32_t *> &>(),
           py::arg("labels_in"), py::arg("arc_map"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("aux_size"))
      .def("get_output", &PyClass::GetOutput, py::arg("labels_out"));
}

void PyBindFstInverter(py::module &m) {
  using PyClass = k2::FstInverter;
  py::class_<PyClass>(m, "_FstInverter")
      .def(py::init<const k2::Fsa &, const k2::AuxLabels &>(),
           py::arg("fsa_in"), py::arg("labels_in"))
      .def("get_sizes", &PyClass::GetSizes, py::arg("fsa_size"),
           py::arg("aux_size"))
      .def("get_output", &PyClass::GetOutput, py::arg("fsa_out"),
           py::arg("labels_out"));
}

void PybindAuxLabels(py::module &m) {
  PyBindAuxLabels1Mapper(m);
  PyBindAuxLabels2Mapper(m);
  PyBindFstInverter(m);
}
