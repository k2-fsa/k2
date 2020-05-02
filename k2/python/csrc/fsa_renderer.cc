// k2/python/csrc/fsa_renderer.cc

// Copyright (c)  2020  Fangjun Kuang

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_renderer.h"

#include "k2/csrc/fsa_renderer.h"

void pybind_fsa_renderer(py::module &m) {
  using namespace k2;
  py::class_<FsaRenderer>(m, "FsaRenderer")
      .def(py::init<const Fsa &>(), py::arg("fsa"))
      .def("render", &FsaRenderer::Render);
}
