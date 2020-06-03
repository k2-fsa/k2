// k2/python/csrc/fsa_renderer.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_renderer.h"

#include "k2/csrc/fsa_renderer.h"

void PybindFsaRenderer(py::module &m) {
  using k2::Fsa;
  using k2::FsaRenderer;

  py::class_<FsaRenderer>(m, "FsaRenderer")
      .def(py::init<const Fsa &>(), py::arg("fsa"))
      .def("render", &FsaRenderer::Render);
}
