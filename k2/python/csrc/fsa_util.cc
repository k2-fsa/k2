// k2/python/csrc/fsa_util.cc

// Copyright (c)  2020  Fangjun Kuang

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_util.h"

#include "k2/csrc/fsa_util.h"

void pybind_fsa_util(py::module &m) {
  using namespace k2;

  m.def("string_to_fsa", &StringToFsa);
  m.def("fsa_to_string", &FsaToString);
}
