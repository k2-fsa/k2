// k2/python/csrc/fsa_util.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa_util.h"

#include "k2/csrc/fsa_util.h"

void PybindFsaUtil(py::module &m) {
  //  m.def("string_to_fsa", &k2::StringToFsa);
  m.def("fsa_to_string", &k2::FsaToString);
}
