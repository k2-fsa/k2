// k2/python/host/csrc/fsa_util.cc

// Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/fsa_util.h"

#include "k2/csrc/host/fsa_util.h"

void PybindFsaUtil(py::module &m) {
  m.def("fsa_to_str", &k2host::FsaToString, py::arg("fsa"));
}
