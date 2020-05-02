// k2/python/csrc/k2.cc

// Copyright (c)  2020  Fangjun Kuang

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/k2.h"

#include "k2/python/csrc/fsa.h"
#include "k2/python/csrc/fsa_renderer.h"
#include "k2/python/csrc/fsa_util.h"

PYBIND11_MODULE(k2, m) {
  m.doc() = "pybind11 binding of k2";
  pybind_fsa(m);
  pybind_fsa_renderer(m);
  pybind_fsa_util(m);
}
