// k2/python/csrc/k2.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/k2.h"

#include "k2/python/csrc/array.h"
#include "k2/python/csrc/fsa.h"
#include "k2/python/csrc/fsa_util.h"

PYBIND11_MODULE(_k2, m) {
  m.doc() = "pybind11 binding of k2";
  PybindArc(m);
  PybindArray(m);
  PybindFsa(m);
  PybindFsaUtil(m);
}
