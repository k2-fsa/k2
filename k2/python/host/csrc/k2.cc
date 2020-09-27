// k2/python/host/csrc/k2.cc

// Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/k2.h"

#include "k2/python/host/csrc/array.h"
#include "k2/python/host/csrc/aux_labels.h"
#include "k2/python/host/csrc/fsa.h"
#include "k2/python/host/csrc/fsa_algo.h"
#include "k2/python/host/csrc/fsa_equivalent.h"
#include "k2/python/host/csrc/fsa_util.h"
#include "k2/python/host/csrc/properties.h"
#include "k2/python/host/csrc/weights.h"

PYBIND11_MODULE(_k2host, m) {
  m.doc() = "pybind11 binding of k2host";
  PybindArc(m);
  PybindArray(m);
  PybindArray2Size(m);
  PybindFsa(m);
  PybindFsaUtil(m);
  PybindFsaAlgo(m);
  PybindFsaEquivalent(m);
  PybindProperties(m);
  PybindAuxLabels(m);
  PybindWeights(m);
}
