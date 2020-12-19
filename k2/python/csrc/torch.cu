/**
 * @brief Everything related to PyTorch for k2 Python wrappers.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/python/csrc/torch.h"

#if defined(K2_USE_PYTORCH)

#include "k2/python/csrc/torch/arc.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/fsa_algo.h"
#include "k2/python/csrc/torch/index_add.h"
#include "k2/python/csrc/torch/index_select.h"
#include "k2/python/csrc/torch/ragged.h"
#include "k2/python/csrc/torch/ragged_ops.h"

void PybindTorch(py::module &m) {
  PybindArc(m);
  PybindRagged(m);
  PybindRaggedOps(m);
  PybindFsa(m);
  PybindFsaAlgo(m);
  PybindIndexAdd(m);
  PybindIndexSelect(m);
}

#else

void PybindTorch(py::module &) {}

#endif
