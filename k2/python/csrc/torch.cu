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
#include "k2/python/csrc/torch/array.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/ragged.h"

void PybindTorch(py::module &m) {
  PybindArc(m);
  PybindArray(m);
  PybindRagged(m);
  PybindFsa(m);
}

#else

void PybindTorch(py::module &) {}

#endif
