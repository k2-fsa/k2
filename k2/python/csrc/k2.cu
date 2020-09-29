/**
 * @brief python wrappers for k2.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/python/csrc/k2.h"

#include "k2/python/csrc/torch.h"

PYBIND11_MODULE(_k2, m) {
  m.doc() = "pybind11 binding of k2";
  PybindTorch(m);
}
