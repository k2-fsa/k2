/**
 * @brief Everything related to PyTorch for k2 Python wrappers.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_TORCH_H_
#define K2_PYTHON_CSRC_TORCH_H_

#include "k2/python/csrc/k2.h"

void PybindTorch(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_H_
