/**
 * @brief python wrappers for Ragged<T>.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_TORCH_RAGGED_H_
#define K2_PYTHON_CSRC_TORCH_RAGGED_H_

#include "k2/python/csrc/k2.h"

void PybindRagged(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_RAGGED_H_
