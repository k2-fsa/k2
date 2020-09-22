/**
 * @brief python wrappers for PyTorch.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/python/csrc/torch.h"

#if defined(K2_USE_PYTORCH)

#include "k2/python/csrc/torch/array.h"

void PybindTorch(py::module &m) { PybindArray(m); }

#else

void PybindTorch(py::module &) {}

#endif
