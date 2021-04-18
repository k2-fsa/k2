/**
 * @brief python wrapper for DiscountedCumSum
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_TORCH_DISCOUNTED_CUM_SUM_H_
#define K2_PYTHON_CSRC_TORCH_DISCOUNTED_CUM_SUM_H_

#include "k2/python/csrc/k2.h"

void PybindDiscountedCumSum(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_DISCOUNTED_CUM_SUM_H
