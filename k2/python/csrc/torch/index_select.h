/**
 * @brief Index select for k2.
 *
 * Unlike torch.index_select, when an entry is -1, it sets
 * the destination entry to 0.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_TORCH_INDEX_SELECT_H_
#define K2_PYTHON_CSRC_TORCH_INDEX_SELECT_H_

#include "k2/python/csrc/k2.h"

void PybindIndexSelect(py::module &m);

#endif  //  K2_PYTHON_CSRC_TORCH_INDEX_SELECT_H_
