/**
 * @brief index_add for k2.
 *
 * It has identical semantics as torch.Tensor.index_add_
 * except that it requires the dtype of the input index
 * to be torch.int32, whereas PyTorch expects the dtype to be
 * torch.int64. Furthermore, it ignores index[i] == -1.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_TORCH_INDEX_ADD_H_
#define K2_PYTHON_CSRC_TORCH_INDEX_ADD_H_

#include "k2/python/csrc/k2.h"

void PybindIndexAdd(py::module &m);

#endif  //  K2_PYTHON_CSRC_TORCH_INDEX_ADD_H_
