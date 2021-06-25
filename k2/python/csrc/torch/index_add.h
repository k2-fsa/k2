/**
 * @brief index_add for k2.
 *
 * It has identical semantics as torch.Tensor.index_add_
 * except that it requires the dtype of the input index
 * to be torch.int32, whereas PyTorch expects the dtype to be
 * torch.int64. Furthermore, it ignores index[i] == -1.
 *
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef K2_PYTHON_CSRC_TORCH_INDEX_ADD_H_
#define K2_PYTHON_CSRC_TORCH_INDEX_ADD_H_

#include "k2/python/csrc/k2.h"

void PybindIndexAdd(py::module &m);

#endif  //  K2_PYTHON_CSRC_TORCH_INDEX_ADD_H_
