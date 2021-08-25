/**
 * @brief Everything related to PyTorch for k2 Python wrappers.
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

#include "k2/python/csrc/torch.h"

#if defined(K2_USE_PYTORCH)

#include "k2/python/csrc/torch/any.h"
#include "k2/python/csrc/torch/arc.h"
#include "k2/python/csrc/torch/discounted_cum_sum.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/fsa_algo.h"
#include "k2/python/csrc/torch/index_add.h"
#include "k2/python/csrc/torch/index_select.h"
#include "k2/python/csrc/torch/nbest.h"
#include "k2/python/csrc/torch/ragged.h"
#include "k2/python/csrc/torch/ragged_ops.h"

void PybindTorch(py::module &m) {
  PybindArc(m);
  PybindDiscountedCumSum(m);
  PybindFsa(m);
  PybindFsaAlgo(m);
  PybindIndexAdd(m);
  PybindIndexSelect(m);
  PybindNbest(m);
  PybindRagged(m);
  PybindRaggedOps(m);
  // TODO: Move Pybind* to the namespace k2
  k2::PybindAny(m);
}

#else

void PybindTorch(py::module &) {}

#endif
