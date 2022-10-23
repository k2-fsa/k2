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

#include "k2/python/csrc/torch/arc.h"
#include "k2/python/csrc/torch/fsa.h"
#include "k2/python/csrc/torch/fsa_algo.h"
#include "k2/python/csrc/torch/index_add.h"
#include "k2/python/csrc/torch/index_select.h"
#include "k2/python/csrc/torch/mutual_information.h"
#include "k2/python/csrc/torch/nbest.h"
#include "k2/python/csrc/torch/ragged.h"
#include "k2/python/csrc/torch/ragged_ops.h"
#include "k2/python/csrc/torch/rnnt_decode.h"
#include "k2/python/csrc/torch/v2/k2.h"

void PybindTorch(py::module &m) {
  PybindArc(m);
  PybindFsa(m);
  PybindFsaAlgo(m);
  PybindIndexAdd(m);
  PybindIndexSelect(m);
  PybindMutualInformation(m);
  PybindNbest(m);
  PybindRagged(m);
  PybindRaggedOps(m);
  PybindRnntDecode(m);

  k2::PybindV2(m);
}

#else

void PybindTorch(py::module &) {}

#endif
