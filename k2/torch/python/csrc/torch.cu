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

#include "k2/torch/python/csrc/torch.h"

#if defined(K2_USE_PYTORCH)

#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/any.h"
#include "k2/torch/python/csrc/arc.h"
#include "k2/torch/python/csrc/discounted_cum_sum.h"
#include "k2/torch/python/csrc/fsa.h"
#include "k2/torch/python/csrc/k2.h"
#include "k2/torch/python/csrc/nbest.h"
#include "k2/torch/python/csrc/ops.h"
#include "k2/torch/python/csrc/ragged_shape.h"
#include "k2/torch/python/csrc/version.h"

void PybindTorch(py::module &m) {
  k2::PybindArc(m);
  k2::PybindDiscountedCumSum(m);
  k2::PybindNbest(m);

  py::module ragged = m.def_submodule(
      "ragged", "Sub module containing operations for ragged tensors in k2");

  k2::PybindRaggedShape(ragged);

  m.attr("RaggedShape") = ragged.attr("RaggedShape");  // TODO: remove it

  k2::PybindRaggedAny(ragged);
  k2::PybindFsaClass(ragged);
  m.attr("RaggedArc") = ragged.attr("Fsa");  // TODO: remove it

  k2::PybindOps(m);
}

#else

void PybindTorch(py::module &) {}

#endif
