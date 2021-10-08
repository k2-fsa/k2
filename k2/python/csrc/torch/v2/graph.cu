/**
 * @brief Wrap k2 graphs
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
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

#include <vector>

#include "k2/csrc/fsa_algo.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/graph.h"

namespace k2 {

RaggedArc CtcTopo(int32_t max_token,
                  torch::optional<torch::Device> device /*= {}*/,
                  bool modified /*= false*/) {
  ContextPtr context = GetContext(device.value_or(torch::Device("cpu")));
  DeviceGuard guard(context);
  Array1<int32_t> aux_labels;
  Fsa fsa = k2::CtcTopo(context, max_token, modified, &aux_labels);
  return RaggedArc(fsa, ToTorch(aux_labels));
}

RaggedArc CtcGraphs(RaggedAny &symbols, bool modified /*= false*/) {
  ContextPtr context = symbols.any.Context();
  DeviceGuard guard(context);
  Array1<int32_t> aux_labels;
  Fsa fsa =
      k2::CtcGraphs(symbols.any.Specialize<int32_t>(), modified, &aux_labels);
  return RaggedArc(fsa, ToTorch(aux_labels));
}

}  // namespace k2
