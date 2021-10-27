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
#include "k2/torch/csrc/graph.h"
#include "k2/torch/csrc/torch_utils.h"

namespace k2 {

FsaClass CtcTopo(int32_t max_token, bool modified /*= false*/,
                 torch::optional<torch::Device> device /*= {}*/) {
  ContextPtr context = GetContext(device.value_or(torch::Device(torch::kCPU)));
  DeviceGuard guard(context);
  Array1<int32_t> aux_labels;
  Fsa fsa = k2::CtcTopo(context, max_token, modified, &aux_labels);
  return FsaClass(fsa, ToTorch(aux_labels));
}

FsaClass CtcGraphs(RaggedAny &symbols, bool modified /*= false*/) {
  ContextPtr context = symbols.any.Context();
  DeviceGuard guard(context);
  Array1<int32_t> aux_labels;
  Fsa fsa =
      k2::CtcGraphs(symbols.any.Specialize<int32_t>(), modified, &aux_labels);
  return FsaClass(fsa, ToTorch(aux_labels));
}

FsaClass CtcGraphs(const std::vector<std::vector<int32_t>> &symbols,
                   bool modified /*= false*/,
                   torch::optional<torch::Device> device /*= {}*/) {
  ContextPtr context = GetContext(device.value_or(torch::Device(torch::kCPU)));
  DeviceGuard guard(context);
  Ragged<int32_t> ragged = CreateRagged2<int32_t>(symbols).To(context);
  RaggedAny ragged_any = RaggedAny(ragged.Generic());
  return CtcGraphs(ragged_any, modified);
}

FsaClass LinearFsa(RaggedAny &labels) {
  ContextPtr context = labels.any.Context();
  DeviceGuard guard(context);
  Fsa fsa = k2::LinearFsas(labels.any.Specialize<int32_t>());
  return FsaClass(fsa);
}

FsaClass LinearFsa(const std::vector<int32_t> &labels,
                   torch::optional<torch::Device> device /*= {}*/) {
  ContextPtr context = GetContext(device.value_or(torch::Device(torch::kCPU)));
  DeviceGuard guard(context);
  Array1<int32_t> array(context, labels);
  Fsa fsa = k2::LinearFsa(array);
  return FsaClass(fsa);
}

FsaClass LinearFsa(const std::vector<std::vector<int32_t>> &labels,
                   torch::optional<torch::Device> device /*= {}*/) {
  ContextPtr context = GetContext(device.value_or(torch::Device(torch::kCPU)));
  DeviceGuard guard(context);
  Ragged<int32_t> ragged = CreateRagged2<int32_t>(labels).To(context);
  Fsa fsa = k2::LinearFsas(ragged);
  return FsaClass(fsa);
}

FsaClass LevenshteinGraphs(RaggedAny &symbols,
                           float ins_del_score /*= -0.501*/) {
  ContextPtr context = symbols.any.Context();
  DeviceGuard guard(context);
  Array1<int32_t> aux_labels;
  Array1<float> score_offsets;
  FsaVec graph =
      k2::LevenshteinGraphs(symbols.any.Specialize<int32_t>(), ins_del_score,
                            &aux_labels, &score_offsets);
  torch::Tensor aux_labels_tensor = ToTorch(aux_labels);
  torch::Tensor score_offsets_tensor = ToTorch(score_offsets);
  FsaClass dest(graph, aux_labels_tensor);
  dest.SetAttr("__ins_del_score_offset_internal_attr_",
               torch::IValue(score_offsets_tensor));
  return dest;
}

FsaClass LevenshteinGraphs(const std::vector<std::vector<int32_t>> &symbols,
                           float ins_del_score /*= -0.501*/,
                           torch::optional<torch::Device> device /*= {}*/) {
  ContextPtr context = GetContext(device.value_or(torch::Device(torch::kCPU)));
  DeviceGuard guard(context);
  Ragged<int32_t> ragged = CreateRagged2<int32_t>(symbols).To(context);
  RaggedAny ragged_any = RaggedAny(ragged.Generic());
  return LevenshteinGraphs(ragged_any, ins_del_score);
}

}  // namespace k2
