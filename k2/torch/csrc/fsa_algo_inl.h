/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
 *
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

#ifndef K2_TORCH_CSRC_FSA_ALGO_INL_H_
#define K2_TORCH_CSRC_FSA_ALGO_INL_H_

#ifndef IS_IN_K2_TORCH_CSRC_FSA_ALGO_H_
#error "this file is supposed to be included only by fsa_algo.h"
#endif

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/torch/csrc/fsa_class.h"

namespace k2 {

template <typename FloatType>
Array1<FloatType> GetTotScores(FsaClass &fsa, bool log_semiring /* = true*/) {
  Ragged<int32_t> state_batches = GetStateBatches(fsa.fsa, true);
  Array1<int32_t> dest_states = GetDestStates(fsa.fsa, true);
  Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa.fsa, dest_states);
  Ragged<int32_t> entering_arc_batches =
      GetEnteringArcIndexBatches(fsa.fsa, incoming_arcs, state_batches);

  auto forward_scores = GetForwardScores<FloatType>(
      fsa.fsa, state_batches, entering_arc_batches, log_semiring, nullptr);
  return GetTotScores<FloatType>(fsa.fsa, forward_scores);
}

}  // namespace k2
#endif  // K2_TORCH_CSRC_FSA_ALGO_INL_H_
