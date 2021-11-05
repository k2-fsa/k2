/**
 * Copyright      2021  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef K2_TORCH_CSRC_FSA_ALGO_H_
#define K2_TORCH_CSRC_FSA_ALGO_H_

#include "k2/csrc/fsa.h"
#include "k2/torch/csrc/fsa_class.h"

namespace k2 {

FsaClass CtcTopo(int32_t max_token, bool modified = false,
                 torch::Device device = torch::kCPU);

FsaClass IntersectDensePruned(FsaClass &graph, DenseFsaVec &dense,
                              float search_beam, float output_beam,
                              int32_t min_activate_states,
                              int32_t max_activate_states);

FsaClass ShortestPath(FsaClass &lattice);
}  // namespace k2

#endif  // K2_TORCH_CSRC_FSA_ALGO_H_
