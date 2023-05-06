/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Liyong Guo)
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

#ifndef K2_CSRC_PRUNED_RANGES_TO_LATTICE_H_
#define K2_CSRC_PRUNED_RANGES_TO_LATTICE_H_

#include <torch/extension.h>

#include <vector>

#include "k2/python/csrc/torch.h"

namespace k2 {

FsaVec PrunedRangesToLattice(
    torch::Tensor ranges,   // [B][T][s_range]
    torch::Tensor frames,   // [B][T]
    torch::Tensor symbols,  // [B][S]
    torch::Tensor logits,   // [B][T][s_range][C]
    Array1<int32_t> *arc_map);

}  // namespace k2

#endif  // K2_CSRC_PRUNED_RANGES_TO_LATTICE_H_
