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

#ifndef K2_PYTHON_CSRC_TORCH_PRUNED_RANGES_TO_LATTICE_H_
#define K2_PYTHON_CSRC_TORCH_PRUNED_RANGES_TO_LATTICE_H_

#include "k2/python/csrc/torch.h"

namespace k2 {

/*
 Convert pruned ranges to lattice while also supporting autograd.

 The input pruned ranges is normally generated by `get_rnnt_prune_ranges`.
 See k2/python/k2/rnnt_loss.py for the process of generating ranges and
 the information it represents.

 When this is implemented, the lattice is used to generate force-alignment.
 Perhaps you could find other uses for this function.

   @param ranges  A tensor containing the symbol indexes for each frame.
       Its shape is (B, T, s_range). See the docs in `get_rnnt_prune_ranges`
       in k2/python/k2/rnnt_loss.py for more details of this tensor.
       Its type is int32, consistent with that in rnnt_loss.py.
   @param frames  The number of frames per sample with shape (B).
       Its type is int32.
   @param symbols  The symbol sequence, a LongTensor of shape (B, S),
       and elements in {0..C-1}.
       Its type is int64(Long), consistent with that in rnnt_loss.py.
   @param logits  The pruned joiner network (or am/lm)
       of shape (B, T, s_range, C).
       Its type can be float32, float64, float16. Though float32 is mainly
       used. float64 and float16 are also supported for future use.
   @param [out] arc_map  A map from arcs in generated lattice to global index
       of logits, or -1 if the arc had no corresponding score in logits,
       e.g. arc pointing to super final state.
   @return  Return an FsaVec, which contains the generated lattice.
*/
FsaVec PrunedRangesToLattice(
    torch::Tensor ranges,   // [B][T][s_range]
    torch::Tensor frames,   // [B]
    torch::Tensor symbols,  // [B][S]
    torch::Tensor logits,   // [B][T][s_range][C]
    Array1<int32_t> *arc_map);

}  // namespace k2

void PybindPrunedRangesToLattice(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_PRUNED_RANGES_TO_LATTICE_H_
