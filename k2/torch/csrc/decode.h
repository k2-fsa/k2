/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef K2_TORCH_CSRC_DECODE_H_
#define K2_TORCH_CSRC_DECODE_H_

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/torch/csrc/fsa_class.h"
#include "torch/script.h"

namespace k2 {

/*
Note: Several functions in this file takes as inputs two kinds of aux_labels:
a linear array and a ragged array. Only one of them is used. We will refactor
the code once `FsaClass` is implemented, which wraps an FsaVec and its
attributes.
*/

/** Get decoding lattice from a neural network output and a decoding graph.

    @param nnet_output  A 3-D tensor with dtype torch.float32. It is usally
                        the last layer of the neural network model, e.g.,
                        the output of `log-softmax` layer. It has shape
                        `(N, T, C)`.
    @param decoding_graph  It is an FsaVec. It usually contains only only
                           on graph. For instance, when using CTC decoding,
                           it contains a single CTC topo graph; when using
                           HLG decoding, it contains a single HLG graph.

   @param supervision_segments  A 2-D tensor with dtype torch.int32.
                                Please refer to `k2::CreateDenseFsaVec()`
                                for its format.
   @param search_beam  See `k2::IntersectDensePruned()` for its meaning.
   @param output_beam  See `k2::IntersectDensePruned()` for its meaning.
   @param min_activate_states  See `k2::IntersectDensePruned()` for its meaning.
   @param max_activate_states  See `k2::IntersectDensePruned()` for its meaning.
   @param subsampling_factor  The subsampling factor of the model.
   @param in_aux_labels  If not empty, it associates an extra label with each
                         arc in decoding graph.
                         in_aux_labels.Dim() == decoding_graph.NumElements()
                         if in_aux_labels is not empty.
   @param in_ragged_aux_labels  If not empty, it must have 2 axes and it
               associates an extra label with each arc in decoding graph.
               in_ragged_aux_labels.tot_size(0) == decoding_graph.NumElements()
               if in_ragged_aux_labels is not empty.
   @param out_aux_labels If in_aux_labels is not empty, it associates an extra
                         label for each arc in the returned FSA
   @param out_ragged_aux_labels If in_aux_labels is not empty, it associates an
                         extra label for each arc in the returned FSA

   @return Return an FsaVec, which is the intersection of decoding graph and
           the FSA constructed from `nnet_output`.
 */
FsaClass GetLattice(torch::Tensor nnet_output, FsaClass &decoding_graph,
                    torch::Tensor supervision_segments, float search_beam,
                    float output_beam, int32_t min_activate_states,
                    int32_t max_activate_states, int32_t subsampling_factor);

/** Get aux labels of each FSA contained in the lattice.

    Note: The input aux labels are for each arc in the lattice, while
          the output aux_labels are for each FSA in the lattice.

    @param lattice An FsaVec containing linear FSAs. It can be the return
                   value of `OneBestDecoding()`.
    @param in_aux_labels  If not empty, it associates an extra label with each
                          arc in the `lattice.
    @param in_ragged_aux_labels  If not empty, it associates an extra label
                                 with each arc in the `lattice.
    @return Return a ragged array with two axes [utt][aux_label].
 */
Ragged<int32_t> GetTexts(FsaClass &lattice);

}  // namespace k2

#endif  // K2_TORCH_CSRC_DECODE_H_
