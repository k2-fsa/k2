/**
 * Copyright (c)  2021  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef K2_CSRC_ONLINE_DENSE_INTERSECTOR_H_
#define K2_CSRC_ONLINE_DENSE_INTERSECTOR_H_

#include "k2/csrc/fsa.h"

namespace k2 {
class MultiGraphDenseIntersectPruned;
/**
     Pruned intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks for online fashion.

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might
                           just be a linear sequence of phones, or might be
                           something more complicated.  Must have either the
                           same Dim0() as b_fsas, or Dim0()==1 in which
                           case the graph is shared.
       @param [in] num_seqs  The number of sequences to do intersection at a
                             time, i.e. batch size. The input DenseFsaVec in
                             `Intersect` function MUST have `Dim0()` equals to
                             this.
       @param [in] search_beam    "Default" search/decoding beam.  The actual
                           beam is dynamic and also depends on max_active and
                           min_active.
       @param [in] output_beam    Beam for pruning the output FSA, will
                                  typically be smaller than search_beam.
       @param [in] min_active  Minimum number of FSA states that are allowed to
                           be active on any given frame for any given
                           intersection/composition task. This is advisory,
                           in that it will try not to have fewer than this
                           number active.
       @param [in] max_active  Maximum number of FSA states that are allowed to
                           be active on any given frame for any given
                           intersection/composition task. This is advisory,
                           in that it will try not to exceed that but may not
                           always succeed.  This determines the hash size.
*/
class OnlineDenseIntersecter {
 public:
  OnlineDenseIntersecter(FsaVec &a_fsas, int32_t num_seqs, float search_beam,
                         float output_beam, int32_t min_states,
                         int32_t max_states);

  /* Does intersection/composition for current chunk of nnet_output(given
     by a DenseFsaVec), but doesn't produce any output; the output is
     provided when you call FormatOutput().

       @param [in] b_fsas  The neural-net output, with each frame containing
                           the log-likes of each phone.
       @param [in] is_final Whether this is the final chunk of the nnet_output,
                            After calling this function with is_final is true,
                            means decoding finished.
   */
  void Intersect(DenseFsaVec &b_fsas, bool is_final);

  /* Format partial/final result of the intersection.

     @param [out] out  The FsaVec to contain the output lattice of the
                       intersection result.
     @param[out] arc_map_a  Will be set to a vector with Dim() equal to
                            the number of arcs in `out`, whose elements
                            contain the corresponding arc_idx012 in decoding
                            graph (i.e. a_fsas).
     @param [in] is_final True for final result, false for partial resutl.
   */
  void FormatOutput(FsaVec *out, Array1<int32_t> *arc_map_a, bool is_final);
  ~OnlineDenseIntersecter();

 private:
  MultiGraphDenseIntersectPruned *impl_;
};
};  // namespace k2

#endif  // K2_CSRC_ONLINE_DENSE_INTERSECTOR_H_
