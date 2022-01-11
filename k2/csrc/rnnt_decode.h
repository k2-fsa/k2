/**
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey, Wei Kang)
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

#ifndef K2_CSRC_RNNT_DECODE_H_
#define K2_CSRC_RNNT_DECODE_H_

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/ragged.h"

namespace k2 {
/*
  This is a utility function to be used in RNN-T decoding.  It performs pruning
  of 3 different types simultaneously, so the tightest of the 3 constraints will
  apply.

    @param [in] shape  A ragged shape with 2 axes; in practice, shape.Dim0()
                  will be the number of decoding streams, and shape.TotSize(1)
                  will be the total number of states over all streams.
    @param [in] scores  The scores we want to prune on, such that within each
                  decoding stream, the highest scores are kept.
                  Must satisfy scores.Dim() == shape.TotSize(1).
    @param [in] categories  An array of with categories->Dim() ==
                  shape.TotSize(1), that streams states into categories so we
                  can impose a maximum number of categories per stream, and
                  also sort the states by category (The categories will actually
                  correspond to limited-symbol "contexts", e.g. 1 or 2 symbols
                  of left-context).
    @param [in] beam  The pruning beam, as a difference in scores.  Within
                  a stream, we will not keep states whose score is less
                  than best_score - beam.
    @param [in] max_per_stream  The maximum number of states to keep per
                  stream, if >0.  (If <=0, there is no such constraint).
    @param [in] max_per_category  The maximum number of states to keep
                  per category per stream, if >0. (If <=0, there is no such
                  constraint). In practice the categories correspond to
                  "contexts", corresponding to the limited symbol histories
                  (1 or 2 symbols) that we feed to the decoder network.
    @param [out] renumbering  A renumbering object that represents the states
                   to keep, with num_old_elems == shape.TotSize(1). Should be
                   constructed with default constructor, this function will
                   assign to it.
    @param [out] kept_states  At exit, will be a ragged array with 3 axes
                   with indexes corresponding to [stream][category][state],
                   containing the numbers of the states we kept after pruning.
                   The elements of this ragged array can be interpreted as
                   indexes into `scores`.  The states will not be in the
                   same order as the original states, because they will be
                   sorted by category.
    @param [out] kept_categories  At exit, will be an array with
                   kept_categories->Dim() == kept_states->TotSize(1),
                   whose elements are categories (elements of the
                   input arary `categories`) to which the kept states belong.

   Implementation note: if max_per_stream >= shape.TotSize(1) or
   max_per_category >= shape.TotSize(1), an optimization may be possible in
  which we can avoid sorting the scores.
 */
void PruneStreams(const RaggedShape &shape, const Array1<double> &scores,
                  const Array1<int32_t> &categories, float beam,
                  int32_t max_per_stream, int32_t max_per_category,
                  Renumbering *renumbering, Ragged<int32_t> *kept_states,
                  Array1<int32_t> *kept_categories);
}  // namespace k2

#endif  // K2_CSRC_RNNT_DECODE_H_
