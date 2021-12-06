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

#ifndef K2_CSRC_ONLINE_DENSE_INTERSECTER_H_
#define K2_CSRC_ONLINE_DENSE_INTERSECTER_H_

#include "k2/csrc/fsa.h"

namespace k2 {
class MultiGraphDenseIntersectPruned;
class OnlineDenseIntersecter {
  public:
    OnlineDenseIntersecter(FsaVec &a_fsas, int32_t num_seqs, float search_beam,
                      float output_beam, int32_t min_active_states,
                      int32_t max_active_states);
    void Intersect(DenseFsaVec &b_fsas, bool is_final);
    void FormatOutput(FsaVec *out, Array1<int32_t> *arc_map_a, bool is_final);
    ~OnlineDenseIntersecter();

  private:
    MultiGraphDenseIntersectPruned *impl_;
};
};  // namespace k2

#endif  // K2_CSRC_ONLINE_DENSE_INTERSECTER_H_

