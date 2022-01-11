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

#include <gtest/gtest.h>

#include <vector>

#include "k2/csrc/rnnt_decode.h"
#include "k2/csrc/test_utils.h"

namespace k2 {
TEST(RnntDecode, PruneStreamsBasic) {
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    RaggedShape shape(c, "[ [ x x x x x ] [ x x x x x x x ] ]");
    Array1<double> scores(
        c, std::vector<double>({2, 1, 5, 3, 2.5, 6, 7, 8, 5, 4, 3, 9}));
    Array1<int32_t> categories(
        c, std::vector<int32_t>({1, 0, 1, 2, 1, 2, 0, 1, 2, 2, 0, 3}));
    Renumbering renumbering;
    Ragged<int32_t> kept_states;
    Array1<int32_t> kept_categories;
    float beam = 2.9;
    int32_t max_per_stream = 4, max_per_category = 2;
    PruneStreams(shape, scores, categories, beam, max_per_stream,
                 max_per_category, &renumbering, &kept_states,
                 &kept_categories);
    Ragged<int32_t> expected_kept_states(
        c, "[ [ [ 2 4 ] [ 3 ] ] [ [ 6 ] [ 7 ] [ 11 ] ] ]");
    Array1<int32_t> expected_kept_categories(
        c, std::vector<int32_t>({1, 2, 0, 1, 3}));
    Array1<int32_t> expected_new2old(c,
                                     std::vector<int32_t>({2, 3, 4, 6, 7, 11}));
    K2_CHECK(Equal(expected_kept_states, kept_states));
    K2_CHECK(Equal(expected_kept_categories, kept_categories));
    K2_CHECK(Equal(expected_new2old, renumbering.New2Old()));
  }
}
}  // namespace k2
