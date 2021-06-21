/**
 * Copyright      2021  Mobvoi Inc.        (authors: Wei Kang)
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

#include <algorithm>
#include <random>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/math.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(Connect, SingleFsa) {
  std::string s = R"(0 2 1 1
    0 3 3 3
    1 4 5 5
    1 6 -1 0
    2 1 2 2
    3 1 4 4
    5 3 6 6
    6
  )";

  auto fsa = FsaFromString(s);

 // int32_t gt = kFsaPropertiesTopSorted | kFsaPropertiesTopSortedAndAcyclic;
  int32_t p = GetFsaBasicProperties(fsa);
  K2_LOG(INFO) << FsaPropertiesAsString(p);
 // EXPECT_NE(p & gt, gt);

  Fsa sorted;
  Array1<int32_t> arc_map;
  Connect(fsa, &sorted, &arc_map);
  p = GetFsaBasicProperties(sorted);
  K2_LOG(INFO) << FsaPropertiesAsString(p);
  K2_LOG(INFO) << sorted;
  K2_LOG(INFO) << arc_map;
  // p = GetFsaBasicProperties(sorted);
  // EXPECT_EQ(p & gt, gt);
  /* top sorted fsa is
  0 2 1 1    // src arc 0
  0 1 2 2    // src arc 1
  1 2 1 21   // src arc 3
  1 3 -1 23  // src arc 4
  2 3 -1 13  // src arc 2
  3
  */

  //CheckArrayData(arc_map, {0, 1, 3, 4, 2});
}

TEST(Connect, CycleFsa) {
  std::string s = R"(0 1 1 1
    0 2 2 2
    1 2 3 3
    2 3 4 4
    2 4 5 5
    3 1 6 6
    3 6 -1 0
    5 2 7 7
    6
  )";

  auto fsa = FsaFromString(s);

  Fsa sorted;
  Array1<int32_t> arc_map;
  Connect(fsa, &sorted, &arc_map);
  K2_LOG(INFO) << sorted;
}

}  // namespace k2
