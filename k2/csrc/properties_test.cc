// k2/csrc/properties_test.cc

// Copyright     2020  Haowen Qiu
//                     Fangjun Kuang (csukuangfj@gmail.com)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "k2/csrc/properties.h"

#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"

using k2::Arc;
using k2::Fsa;
using k2::Range;

// TODO(haowen): create Fsa examples in a more elegant way (add methods
// addState, addArc, etc.)
static Fsa CreateNonTopSortedFsaExample() {
  std::vector<Arc> arcs = {
      {0, 2, 0},
      {2, 1, 0},
      {0, 1, 0},
  };
  std::vector<Range> leaving_arcs = {
      {0, 1},
      {1, 2},
      {2, 3},
  };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  return fsa;
}

TEST(Properties, IsTopSorted) {
  Fsa fsa = CreateNonTopSortedFsaExample();
  bool sorted = IsTopSorted(fsa);
  EXPECT_FALSE(sorted);

  std::vector<Arc> arcs = {
      {0, 1, 0},
      {1, 2, 0},
      {0, 2, 0},
  };
  fsa.arcs = arcs;
  sorted = IsTopSorted(fsa);
  EXPECT_TRUE(sorted);
}
