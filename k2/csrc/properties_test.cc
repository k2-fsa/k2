// k2/csrc/properties_test.cc

// Copyright (c)  2020  Haowen Qiue
//                      Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

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
