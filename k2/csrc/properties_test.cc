// k2/csrc/properties_test.cc

// Copyright (c)  2020  Haowen Qiu
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
// addState, addArc, etc.) and use Test Fixtures by constructing
// reusable FSA examples.
TEST(Properties, IsNotTopSorted) {
  std::vector<Arc> arcs = {{0, 1, 0}, {0, 2, 0}, {2, 1, 0}, };
  std::vector<Range> leaving_arcs = {{0, 2}, {2, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool sorted = IsTopSorted(fsa);
  EXPECT_FALSE(sorted);
}

TEST(Properties, IsTopSorted) {
  std::vector<Arc> arcs = {{0, 1, 0}, {0, 2, 0}, {1, 2, 0}, };
  std::vector<Range> leaving_arcs = {{0, 2}, {2, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool sorted = IsTopSorted(fsa);
  EXPECT_TRUE(sorted);
}

TEST(Properties, HasNotSelfLoops) {
  std::vector<Arc> arcs = {{0, 1, 0}, {0, 2, 0}, {1, 2, 0}, };
  std::vector<Range> leaving_arcs = {{0, 2}, {2, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool hasSelfLoops = HasSelfLoops(fsa);
  EXPECT_FALSE(hasSelfLoops);
}

TEST(Properties, HasSelfLoops) {
  std::vector<Arc> arcs = {{0, 1, 0}, {1, 2, 0}, {1, 1, 0}, };
  std::vector<Range> leaving_arcs = {{0, 1}, {1, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool hasSelfLoops = HasSelfLoops(fsa);
  EXPECT_TRUE(hasSelfLoops);
}

TEST(Properties, IsNotDeterministic) {
  std::vector<Arc> arcs = {{0, 1, 2}, {1, 2, 0}, {1, 3, 0}, };
  std::vector<Range> leaving_arcs = {{0, 1}, {1, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool isDeterministic = IsDeterministic(fsa);
  EXPECT_FALSE(isDeterministic);
}

TEST(Properties, IsDeterministic) {
  std::vector<Arc> arcs = {{0, 1, 2}, {1, 2, 0}, {1, 3, 2}, };
  std::vector<Range> leaving_arcs = {{0, 1}, {1, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool isDeterministic = IsDeterministic(fsa);
  EXPECT_TRUE(isDeterministic);
}

TEST(Properties, IsNotEpsilonFree) {
  std::vector<Arc> arcs = {{0, 1, 2}, {0, 2, 0}, {1, 2, 1}, };
  std::vector<Range> leaving_arcs = {{0, 2}, {2, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool isEpsilonFree = IsEpsilonFree(fsa);
  EXPECT_FALSE(isEpsilonFree);
}

TEST(Properties, IsEpsilonFree) {
  std::vector<Arc> arcs = {{0, 1, 2}, {0, 2, 1}, {1, 2, 1}, };
  std::vector<Range> leaving_arcs = {{0, 2}, {2, 3}, };
  Fsa fsa;
  fsa.leaving_arcs = leaving_arcs;
  fsa.arcs = arcs;
  bool isEpsilonFree = IsEpsilonFree(fsa);
  EXPECT_TRUE(isEpsilonFree);
}
