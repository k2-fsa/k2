// k2/csrc/fsa_util_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace k2 {

TEST(FsaUtil, GetEnteringArcs) {
  std::vector<Arc> arcs = {
      {0, 1, 2}, {0, 2, 1}, {1, 2, 0}, {1, 3, 5}, {2, 3, 6},
  };
  std::vector<Range> leaving_arcs = {
      {0, 2}, {2, 4}, {4, 5}, {0, 0},  // the last state has no entering arcs
  };

  Fsa fsa;
  fsa.leaving_arcs = std::move(leaving_arcs);
  fsa.arcs = std::move(arcs);

  VecOfVec entering_arcs;
  GetEnteringArcs(fsa, &entering_arcs);

  const auto &ranges = entering_arcs.ranges;
  const auto &values = entering_arcs.values;
  EXPECT_EQ(ranges.size(), 4u);  // there are 4 states
  EXPECT_EQ(values.size(), 5u);  // there are 5 arcs

  // state 0, no entering arcs
  EXPECT_EQ(ranges[0].begin, ranges[0].end);

  // state 1 has one entering arc from state 0 with label 2
  EXPECT_EQ(ranges[1].begin, 0);
  EXPECT_EQ(ranges[1].end, 1);
  EXPECT_EQ(values[0].first, 2);   // label is 2
  EXPECT_EQ(values[0].second, 0);  // state is 0

  // state 2 has two entering arcs
  //  the first one: from state 0 with label 1
  //  the second one: from state 1 with label 0
  EXPECT_EQ(ranges[2].begin, 1);
  EXPECT_EQ(ranges[2].end, 3);
  EXPECT_EQ(values[1].first, 1);   // label is 1
  EXPECT_EQ(values[1].second, 0);  // state is 0

  EXPECT_EQ(values[2].first, 0);   // label is 0
  EXPECT_EQ(values[2].second, 1);  // state is 1

  // state 3 has two entering arcs
  //  the first one: from state 1 with label 5
  //  the second one: from state 2 with label 6
  EXPECT_EQ(ranges[3].begin, 3);
  EXPECT_EQ(ranges[3].end, 5);
  EXPECT_EQ(values[3].first, 5);   // label is 5
  EXPECT_EQ(values[3].second, 1);  // state is 1

  EXPECT_EQ(values[4].first, 6);   // label is 6
  EXPECT_EQ(values[4].second, 2);  // state is 2
}

}  // namespace k2
