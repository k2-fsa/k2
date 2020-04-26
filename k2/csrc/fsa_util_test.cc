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
      {0, 1, 2},  // 0
      {0, 2, 1},  // 1
      {1, 2, 0},  // 2
      {1, 3, 5},  // 3
      {2, 3, 6},  // 4
  };
  std::vector<int32_t> arc_indexes = {0, 2, 4, 5};

  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);

  std::vector<int32_t> arc_index(10);  // an arbitray number
  std::vector<int32_t> end_index(20);

  GetEnteringArcs(fsa, &arc_index, &end_index);

  EXPECT_EQ(end_index.size(), 4u);  // there are 4 states
  EXPECT_EQ(arc_index.size(), 5u);  // there are 5 arcs

  EXPECT_EQ(end_index[0], 0);  // state 0 has no entering arcs

  EXPECT_EQ(end_index[1], 1);  // state 1 has one entering arc
  EXPECT_EQ(arc_index[0], 0);  // arc index 0 from state 0

  EXPECT_EQ(end_index[2], 3);  // state 2 has two entering arcs
  EXPECT_EQ(arc_index[1], 1);  // arc index 1 from state 0
  EXPECT_EQ(arc_index[2], 2);  // arc index 2 from state 1

  EXPECT_EQ(end_index[3], 5);  // state 3 has two entering arcs
  EXPECT_EQ(arc_index[3], 3);  // arc index 3 from state 1
  EXPECT_EQ(arc_index[4], 4);  // arc index 4 from state 2
}

}  // namespace k2
