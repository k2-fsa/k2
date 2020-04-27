// k2/csrc/fsa_algo_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace k2 {

TEST(FsaAlgo, Connect) {
  std::vector<Arc> arcs = {
      {0, 1, 1}, {0, 2, 2}, {1, 3, 3}, {1, 6, 6},
      {2, 4, 2}, {2, 6, 3}, {5, 0, 1},
  };

  std::vector<int32_t> arc_indexes = {0, 2, 4, 6, 6, 6, 7};

  {
    Fsa fsa;
    std::vector<int32_t> state_map(10);  // an arbitrary number
    ConnectCore(fsa, &state_map);
    EXPECT_TRUE(state_map.empty());
  }

  {
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);

    std::vector<int32_t> state_map(10);  // an arbitrary number
    ConnectCore(fsa, &state_map);

    EXPECT_EQ(state_map.size(), 4u);
    EXPECT_EQ(state_map[0], 0);
    EXPECT_EQ(state_map[1], 1);
    EXPECT_EQ(state_map[2], 2);
    EXPECT_EQ(state_map[3], 6);

    Fsa connected;
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    Connect(fsa, &connected, &arc_map);

    EXPECT_EQ(connected.NumStates(), 4u);  // state 3,4,5 from fsa are removed
    EXPECT_EQ(connected.arc_indexes[0], 0);
    EXPECT_EQ(connected.arc_indexes[1], 2);
    EXPECT_EQ(connected.arc_indexes[2], 3);
    EXPECT_EQ(connected.arc_indexes[3], 4);

    EXPECT_EQ(connected.arcs[0].src_state, 0);
    EXPECT_EQ(connected.arcs[0].dest_state, 1);
    EXPECT_EQ(connected.arcs[0].label, 1);

    EXPECT_EQ(connected.arcs[1].src_state, 0);
    EXPECT_EQ(connected.arcs[1].dest_state, 2);
    EXPECT_EQ(connected.arcs[1].label, 2);

    EXPECT_EQ(connected.arcs[2].src_state, 1);
    EXPECT_EQ(connected.arcs[2].dest_state, 3);
    EXPECT_EQ(connected.arcs[2].label, 6);

    EXPECT_EQ(connected.arcs[3].src_state, 2);
    EXPECT_EQ(connected.arcs[3].dest_state, 3);
    EXPECT_EQ(connected.arcs[3].label, 3);

    EXPECT_EQ(arc_map.size(), 4u);
    EXPECT_EQ(arc_map[0], 0);  // arc index 0 of original state 0 -> 1
    EXPECT_EQ(arc_map[1], 1);  // arc index 1 of original state 0 -> 2
    EXPECT_EQ(arc_map[2], 3);  // arc index 3 of original state 1 -> 6
    EXPECT_EQ(arc_map[3], 5);  // arc index 5 of original state 2 -> 6
  }
}

}  // namespace k2
