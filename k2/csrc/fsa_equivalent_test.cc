// k2/csrc/fsa_equivalent_test.cc

// Copyright (c)  2020  Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_equivalent.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"

namespace k2 {

TEST(Properties, RandomPathFail) {
  {
    Fsa fsa;
    Fsa path;
    bool status = RandomPath(fsa, &path);
    EXPECT_FALSE(status);
  }
  // TODO(haowen): add tests for non-connected fsa
}

TEST(Properties, RandomPathSuccess) {
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {2, 3, 4},
        {2, 4, 5}, {3, 4, 7}, {4, 5, 9},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 3, 5, 6, 7};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    Fsa path;

    {
      bool status = RandomPath(fsa, &path);
      EXPECT_TRUE(status);
    }

    {
      std::vector<int32_t> state_map;
      for (auto i = 0; i != 20; ++i) {
        bool status = RandomPath(fsa, &path, &state_map);
        EXPECT_TRUE(status);
        EXPECT_GT(state_map.size(), 0);
      }
    }
  }

  // test with linear structure fsa to check the resulted path
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 1},
        {1, 2, 3},
        {2, 3, 4},
    };
    std::vector<int32_t> arc_indexes = {0, 1, 2, 3};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    Fsa path;

    std::vector<int32_t> state_map;
    bool status = RandomPath(fsa, &path, &state_map);
    EXPECT_TRUE(status);
    ASSERT_EQ(fsa.arcs.size(), path.arcs.size());
    EXPECT_TRUE(fsa.arcs == path.arcs);
    ASSERT_EQ(fsa.arc_indexes.size(), path.arc_indexes.size());
    EXPECT_TRUE(fsa.arc_indexes == path.arc_indexes);
    EXPECT_THAT(state_map, ::testing::ElementsAre(0, 1, 2, 3));
  }

  // TODO(haowen): add tests for non-connected fsa
  std::vector<Arc> arcs = {
      {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {2, 3, 4}, {2, 4, 5},
      {3, 1, 6}, {3, 4, 7}, {4, 3, 8}, {4, 5, 9},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 3, 5, 7, 9};
}
}  // namespace k2
