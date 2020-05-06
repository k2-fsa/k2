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

TEST(FsaEquivalent, IsNotRandEquivalent) {
  {
    // one fsa will be empty after connecting
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5}, };
    Fsa a(std::move(arcs_a), 3);

    std::vector<Arc> arcs_b = {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}};
    Fsa b(std::move(arcs_b), 3);

    bool status = IsRandEquivalent(a, b);
    EXPECT_FALSE(status);
  }

  {
    // two fsas hold different set of arc labels
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5}, };
    Fsa a(std::move(arcs_a), 3);

    std::vector<Arc> arcs_b = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 6}, };
    Fsa b(std::move(arcs_b), 3);

    bool status = IsRandEquivalent(a, b);
    EXPECT_FALSE(status);
  }

  {
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5}, };
    Fsa a(std::move(arcs_a), 3);

    std::vector<Arc> arcs_b = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5}, {3, 4, 5}, };
    Fsa b(std::move(arcs_b), 4);

    bool status = IsRandEquivalent(a, b, 100);
    // Caution: this test may fail with a very low probability if the program
    // generates path from `a` every time
    EXPECT_FALSE(status);
  }
}

TEST(FsaEquivalent, IsRandEquivalent) {
  {
    // both fsas will be empty after connecting
    std::vector<Arc> arcs_a = {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}};
    Fsa a(std::move(arcs_a), 3);

    std::vector<Arc> arcs_b = {{0, 1, 1}, {0, 2, 2}};
    Fsa b(std::move(arcs_b), 3);

    bool status = IsRandEquivalent(a, b);
    EXPECT_TRUE(status);
  }

  {
    // same fsas
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5}, };
    Fsa a(std::move(arcs_a), 3);
    Fsa b = a;
    bool status = IsRandEquivalent(a, b);
    EXPECT_TRUE(status);
  }

  {
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {0, 3, 8}, {1, 4, 4}, {2, 4, 5}, };
    Fsa a(std::move(arcs_a), 4);

    std::vector<Arc> arcs_b = {
        {0, 2, 1}, {0, 1, 2}, {0, 3, 9}, {1, 4, 5}, {2, 4, 4}, };
    Fsa b(std::move(arcs_b), 4);

    bool status = IsRandEquivalent(a, b);
    EXPECT_TRUE(status);
  }
}

TEST(FsaEquivalent, RandomPathFail) {
  {
    Fsa fsa;
    Fsa path;
    bool status = RandomPath(fsa, &path);
    EXPECT_FALSE(status);
  }

  {
    // non-connected fsa
    std::vector<Arc> arcs = {{0, 1, 1}, {0, 2, 2}, {1, 3, 4}, };
    Fsa fsa(std::move(arcs), 3);
    Fsa path;
    bool status = RandomPath(fsa, &path);
    EXPECT_FALSE(status);
  }
}

TEST(FsaEquivalent, RandomPathSuccess) {
  {
    std::vector<Arc> arcs = {{0, 1, 1},
                             {0, 2, 2},
                             {1, 2, 3},
                             {2, 3, 4},
                             {2, 4, 5},
                             {3, 4, 7},
                             {4, 5, 9}, };
    Fsa fsa(std::move(arcs), 5);
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
    std::vector<Arc> arcs = {{0, 1, 1}, {1, 2, 3}, {2, 3, 4}, };
    Fsa fsa(std::move(arcs), 3);
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
}
}  // namespace k2
