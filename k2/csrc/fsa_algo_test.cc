// k2/csrc/fsa_algo_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa_renderer.h"
#include "k2/csrc/fsa_util.h"

namespace k2 {

TEST(FsaAlgo, ConnectCore) {
  {
    // case 1: an empty input fsa
    Fsa a;
    std::vector<int32_t> state_b_to_a(10);
    ConnectCore(a, &state_b_to_a);
    EXPECT_TRUE(state_b_to_a.empty());
  }
  {
    // case 2: a connected, acyclic FSA
    std::string s = R"(
      0 1 1
      1 2 2
      1 3 3
      2 4 4
      3 4 4
      4
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);
    std::vector<int32_t> state_b_to_a;
    ConnectCore(*a, &state_b_to_a);
    ASSERT_EQ(state_b_to_a.size(), 5u);
    // notice that state_b_to_a maps:
    //   2 -> 3
    //   3 -> 2
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 1, 3, 2, 4));
  }
  {
    // case 3: a connected, cyclic FSA
    std::string s = R"(
      0 1 1
      1 2 2
      1 3 3
      2 2 2
      2 4 4
      3 4 4
      4
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);
    std::vector<int32_t> state_b_to_a;
    ConnectCore(*a, &state_b_to_a);
    ASSERT_EQ(state_b_to_a.size(), 5u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 1, 2, 3, 4));
  }
  {
    // case 4: a non-connected, acyclic, non-topsorted FSA
    std::string s = R"(
      1 0 1
      4 2 2
      3 5 5
      4 3 3
      0 4 4
      0 3 3
      5
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a;
    ConnectCore(*a, &state_b_to_a);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    /*                                               0  1  2  3 */
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 4, 3, 5));  // topsorted
  }

  {
    // case 5: a non-connected, cyclic, non-topsorted FSA
    std::string s = R"(
      1 0 1
      4 2 2
      3 5 5
      4 3 3
      0 4 4
      0 3 3
      3 0 3
      5
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a;
    ConnectCore(*a, &state_b_to_a);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    /*                                               0  1  2  3 */
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 3, 4, 5));  // topsorted
  }

  {
    // case 6 (another one): a non-connected, cyclic, non-topsorted FSA
    std::string s = R"(
      1 0 1
      4 2 2
      3 5 5
      4 3 3
      0 4 4
      0 3 3
      2 4 4
      5
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a;
    ConnectCore(*a, &state_b_to_a);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    /*                                               0  1  2  3 */
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 3, 4, 5));  // topsorted
  }
}
TEST(FsaAlgo, Connect) {
  {
    // case 1: a non-connected, non-topsorted, acylic fsa
    std::string s = R"(
      0 1 1
      0 2 2
      1 3 3
      1 6 6
      2 4 2
      2 6 3
      5 0 1
      2 1 1
      6
    )";

    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a(10);  // an arbitrary number
    ConnectCore(*a, &state_b_to_a);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 2, 1, 6));

    Fsa b;
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    Connect(*a, &b, &arc_map);
    EXPECT_TRUE(IsTopSorted(b));

    ASSERT_EQ(b.NumStates(), 4u);  // state 3,4,5 from `a` are removed
    EXPECT_THAT(b.arc_indexes, ::testing::ElementsAre(0, 2, 4, 5));

    std::vector<Arc> target_arcs = {
        {0, 2, 1}, {0, 1, 2}, {1, 3, 3}, {1, 2, 1}, {2, 3, 6},
    };
    EXPECT_EQ(b.arcs[0], target_arcs[0]);
    EXPECT_EQ(b.arcs[1], target_arcs[1]);
    EXPECT_EQ(b.arcs[2], target_arcs[2]);
    EXPECT_EQ(b.arcs[3], target_arcs[3]);
    EXPECT_EQ(b.arcs[4], target_arcs[4]);

    ASSERT_EQ(arc_map.size(), 5u);
    EXPECT_THAT(arc_map, ::testing::ElementsAre(0, 1, 5, 6, 3));
  }

  {
    // A non-empty fsa that after trimming, it returns an empty fsa.
    std::string s = R"(
      0 1 1
      0 2 2
      1 3 3
      1 6 6
      2 4 2
      2 6 3
      5 0 1
      5 7 2
      7
    )";

    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    Fsa b;
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    Connect(*a, &b, &arc_map);
    EXPECT_TRUE(IsEmpty(b));
    EXPECT_TRUE(arc_map.empty());
  }
}

TEST(FsaAlgo, Intersect) {
  // empty fsa
  {
    Fsa a;
    Fsa b;
    Fsa c;
    // arbitrary states and arcs
    c.arcs = {{0, 1, 2}};
    c.arc_indexes = {1};
    std::vector<int32_t> arc_map_a(10);  // an arbitrary number
    std::vector<int32_t> arc_map_b(5);
    bool status = Intersect(a, b, &c, &arc_map_a, &arc_map_b);
    EXPECT_TRUE(status);
    EXPECT_TRUE(c.arc_indexes.empty());
    EXPECT_TRUE(c.arcs.empty());
    EXPECT_TRUE(arc_map_a.empty());
    EXPECT_TRUE(arc_map_b.empty());
  }

  {
    std::vector<Arc> arcs_a = {{0, 1, 1}, {1, 2, 0}, {1, 3, 1},
                               {1, 4, 2}, {2, 2, 1}, {2, 3, 1},
                               {2, 3, 2}, {3, 3, 0}, {3, 4, 1}};
    std::vector<int32_t> arc_indexes_a = {0, 1, 4, 7, 9};
    Fsa a;
    a.arc_indexes = std::move(arc_indexes_a);
    a.arcs = std::move(arcs_a);

    std::vector<Arc> arcs_b = {
        {0, 1, 1},
        {1, 3, 1},
        {1, 2, 2},
        {2, 3, 1},
    };
    std::vector<int32_t> arc_indexes_b = {
        0,
        1,
        3,
        4,
    };
    Fsa b;
    b.arc_indexes = std::move(arc_indexes_b);
    b.arcs = std::move(arcs_b);

    Fsa c;
    std::vector<int32_t> arc_map_a(10);  // an arbitrary number
    std::vector<int32_t> arc_map_b(5);
    bool status = Intersect(a, b, &c, &arc_map_a, &arc_map_b);
    EXPECT_TRUE(status);

    std::vector<Arc> arcs_c = {
        {0, 1, 1}, {1, 2, 0}, {1, 3, 1}, {1, 4, 2}, {2, 5, 1},
        {2, 6, 1}, {2, 6, 2}, {3, 3, 0}, {6, 6, 0}, {6, 7, 1},
    };
    std::vector<int32_t> arc_indexes_c = {0, 1, 4, 6, 8, 8, 8, 10};

    ASSERT_EQ(c.arc_indexes.size(), arc_indexes_c.size());
    EXPECT_THAT(c.arc_indexes, ::testing::ElementsAre(0, 1, 4, 7, 8, 8, 8, 10));
    ASSERT_EQ(c.arcs.size(), arcs_c.size());
    for (std::size_t i = 0; i != arcs_c.size(); ++i)
      EXPECT_EQ(c.arcs[i], arcs_c[i]);

    // arc index in `c` -> arc index in `a`
    // 0 -> 0
    // 1 -> 1
    // 2 -> 2
    // 3 -> 3
    // 4 -> 4
    // 5 -> 5
    // 6 -> 6
    // 7 -> 7
    // 8 -> 7
    // 9 -> 8
    EXPECT_THAT(arc_map_a,
                ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 7, 8));

    // arc index in `c` -> arc index in `b`
    // 0 -> 0
    // 1 -> -1
    // 2 -> 1
    // 3 -> 2
    // 4 -> 1
    // 5 -> 1
    // 6 -> 2
    // 7 -> -1
    // 8 -> -1
    // 9 -> 3
    EXPECT_THAT(arc_map_b,
                ::testing::ElementsAre(0, -1, 1, 2, 1, 1, 2, -1, -1, 3));
  }
}

TEST(FsaAlgo, ArcSort) {
  // empty fsa
  {
    Fsa fsa;
    Fsa arc_sorted;
    // arbitrary states and arcs
    arc_sorted.arcs = {{0, 1, 2}};
    arc_sorted.arc_indexes = {1};
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    ArcSort(fsa, &arc_sorted, &arc_map);
    EXPECT_TRUE(arc_sorted.arcs.empty());
    EXPECT_TRUE(arc_sorted.arc_indexes.empty());
    EXPECT_TRUE(arc_map.empty());
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 2}, {0, 4, 0}, {0, 2, 0}, {1, 2, 1}, {1, 3, 0}, {2, 1, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 3, 5, 6, 6};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    Fsa arc_sorted;
    std::vector<int32_t> arc_map;
    ArcSort(fsa, &arc_sorted, &arc_map);
    EXPECT_THAT(arc_sorted.arc_indexes, ::testing::ElementsAre(0, 3, 5, 6, 6));
    ASSERT_EQ(arc_sorted.arcs.size(), fsa.arcs.size());
    std::vector<Arc> target_arcs = {
        {0, 2, 0}, {0, 4, 0}, {0, 1, 2}, {1, 3, 0}, {1, 2, 1}, {2, 1, 0},
    };
    for (std::size_t i = 0; i != target_arcs.size(); ++i)
      EXPECT_EQ(arc_sorted.arcs[i], target_arcs[i]);

    // arc index in `arc_sortd` -> arc index in original `fsa`
    // 0 -> 2
    // 1 -> 1
    // 2 -> 0
    // 3 -> 4
    // 4 -> 3
    // 5 -> 5
    EXPECT_THAT(arc_map, ::testing::ElementsAre(2, 1, 0, 4, 3, 5));
  }
}

TEST(FsaAlgo, TopSort) {
  {
    // case 1: empty input fsa
    Fsa fsa;
    Fsa top_sorted;
    std::vector<int32_t> state_map(10);
    bool status = TopSort(fsa, &top_sorted, &state_map);

    EXPECT_TRUE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 2: non-connected fsa (not co-accessible)
    std::string s = R"(
0 2 3
1 2 1
2
)";
    auto fsa = StringToFsa(s);
    ASSERT_NE(fsa.get(), nullptr);

    Fsa top_sorted;
    std::vector<int32_t> state_map(10);
    bool status = TopSort(*fsa, &top_sorted, &state_map);

    EXPECT_FALSE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 3: non-connected fsa (not accessible)
    std::string s = R"(
0 2 3
1 0 1
2
)";
    auto fsa = StringToFsa(s);
    ASSERT_NE(fsa.get(), nullptr);

    Fsa top_sorted;
    std::vector<int32_t> state_map(10);
    bool status = TopSort(*fsa, &top_sorted, &state_map);

    EXPECT_FALSE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 4: connected fsa
    std::string s = R"(
0 4 40
0 2 20
1 6 2
2 3 30
3 6 60
3 1 10
4 5 50
5 2 8
6
)";
    auto fsa = StringToFsa(s);
    ASSERT_NE(fsa.get(), nullptr);

    Fsa top_sorted;
    std::vector<int32_t> state_map;

    TopSort(*fsa, &top_sorted, &state_map);

    ASSERT_EQ(top_sorted.NumStates(), fsa->NumStates());

    ASSERT_FALSE(state_map.empty());
    EXPECT_THAT(state_map, ::testing::ElementsAre(0, 4, 5, 2, 3, 1, 6));

    ASSERT_FALSE(IsEmpty(top_sorted));

    const auto &arc_indexes = top_sorted.arc_indexes;
    const auto &arcs = top_sorted.arcs;

    ASSERT_EQ(arc_indexes.size(), 7u);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 3, 4, 5, 7, 8));
    std::vector<Arc> expected_arcs = {
        {0, 1, 40}, {0, 3, 20}, {1, 2, 50}, {2, 3, 8},
        {3, 4, 30}, {4, 6, 60}, {4, 5, 10}, {5, 6, 2},
    };

    for (auto i = 0; i != 8; ++i) {
      EXPECT_EQ(arcs[i], expected_arcs[i]);
    }
  }
}

}  // namespace k2
