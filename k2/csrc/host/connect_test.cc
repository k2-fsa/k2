// k2/csrc/connect_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/connect.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {
TEST(ConnectTest, ConnectCore) {
  {
    // case 1: an empty input fsa
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
    std::vector<int32_t> state_b_to_a(10);
    bool status = ConnectCore(fsa, &state_b_to_a);
    EXPECT_TRUE(state_b_to_a.empty());
    EXPECT_TRUE(status);
  }
  {
    // case 2: a connected, acyclic FSA
    std::vector<Arc> arcs = {
        {0, 1, 1}, {1, 2, 2}, {1, 3, 3}, {2, 4, -1}, {3, 4, -1},
    };
    FsaCreator fsa_creator(arcs, 4);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(fsa, &state_b_to_a);
    ASSERT_EQ(state_b_to_a.size(), 5u);
    // notice that state_b_to_a maps:
    //   2 -> 3
    //   3 -> 2
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 1, 3, 2, 4));
    EXPECT_TRUE(status);
  }
  {
    // case 3: a connected, cyclic FSA
    // the cycle is a self-loop, the output is still topsorted.
    std::vector<Arc> arcs = {
        {0, 1, 1}, {1, 2, 2}, {1, 3, 3}, {2, 2, 2}, {2, 4, -1}, {3, 4, -1},
    };
    FsaCreator fsa_creator(arcs, 4);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(fsa, &state_b_to_a);
    ASSERT_EQ(state_b_to_a.size(), 5u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 1, 3, 2, 4));
    EXPECT_TRUE(status);
  }
  {
    // case 4: a non-connected, acyclic, non-topsorted FSA
    std::vector<Arc> arcs = {
        {0, 4, 4}, {0, 3, 3}, {1, 0, 1}, {3, 5, -1}, {4, 2, 2}, {4, 3, 3},
    };
    FsaCreator fsa_creator(arcs, 5);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(fsa, &state_b_to_a);
    EXPECT_TRUE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    /*                                               0  1  2  3 */
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 4, 3, 5));  // topsorted
  }

  {
    // case 5: a non-connected, cyclic, non-topsorted FSA
    // the output fsa will contain a cycle
    std::vector<Arc> arcs = {
        {0, 4, 4},  {0, 3, 3}, {1, 0, 1}, {3, 0, 3},
        {3, 5, -1}, {4, 2, 2}, {4, 3, 3},
    };
    FsaCreator fsa_creator(arcs, 5);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(fsa, &state_b_to_a);
    EXPECT_FALSE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 3, 4, 5));
  }
  {
    // case 6 (another one): a non-connected, cyclic, non-topsorted FSA;
    // the cycle is removed since state 2 is not co-accessible
    std::vector<Arc> arcs = {
        {0, 4, 4},  {0, 3, 3}, {1, 0, 1}, {2, 2, 2},
        {3, 5, -1}, {4, 2, 2}, {4, 3, 3},
    };
    FsaCreator fsa_creator(arcs, 5);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(fsa, &state_b_to_a);
    EXPECT_TRUE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 4, 3, 5));
  }
}

TEST(ConnectTest, Connect) {
  {
    // case 1: a non-connected, non-topsorted, acyclic input fsa;
    // the output fsa is topsorted.
    std::vector<Arc> src_arcs = {
        {0, 1, 1}, {0, 2, 2},  {1, 3, 3}, {1, 6, -1},
        {2, 4, 2}, {2, 6, -1}, {2, 1, 1}, {5, 0, 1},
    };
    FsaCreator fsa_creator(src_arcs, 6);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> state_b_to_a(10);  // an arbitrary number
    bool status = ConnectCore(fsa, &state_b_to_a);
    EXPECT_TRUE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 2, 1, 6));

    Array2Size<int32_t> fsa_size;
    Connection connection(fsa);
    connection.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &connected_fsa = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    connection.GetOutput(&connected_fsa, arc_map.data());

    std::vector<int32_t> arc_indexes(
        connected_fsa.indexes, connected_fsa.indexes + connected_fsa.size1 + 1);
    std::vector<Arc> arcs(connected_fsa.data,
                          connected_fsa.data + connected_fsa.size2);
    ASSERT_EQ(connected_fsa.size1, 4);  // state 3,4,5 from `fsa` are removed
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 4, 5, 5));

    std::vector<Arc> target_arcs = {
        {0, 2, 1}, {0, 1, 2}, {1, 3, -1}, {1, 2, 1}, {2, 3, -1},
    };
    for (auto i = 0; i != target_arcs.size(); ++i)
      EXPECT_EQ(arcs[i], target_arcs[i]);

    ASSERT_EQ(arc_map.size(), 5u);
    EXPECT_THAT(arc_map, ::testing::ElementsAre(0, 1, 5, 6, 3));
  }

  {
    // A non-empty fsa that after trimming, it returns an empty fsa.
    std::vector<Arc> arcs = {
        {0, 1, 1}, {0, 2, 2},  {1, 3, 3}, {1, 6, 6},  {2, 4, 2},
        {2, 6, 3}, {2, 6, -1}, {5, 0, 1}, {5, 7, -1},
    };
    FsaCreator fsa_creator(arcs, 7);
    const auto &fsa = fsa_creator.GetFsa();

    Array2Size<int32_t> fsa_size;
    Connection connection(fsa);
    connection.GetSizes(&fsa_size);
    FsaCreator fsa_creator_out(fsa_size);
    auto &connected_fsa = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    bool status = connection.GetOutput(&connected_fsa, arc_map.data());

    EXPECT_TRUE(IsEmpty(connected_fsa));
    EXPECT_TRUE(status);
    EXPECT_TRUE(arc_map.empty());
  }
  {
    // a cyclic input fsa
    // after trimming, the cycle is removed;
    // so the output fsa should be topsorted.
    std::vector<Arc> arcs = {
        {0, 3, 3}, {0, 5, 5},  {1, 2, 2}, {2, 1, 1},  {3, 5, 5},  {3, 2, 2},
        {3, 4, 4}, {3, 6, -1}, {4, 5, 5}, {4, 6, -1}, {5, 6, -1},
    };
    FsaCreator fsa_creator(arcs, 6);
    const auto &fsa = fsa_creator.GetFsa();
    Array2Size<int32_t> fsa_size;
    Connection connection(fsa);
    connection.GetSizes(&fsa_size);
    FsaCreator fsa_creator_out(fsa_size);
    auto &connected_fsa = fsa_creator_out.GetFsa();
    connection.GetOutput(&connected_fsa);

    EXPECT_TRUE(IsTopSorted(connected_fsa));
  }

  {
    // a cyclic input fsa
    // after trimming, the cycle remains (it is not a self-loop);
    // so the output fsa is NOT topsorted.
    std::vector<Arc> arcs = {
        {0, 3, 3}, {0, 2, 2}, {1, 0, 1}, {2, 6, -1}, {3, 5, 5},
        {3, 2, 2}, {3, 5, 5}, {4, 4, 4}, {5, 3, 3},  {5, 4, 4},
    };
    FsaCreator fsa_creator(arcs, 6);
    const auto &fsa = fsa_creator.GetFsa();
    Array2Size<int32_t> fsa_size;
    Connection connection(fsa);
    connection.GetSizes(&fsa_size);
    FsaCreator fsa_creator_out(fsa_size);
    auto &connected_fsa = fsa_creator_out.GetFsa();
    bool status = connection.GetOutput(&connected_fsa);

    EXPECT_FALSE(IsTopSorted(connected_fsa));
    EXPECT_FALSE(status);
  }

  {
    // a cyclic input fsa
    // after trimming, the cycle remains (it is not a self-loop);
    // so the output fsa is NOT topsorted.
    std::vector<Arc> arcs = {{0, 1, 1}, {0, 2, 2}, {1, 1, 1}, {1, 3, -1},
                             {2, 1, 1}, {2, 2, 2}, {2, 3, -1}};
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    Array2Size<int32_t> fsa_size;
    Connection connection(fsa);
    connection.GetSizes(&fsa_size);
    FsaCreator fsa_creator_out(fsa_size);
    auto &connected_fsa = fsa_creator_out.GetFsa();
    bool status = connection.GetOutput(&connected_fsa);

    EXPECT_TRUE(IsTopSorted(connected_fsa));
    EXPECT_TRUE(status);
  }
}

}  // namespace k2
