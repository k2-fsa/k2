// k2/csrc/topsort_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/topsort.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {
TEST(TopSortTest, TopSort) {
  {
    // case 1: empty input fsa
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
    TopSorter sorter(fsa);
    Array2Size<int32_t> fsa_size;
    sorter.GetSizes(&fsa_size);
    EXPECT_EQ(fsa_size.size1, 0);
    EXPECT_EQ(fsa_size.size2, 0);

    FsaCreator fsa_creator_out(fsa_size);
    auto &top_sorted = fsa_creator_out.GetFsa();
    std::vector<int32_t> state_map(fsa_size.size1);
    bool status = sorter.GetOutput(&top_sorted, state_map.data());

    EXPECT_TRUE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 2: non-connected fsa (not co-accessible)
    std::vector<Arc> arcs = {
        {0, 2, -1},
        {1, 2, -1},
        {1, 2, 0},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    TopSorter sorter(fsa);
    Array2Size<int32_t> fsa_size;
    sorter.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &top_sorted = fsa_creator_out.GetFsa();
    std::vector<int32_t> state_map(fsa_size.size1);
    bool status = sorter.GetOutput(&top_sorted, state_map.data());

    EXPECT_FALSE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 3: non-connected fsa (not accessible)
    std::vector<Arc> arcs = {
        {0, 2, -1},
        {1, 0, 1},
        {1, 2, 0},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    TopSorter sorter(fsa);
    Array2Size<int32_t> fsa_size;
    sorter.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &top_sorted = fsa_creator_out.GetFsa();
    std::vector<int32_t> state_map(fsa_size.size1);
    bool status = sorter.GetOutput(&top_sorted, state_map.data());

    EXPECT_FALSE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 4: connected fsa
    std::vector<Arc> src_arcs = {
        {0, 4, 40}, {0, 2, 20}, {1, 6, -1}, {2, 3, 30},
        {3, 6, -1}, {3, 1, 10}, {4, 5, 50}, {5, 2, 8},
    };
    FsaCreator fsa_creator(src_arcs, 6);
    const auto &fsa = fsa_creator.GetFsa();
    TopSorter sorter(fsa);
    Array2Size<int32_t> fsa_size;
    sorter.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &top_sorted = fsa_creator_out.GetFsa();
    std::vector<int32_t> state_map(fsa_size.size1);
    bool status = sorter.GetOutput(&top_sorted, state_map.data());

    ASSERT_EQ(top_sorted.NumStates(), fsa.NumStates());

    ASSERT_FALSE(state_map.empty());
    EXPECT_THAT(state_map, ::testing::ElementsAre(0, 4, 5, 2, 3, 1, 6));

    ASSERT_FALSE(IsEmpty(top_sorted));
    EXPECT_TRUE(IsTopSorted(top_sorted));

    std::vector<int32_t> arc_indexes(top_sorted.indexes,
                                     top_sorted.indexes + top_sorted.size1 + 1);
    std::vector<Arc> arcs(top_sorted.data, top_sorted.data + top_sorted.size2);
    ASSERT_EQ(arc_indexes.size(), 8u);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 3, 4, 5, 7, 8, 8));
    std::vector<Arc> expected_arcs = {
        {0, 1, 40}, {0, 3, 20}, {1, 2, 50}, {2, 3, 8},
        {3, 4, 30}, {4, 6, -1}, {4, 5, 10}, {5, 6, -1},
    };

    for (auto i = 0; i != 8; ++i) {
      EXPECT_EQ(arcs[i], expected_arcs[i]);
    }
  }
}

}  // namespace k2
