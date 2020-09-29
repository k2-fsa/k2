/**
 * @brief
 * topsort_test
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/topsort.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"

namespace k2host {
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
    std::vector<Arc> arcs = {{0, 2, -1, 0}, {1, 2, -1, 0}, {1, 2, 0, 0}};
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
        {0, 2, -1, 0},
        {1, 0, 1, 0},
        {1, 2, 0, 0},
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
        {0, 4, 40, 0}, {0, 2, 20, 0}, {1, 6, -1, 0}, {2, 3, 30, 0},
        {3, 6, -1, 0}, {3, 1, 10, 0}, {4, 5, 50, 0}, {5, 2, 8, 0},
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
        {0, 1, 40, 0}, {0, 3, 20, 0}, {1, 2, 50, 0}, {2, 3, 8, 0},
        {3, 4, 30, 0}, {4, 6, -1, 0}, {4, 5, 10, 0}, {5, 6, -1, 0},
    };

    for (auto i = 0; i != 8; ++i) {
      EXPECT_EQ(arcs[i], expected_arcs[i]);
    }
  }
}

}  // namespace k2host
