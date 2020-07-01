// k2/csrc/arcsort_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/arcsort.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {
TEST(ArcSortTest, ArcSorter) {
  // empty fsa
  {
    // case 1: empty input fsa
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
    ArcSorter sorter(fsa);
    Array2Size<int32_t> fsa_size;
    sorter.GetSizes(&fsa_size);
    EXPECT_EQ(fsa_size.size1, 0);
    EXPECT_EQ(fsa_size.size2, 0);

    FsaCreator fsa_creator_out(fsa_size);
    auto &arc_sorted = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    sorter.GetOutput(&arc_sorted, arc_map.data());

    EXPECT_TRUE(IsEmpty(arc_sorted));
    EXPECT_TRUE(arc_map.empty());
  }

  {
    std::vector<Arc> src_arcs = {
        {0, 1, 2}, {0, 4, 0}, {0, 2, 0}, {1, 2, 1}, {1, 3, 0}, {2, 1, 0},
    };
    FsaCreator fsa_creator(src_arcs, 4);
    const auto &fsa = fsa_creator.GetFsa();
    ArcSorter sorter(fsa);
    Array2Size<int32_t> fsa_size;
    sorter.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &arc_sorted = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    sorter.GetOutput(&arc_sorted, arc_map.data());

    EXPECT_FALSE(arc_map.empty());
    EXPECT_TRUE(IsArcSorted(arc_sorted));

    std::vector<int32_t> arc_indexes(arc_sorted.indexes,
                                     arc_sorted.indexes + arc_sorted.size1 + 1);
    std::vector<Arc> arcs(arc_sorted.data, arc_sorted.data + arc_sorted.size2);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 3, 5, 6, 6, 6));
    ASSERT_EQ(arcs.size(), fsa.size2);
    std::vector<Arc> target_arcs = {
        {0, 2, 0}, {0, 4, 0}, {0, 1, 2}, {1, 3, 0}, {1, 2, 1}, {2, 1, 0},
    };
    for (std::size_t i = 0; i != target_arcs.size(); ++i)
      EXPECT_EQ(arcs[i], target_arcs[i]);

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

TEST(ArcSortTest, ArcSort) {
  // empty fsa
  {
    // case 1: empty input fsa
    FsaCreator fsa_creator;
    auto &fsa = fsa_creator.GetFsa();
    std::vector<int32_t> arc_map(fsa.size2);
    ArcSort(&fsa, arc_map.data());

    EXPECT_TRUE(IsEmpty(fsa));
    EXPECT_TRUE(arc_map.empty());
  }

  {
    std::vector<Arc> src_arcs = {
        {0, 1, 2}, {0, 4, 0}, {0, 2, 0}, {1, 2, 1}, {1, 3, 0}, {2, 1, 0},
    };
    FsaCreator fsa_creator(src_arcs, 4);
    auto &fsa = fsa_creator.GetFsa();
    std::vector<int32_t> arc_map(fsa.size2);
    ArcSort(&fsa, arc_map.data());

    EXPECT_TRUE(IsArcSorted(fsa));

    std::vector<int32_t> arc_indexes(fsa.indexes, fsa.indexes + fsa.size1 + 1);
    std::vector<Arc> arcs(fsa.data, fsa.data + fsa.size2);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 3, 5, 6, 6, 6));
    std::vector<Arc> target_arcs = {
        {0, 2, 0}, {0, 4, 0}, {0, 1, 2}, {1, 3, 0}, {1, 2, 1}, {2, 1, 0},
    };
    for (std::size_t i = 0; i != target_arcs.size(); ++i)
      EXPECT_EQ(arcs[i], target_arcs[i]);

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
}  // namespace k2
