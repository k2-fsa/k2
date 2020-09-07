// k2/csrc/intersect_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/intersect.h"

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

TEST(IntersectTest, Intersect) {
  // empty fsa
  {
    FsaCreator fsa_creator_a;
    const auto &a = fsa_creator_a.GetFsa();
    FsaCreator fsa_creator_b;
    const auto &b = fsa_creator_b.GetFsa();
    Intersection intersection(a, b);
    Array2Size<int32_t> fsa_size;
    intersection.GetSizes(&fsa_size);
    EXPECT_EQ(fsa_size.size1, 0);
    EXPECT_EQ(fsa_size.size2, 0);

    FsaCreator fsa_creator_out(fsa_size);
    auto &c = fsa_creator_out.GetFsa();

    std::vector<int32_t> arc_map_a(fsa_size.size2);
    std::vector<int32_t> arc_map_b(fsa_size.size2);
    bool status =
        intersection.GetOutput(&c, arc_map_a.data(), arc_map_b.data());
    EXPECT_TRUE(status);
    EXPECT_TRUE(IsEmpty(c));
    EXPECT_TRUE(arc_map_a.empty());
    EXPECT_TRUE(arc_map_b.empty());
  }

  {
    std::vector<Arc> arcs_a = {{0, 1, 1}, {1, 2, 0}, {1, 3, 1},
                               {1, 4, 2}, {2, 2, 1}, {2, 3, 1},
                               {2, 3, 2}, {3, 3, 0}, {3, 4, 1}};
    FsaCreator fsa_creator_a(arcs_a, 4);
    const auto &a = fsa_creator_a.GetFsa();

    std::vector<Arc> arcs_b = {
        {0, 1, 1},
        {1, 3, 1},
        {1, 2, 2},
        {2, 3, 1},
    };
    FsaCreator fsa_creator_b(arcs_b, 3);
    const auto &b = fsa_creator_b.GetFsa();

    Intersection intersection(a, b);
    Array2Size<int32_t> fsa_size;
    intersection.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &c = fsa_creator_out.GetFsa();

    std::vector<int32_t> arc_map_a(fsa_size.size2);
    std::vector<int32_t> arc_map_b(fsa_size.size2);
    bool status =
        intersection.GetOutput(&c, arc_map_a.data(), arc_map_b.data());
    EXPECT_TRUE(status);

    std::vector<int32_t> arc_indexes(c.indexes, c.indexes + c.size1 + 1);
    std::vector<Arc> arcs(c.data, c.data + c.size2);
    std::vector<Arc> arcs_c = {
        {0, 1, 1}, {1, 2, 0}, {1, 3, 1}, {1, 4, 2}, {2, 5, 1},
        {2, 3, 1}, {2, 6, 2}, {3, 3, 0}, {6, 6, 0}, {6, 7, 1},
    };
    ASSERT_EQ(arc_indexes.size(), 9);
    EXPECT_THAT(arc_indexes,
                ::testing::ElementsAre(0, 1, 4, 7, 8, 8, 8, 10, 10));
    ASSERT_EQ(arcs.size(), arcs_c.size());
    for (std::size_t i = 0; i != arcs_c.size(); ++i)
      EXPECT_EQ(arcs[i], arcs_c[i]);

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
}  // namespace k2
