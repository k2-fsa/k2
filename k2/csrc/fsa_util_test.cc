// k2/csrc/fsa_util_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/properties.h"

namespace k2 {

TEST(FsaUtil, GetEnteringArcs) {
  std::vector<Arc> arcs = {
      {0, 1, 2}, {0, 2, 1}, {1, 2, 0}, {1, 3, 5}, {2, 3, 6}};
  FsaCreator fsa_creator(arcs, 3);
  const auto &fsa = fsa_creator.GetFsa();
  Array2Storage<int32_t *, int32_t> arc_indexes_storage({fsa.size1, fsa.size2},
                                                        1);
  auto &arc_indexes = arc_indexes_storage.GetArray2();
  GetEnteringArcs(fsa, &arc_indexes);

  ASSERT_EQ(arc_indexes.size1, 4);  // there are 4 states
  ASSERT_EQ(arc_indexes.size2, 5);  // there are 5 arcs

  // state 0 has no entering arcs
  EXPECT_EQ(arc_indexes.indexes[0], 0);
  EXPECT_EQ(arc_indexes.indexes[1], 0);

  // state 1 has one entering arc
  EXPECT_EQ(arc_indexes.indexes[2], 1);
  // arc index 0 from state 0
  EXPECT_EQ(arc_indexes.data[0], 0);

  // state 2 has two entering arcs
  EXPECT_EQ(arc_indexes.indexes[3], 3);
  // arc index 1 from state 0
  EXPECT_EQ(arc_indexes.data[1], 1);
  // arc index 2 from state 1
  EXPECT_EQ(arc_indexes.data[2], 2);

  // state 3 has two entering arcs
  EXPECT_EQ(arc_indexes.indexes[4], 5);
  // arc index 3 from state 1
  EXPECT_EQ(arc_indexes.data[3], 3);
  // arc index 4 from state 2
  EXPECT_EQ(arc_indexes.data[4], 4);
}

TEST(FsaUtil, StringToFsa) {
  std::string s = R"(
0 1 2
0 2 10
1 3 3
1 6 6
2 6 1
2 4 2
5 0 1
6
)";
  StringToFsa fsa_creator(s);
  Array2Size<int32_t> fsa_size;
  fsa_creator.GetSizes(&fsa_size);

  FsaCreator fsa_creator_out(fsa_size);
  auto &fsa = fsa_creator_out.GetFsa();
  fsa_creator.GetOutput(&fsa);

  ASSERT_FALSE(IsEmpty(fsa));

  ASSERT_EQ(fsa.size1, 7);
  ASSERT_EQ(fsa.size2, 7);

  std::vector<int32_t> arc_indexes(fsa.indexes, fsa.indexes + fsa.size1 + 1);
  std::vector<Arc> arcs(fsa.data, fsa.data + fsa.size2);

  EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 4, 6, 6, 6, 7, 7));

  std::vector<Arc> expected_arcs = {
      {0, 1, 2}, {0, 2, 10}, {1, 3, 3}, {1, 6, 6},
      {2, 6, 1}, {2, 4, 2},  {5, 0, 1},
  };

  auto n = static_cast<int32_t>(expected_arcs.size());
  for (auto i = 0; i != n; ++i) {
    EXPECT_EQ(arcs[i], expected_arcs[i]);
  }
}

TEST(FsaUtil, RandFsa) {
  RandFsaOptions opts;
  opts.num_syms = 20;
  opts.num_states = 10;
  opts.num_arcs = 20;
  opts.allow_empty = false;
  opts.acyclic = true;
  opts.seed = 20200517;

  RandFsaGenerator generator(opts);
  Array2Size<int32_t> fsa_size;
  generator.GetSizes(&fsa_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa = fsa_creator.GetFsa();
  generator.GetOutput(&fsa);

  EXPECT_TRUE(IsAcyclic(fsa));

  // some states and arcs may be removed due to `Connect`.
  EXPECT_LE(fsa.NumStates(), opts.num_states);
  EXPECT_LE(fsa.size2, opts.num_arcs);

  EXPECT_FALSE(IsEmpty(fsa));
}

TEST(FsaUtil, ReorderArcs) {
  {
    // empty input arcs
    std::vector<Arc> arcs;
    FsaCreator fsa_creator({0, 0});
    auto &fsa = fsa_creator.GetFsa();
    std::vector<int32_t> arc_map = {1};  // dirty data
    ReorderArcs(arcs, &fsa, &arc_map);
    EXPECT_TRUE(IsEmpty(fsa));
    EXPECT_TRUE(arc_map.empty());
  }

  {
    std::vector<Arc> arcs = {{0, 1, 1}, {0, 2, 2},  {2, 3, 3},  {2, 4, 4},
                             {1, 2, 5}, {2, 5, -1}, {4, 5, -1}, {3, 5, -1}};
    FsaCreator fsa_creator({6, 8});
    auto &fsa = fsa_creator.GetFsa();
    std::vector<int32_t> arc_map;
    ReorderArcs(arcs, &fsa, &arc_map);

    std::vector<Arc> expected_arcs = {{0, 1, 1},  {0, 2, 2}, {1, 2, 5},
                                      {2, 3, 3},  {2, 4, 4}, {2, 5, -1},
                                      {3, 5, -1}, {4, 5, -1}};
    ASSERT_EQ(fsa.size2, arcs.size());
    ASSERT_EQ(fsa.size2, expected_arcs.size());
    for (auto i = 0; i != expected_arcs.size(); ++i) {
      EXPECT_EQ(fsa.data[i], expected_arcs[i]);
    }
    ASSERT_EQ(fsa.size1, 6);
    std::vector<int32_t> arc_indexes(fsa.indexes, fsa.indexes + fsa.size1 + 1);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 3, 6, 7, 8, 8));
    ASSERT_EQ(arc_map.size(), expected_arcs.size());
    EXPECT_THAT(arc_map, ::testing::ElementsAre(0, 1, 4, 2, 3, 5, 7, 6));
  }
}

TEST(FsaUtil, FsaCreator) {
  {
    // create empty fsa
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
    EXPECT_TRUE(IsEmpty(fsa));

    // test `begin` and `end` for empty fsa
    for (const auto &arc : fsa) {
      Arc tmp = arc;
    }
    for (auto &arc : fsa) {
      arc.label = 1;
    }
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 1}, {0, 2, 2}, {2, 3, 3}, {2, 4, 4}, {3, 5, -1}};
    FsaCreator fsa_creator(arcs, 6);
    const auto &fsa = fsa_creator.GetFsa();

    std::vector<int32_t> arc_indexes(fsa.indexes, fsa.indexes + fsa.size1 + 1);
    ASSERT_EQ(fsa.size1, 7);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 2, 4, 5, 5, 5, 5));
    ASSERT_EQ(fsa.size2, arcs.size());
    for (auto i = 0; i != arcs.size(); ++i) {
      EXPECT_EQ(fsa.data[i], arcs[i]);
    }
  }
}

TEST(FsaAlgo, CreateFsa) {
  {
    // clang-format off
    std::vector<Arc> arcs = {
      {0, 3, 3},
      {0, 2, 2},
      {2, 3, 3},
      {2, 4, 4},
      {3, 1, 1},
      {1, 4, 4},
      {1, 8, 8},
      {4, 8, 8},
      {8, 6, 6},
      {8, 7, 7},
      {6, 7, 7},
      {7, 5, 5},
    };
    // clang-format on
    Array2Size<int32_t> fsa_size;
    fsa_size.size1 = 9;            // num_states
    fsa_size.size2 = arcs.size();  // num_arcs
    FsaCreator fsa_creator(fsa_size);
    auto &fsa_out = fsa_creator.GetFsa();
    std::vector<int32_t> arc_map;
    CreateFsa(arcs, &fsa_out, &arc_map);

    EXPECT_EQ(arc_map.size(), arcs.size());
    EXPECT_THAT(arc_map,
                ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
  }
}
}  // namespace k2
