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
  std::string s = R"(
0 1 2
0 2 1
1 2 0
1 3 5
2 3 6
3
)";

  auto fsa = StringToFsa(s);

  std::vector<int32_t> arc_index(10);  // an arbitrary number
  std::vector<int32_t> end_index(20);

  GetEnteringArcs(*fsa, &arc_index, &end_index);

  ASSERT_EQ(end_index.size(), 4u);  // there are 4 states
  ASSERT_EQ(arc_index.size(), 5u);  // there are 5 arcs

  EXPECT_EQ(end_index[0], 0);  // state 0 has no entering arcs

  EXPECT_EQ(end_index[1], 1);  // state 1 has one entering arc
  EXPECT_EQ(arc_index[0], 0);  // arc index 0 from state 0

  EXPECT_EQ(end_index[2], 3);  // state 2 has two entering arcs
  EXPECT_EQ(arc_index[1], 1);  // arc index 1 from state 0
  EXPECT_EQ(arc_index[2], 2);  // arc index 2 from state 1

  EXPECT_EQ(end_index[3], 5);  // state 3 has two entering arcs
  EXPECT_EQ(arc_index[3], 3);  // arc index 3 from state 1
  EXPECT_EQ(arc_index[4], 4);  // arc index 4 from state 2
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
  auto fsa = StringToFsa(s);
  ASSERT_NE(fsa.get(), nullptr);

  const auto &arc_indexes = fsa->arc_indexes;
  const auto &arcs = fsa->arcs;

  ASSERT_EQ(arc_indexes.size(), 8u);
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

  Fsa fsa;
  GenerateRandFsa(opts, &fsa);

  EXPECT_TRUE(IsAcyclic(fsa));

  // some states and arcs may be removed due to `Connect`.
  EXPECT_LE(fsa.NumStates(), opts.num_states);
  EXPECT_LE(fsa.arcs.size(), opts.num_arcs);

  EXPECT_FALSE(IsEmpty(fsa));
}

}  // namespace k2
