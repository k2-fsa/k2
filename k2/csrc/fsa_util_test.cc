// k2/csrc/fsa_util_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa_renderer.h"

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
0 1 10
0 2 20
1 3 30
1 6 60
2 6 3
2 4 2
0 0 0
5 0 1
6
)";
  auto fsa = StringToFsa(s);
  FsaRenderer renderer(*fsa);
  std::cerr << renderer.Render();
}

}  // namespace k2
