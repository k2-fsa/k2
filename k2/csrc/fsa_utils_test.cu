/**
 * @brief Unittest for fsa utils.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include "k2/csrc/fsa_utils.h"

namespace k2 {

// clang-format off
bool operator==(const Arc &a, const Arc &b) {
  return a.src_state == b.src_state && \
         a.dest_state == b.dest_state && \
         a.symbol == b.symbol && \
         fabs(a.score - b.score) < 1e-6;
}
// clang-format on

TEST(FsaFromString, Acceptor) {
  // src_state dst_state label cost
  std::string s = R"(0 1 2   -1.2
    0 2  10 -2.2
    1 3  3  -3.2
    1 6 -1  -4.2
    2 6 -1  -5.2
    2 4  2  -6.2
    3 6 -1  -7.2
    5 0  1  -8.2
    6
  )";

  auto fsa = FsaFromString(s);

  EXPECT_EQ(fsa.NumAxes(), 2);
  EXPECT_EQ(fsa.shape.Dim0(), 7);         // there are 7 states
  EXPECT_EQ(fsa.shape.NumElements(), 8);  // there are 8 arcs
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 2, -1.2f}));
  EXPECT_EQ((fsa[{0, 1}]), (Arc{0, 2, 10, -2.2f}));
  EXPECT_EQ((fsa[{0, 2}]), (Arc{1, 3, 3, -3.2f}));
  EXPECT_EQ((fsa[{0, 3}]), (Arc{1, 6, -1, -4.2f}));
  EXPECT_EQ((fsa[{0, 4}]), (Arc{2, 6, -1, -5.2f}));
  EXPECT_EQ((fsa[{0, 5}]), (Arc{2, 4, 2, -6.2f}));
  EXPECT_EQ((fsa[{0, 6}]), (Arc{3, 6, -1, -7.2f}));
  EXPECT_EQ((fsa[{0, 7}]), (Arc{5, 0, 1, -8.2f}));
}

}  // namespace k2
