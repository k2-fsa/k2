/**
 * @brief Unittest for fsa utils.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
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

TEST(FsaFromString, K2Acceptor) {
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

  {
    auto fsa = FsaFromString(s);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);

    EXPECT_EQ(fsa.NumAxes(), 2);
    EXPECT_EQ(fsa.shape.Dim0(), 7);         // there are 7 states
    EXPECT_EQ(fsa.shape.NumElements(), 8);  // there are 8 arcs
    // Arc sorting order: src_state, symbol, dest_state, score.
    EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 2, -1.2f}));
    EXPECT_EQ((fsa[{0, 1}]), (Arc{0, 2, 10, -2.2f}));
    EXPECT_EQ((fsa[{0, 2}]), (Arc{1, 3, 3, -3.2f}));
    EXPECT_EQ((fsa[{0, 3}]), (Arc{1, 6, -1, -4.2f}));
    EXPECT_EQ((fsa[{0, 4}]), (Arc{2, 6, -1, -5.2f}));
    EXPECT_EQ((fsa[{0, 5}]), (Arc{2, 4, 2, -6.2f}));
    EXPECT_EQ((fsa[{0, 6}]), (Arc{3, 6, -1, -7.2f}));
    EXPECT_EQ((fsa[{0, 7}]), (Arc{5, 0, 1, -8.2f}));
  }
}

TEST(FsaFromString, OpenFstAcceptor) {
  // src_state dst_state label cost
  std::string s = R"(0 1 2   -1.2
    0 2  10 -2.2
    1 3  3  -3.2
    1 6  4  -4.2
    2 6  5  -5.2
    3 6  7  -7.2
    2 4  2  -6.2
    5 7  1  -8.2
    7 -2.3
    6 -1.2
  )";

  {
    auto fsa = FsaFromString(s, true);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);

    EXPECT_EQ(fsa.NumAxes(), 2);
    EXPECT_EQ(fsa.shape.Dim0(), 9);          // there are 9 states
    EXPECT_EQ(fsa.shape.NumElements(), 10);  // there are 10 arcs
    // Arc sorting order: src_state, symbol, dest_state, score.
    EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 2, 1.2f}));
    EXPECT_EQ((fsa[{0, 1}]), (Arc{0, 2, 10, 2.2f}));
    EXPECT_EQ((fsa[{0, 2}]), (Arc{1, 3, 3, 3.2f}));
    EXPECT_EQ((fsa[{0, 3}]), (Arc{1, 6, 4, 4.2f}));
    EXPECT_EQ((fsa[{0, 4}]), (Arc{2, 6, 5, 5.2f}));
    EXPECT_EQ((fsa[{0, 5}]), (Arc{2, 4, 2, 6.2f}));
    EXPECT_EQ((fsa[{0, 6}]), (Arc{3, 6, 7, 7.2f}));
    EXPECT_EQ((fsa[{0, 7}]), (Arc{5, 7, 1, 8.2f}));
    EXPECT_EQ((fsa[{0, 8}]), (Arc{6, 8, -1, 1.2f}));
    EXPECT_EQ((fsa[{0, 9}]), (Arc{7, 8, -1, 2.3f}));
  }
}

TEST(FsaFromString, K2Transducer) {
  // src_state dst_state label aux_label cost
  std::string s = R"(0 1 2 22  -1.2
    0 2  10 100 -2.2
    1 3  3  33  -3.2
    1 6 -1  16  -4.2
    2 6 -1  26  -5.2
    2 4  2  22  -6.2
    3 6 -1  36  -7.2
    5 0  1  50  -8.2
    6
  )";

  {
    Array1<int32_t> aux_labels;
    auto fsa = FsaFromString(s, false, &aux_labels);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(aux_labels.Context()->GetDeviceType(), kCpu);

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

    EXPECT_EQ(aux_labels[0], 22);
    EXPECT_EQ(aux_labels[1], 100);
    EXPECT_EQ(aux_labels[2], 33);
    EXPECT_EQ(aux_labels[3], 16);
    EXPECT_EQ(aux_labels[4], 26);
    EXPECT_EQ(aux_labels[5], 22);
    EXPECT_EQ(aux_labels[6], 36);
    EXPECT_EQ(aux_labels[7], 50);
  }
}

TEST(FsaFromString, OpenFstTransducer) {
  // src_state dst_state label aux_label cost
  std::string s = R"(0 1 2 22  -1.2
    0 2  10 100 -2.2
    1 3  3  33  -3.2
    1 6  4  16  -4.2
    6 -1.2
    2 6  5  26  -5.2
    3 6  7  36  -7.2
    2 4  2  22  -6.2
    5 7  1  50  -8.2
    7 -2.3
  )";

  {
    Array1<int32_t> aux_labels;
    auto fsa = FsaFromString(s, true, &aux_labels);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(aux_labels.Context()->GetDeviceType(), kCpu);

    EXPECT_EQ(fsa.NumAxes(), 2);
    EXPECT_EQ(fsa.shape.Dim0(), 9);          // there are 9 states
    EXPECT_EQ(fsa.shape.NumElements(), 10);  // there are 10 arcs
    EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 2, 1.2f}));
    EXPECT_EQ((fsa[{0, 1}]), (Arc{0, 2, 10, 2.2f}));
    EXPECT_EQ((fsa[{0, 2}]), (Arc{1, 3, 3, 3.2f}));
    EXPECT_EQ((fsa[{0, 3}]), (Arc{1, 6, 4, 4.2f}));
    EXPECT_EQ((fsa[{0, 4}]), (Arc{2, 6, 5, 5.2f}));
    EXPECT_EQ((fsa[{0, 5}]), (Arc{2, 4, 2, 6.2f}));
    EXPECT_EQ((fsa[{0, 6}]), (Arc{3, 6, 7, 7.2f}));
    EXPECT_EQ((fsa[{0, 7}]), (Arc{5, 7, 1, 8.2f}));
    EXPECT_EQ((fsa[{0, 8}]), (Arc{6, 8, -1, 1.2f}));
    EXPECT_EQ((fsa[{0, 9}]), (Arc{7, 8, -1, 2.3f}));

    EXPECT_EQ(aux_labels[0], 22);
    EXPECT_EQ(aux_labels[1], 100);
    EXPECT_EQ(aux_labels[2], 33);
    EXPECT_EQ(aux_labels[3], 16);
    EXPECT_EQ(aux_labels[4], 26);
    EXPECT_EQ(aux_labels[5], 22);
    EXPECT_EQ(aux_labels[6], 36);
    EXPECT_EQ(aux_labels[7], 50);
    EXPECT_EQ(aux_labels[8], 0);
    EXPECT_EQ(aux_labels[9], 0);
  }
}

// TODO(fangjun): write code to check the printed
// strings matching expected ones.
TEST(FsaToString, Acceptor) {
  // src_state dst_state label cost
  std::string s = R"(0 1 2   -1.2
    0 2  10 -2.2
    1 5  -1  -3.2
    5
  )";
  auto fsa = FsaFromString(s);
  auto str = FsaToString(fsa);
  K2_LOG(INFO) << "\n" << str;

  str = FsaToString(fsa, true);
  K2_LOG(INFO) << "\n---negating---\n" << str;
}

TEST(FsaToString, Transducer) {
  // src_state dst_state label aux_label cost
  std::string s = R"(0 1 2 100 -1.2
    0 2  10 200 -2.2
    1 5  -1 300  -3.2
    5
  )";
  Array1<int32_t> aux_labels;
  auto fsa = FsaFromString(s, false, &aux_labels);
  auto str = FsaToString(fsa, false, &aux_labels);
  K2_LOG(INFO) << "\n" << str;

  str = FsaToString(fsa, true, &aux_labels);
  K2_LOG(INFO) << "\n---negating---\n" << str;
}

}  // namespace k2
