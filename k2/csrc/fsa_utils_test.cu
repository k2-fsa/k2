/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

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
    Array2<int32_t> aux_labels_array;
    int32_t num_aux_labels = 1;
    auto fsa = FsaFromString(s, false, num_aux_labels, &aux_labels_array);
    Array1<int32_t> aux_labels = aux_labels_array.Row(0);
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


TEST(FsaFromString, K2TransducerRagged1) {
  // src_state dst_state label aux_label cost
  std::string s = R"(0 1 2 22 [11] -1.2
0 2  10 100 [12] -2.2
1 3  3  33  [] -3.2
1 6 -1  16  [13 14] -4.2
2 6 -1  26 [15]  -5.2
2 4  2  22 [] -6.2
3 6 -1  36 [16 17] -7.2
5 0  1  50 [18] -8.2
6
)";

  {
    Array2<int32_t> aux_labels_array;
    Ragged<int32_t> r,
        r2("[ [11][12][][13 14][15][][16 17][18]]");
    int32_t num_aux_labels = 1, num_ragged_labels = 1;
    auto fsa = FsaFromString(s, false, num_aux_labels, &aux_labels_array,
                             num_ragged_labels, &r);
    Array1<int32_t> aux_labels = aux_labels_array.Row(0);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(aux_labels.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(Equal(r, r2), true);

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

TEST(FsaFromString, K2TransducerRagged2) {
  // src_state dst_state label aux_label cost
  std::string s = R"(0 1 2 22 [11] []-1.2
0 2  10 100 [12] []-2.2
1 3  3  33  [] [101]-3.2
1 6 -1  16  [13 14] []-4.2
2 6 -1  26 [15][]  -5.2
2 4  2  22 [] []-6.2
3 6 -1  36 [16 17] []-7.2
5 0  1  50 [18] [] -8.2
6
)";

  {
    Array2<int32_t> aux_labels_array;
    std::vector<Ragged<int32_t> > r(2);
    Ragged<int32_t>
        r1b("[ [11][12][][13 14][15][][16 17][18]]"),
        r2b("[[][][101][][][][][]]");
    int32_t num_aux_labels = 1, num_ragged_labels = 2;
    auto fsa = FsaFromString(s, false, num_aux_labels, &aux_labels_array,
                             num_ragged_labels, r.data());
    Array1<int32_t> aux_labels = aux_labels_array.Row(0);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(aux_labels.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(Equal(r[0], r1b), true);
    EXPECT_EQ(Equal(r[1], r2b), true);

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
    Array2<int32_t> aux_labels_array;
    auto fsa = FsaFromString(s, true, 1, &aux_labels_array);
    EXPECT_EQ(fsa.Context()->GetDeviceType(), kCpu);
    EXPECT_EQ(aux_labels_array.Context()->GetDeviceType(), kCpu);

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

    Array1<int32_t> aux_labels = aux_labels_array.Row(0);
    EXPECT_EQ(aux_labels[0], 22);
    EXPECT_EQ(aux_labels[1], 100);
    EXPECT_EQ(aux_labels[2], 33);
    EXPECT_EQ(aux_labels[3], 16);
    EXPECT_EQ(aux_labels[4], 26);
    EXPECT_EQ(aux_labels[5], 22);
    EXPECT_EQ(aux_labels[6], 36);
    EXPECT_EQ(aux_labels[7], 50);
    EXPECT_EQ(aux_labels[8], -1);
    EXPECT_EQ(aux_labels[9], -1);
  }
}

TEST(FsaFromString, OpenFstAcceptorNonZeroStart) {
  std::string s = R"(
    1 0 0 0.1
    0 0 4 0.3
    0 0 3 0.2
    0 0.4
  )";
  Fsa fsa = FsaFromString(s, true);
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 0, -0.1f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 1, 4, -0.3f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 1, 3, -0.2f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 2, -1, -0.4f}));
}

TEST(FsaFromString, OpenFstAcceptorNonZeroStartRagged1) {
  std::string s = R"(
    1 0 0 [ 10 ] 0.1
    0 0 4 [ ] 0.3
    0 0 3 [ 11 12] 0.2
    0 0.4
  )";
  Ragged<int32_t> r,
      r2("[ [10] []  [ 11 12 ] [] ]");
  Fsa fsa = FsaFromString(s, true, 0, nullptr, 1, &r);
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 0, -0.1f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 1, 4, -0.3f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 1, 3, -0.2f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 2, -1, -0.4f}));
  EXPECT_EQ(Equal(r, r2), true) << "r = " << r;
}


TEST(FsaFromString, OpenFstAcceptorNonZeroStartRagged2) {
  std::string s = R"(
    1 0 0 [ 10 ] [] 0.1
    0 0 4 [ ] [ 90 91 ]0.3
    0 0 3 [ 11 12] [ 92 93 94 ] 0.2
    0 0.4
  )";
  std::vector<Ragged<int32_t>> r(2);

  Ragged<int32_t> r1b("[ [10] []  [ 11 12 ] []  ]"),
      r2b("[ [] [ 90 91 ] [ 92 93 94 ] [] ]");
  Fsa fsa = FsaFromString(s, true, 0, nullptr, 2, r.data());
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 0, -0.1f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 1, 4, -0.3f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 1, 3, -0.2f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 2, -1, -0.4f}));
  EXPECT_EQ(Equal(r[0], r1b), true);
  EXPECT_EQ(Equal(r[1], r2b), true);
}


TEST(FsaFromString, OpenFstAcceptorNonZeroStartCase2) {
  std::string s = R"(
    1 3 10 0.1
    1 3 20 0.2
    1 0 90 0.8
    3 0 30 0.3
    3 0 40 0.4
    3 1 6 0.33
    0 4 50 0.5
    0 3 0 0.55
    0 1 9 0.9
    4 0.6
  )";
  Fsa fsa = FsaFromString(s, true);
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 3, 10, -0.1f}));
  EXPECT_EQ((fsa[{0, 1}]), (Arc{0, 3, 20, -0.2f}));
  EXPECT_EQ((fsa[{0, 2}]), (Arc{0, 1, 90, -0.8f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 4, 50, -0.5f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 3, 0, -0.55f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 0, 9, -0.9f}));
  EXPECT_EQ((fsa[{3, 0}]), (Arc{3, 1, 30, -0.3f}));
  EXPECT_EQ((fsa[{3, 1}]), (Arc{3, 1, 40, -0.4f}));
  EXPECT_EQ((fsa[{3, 2}]), (Arc{3, 0, 6, -0.33f}));
  EXPECT_EQ((fsa[{4, 0}]), (Arc{4, 5, -1, -0.6f}));
}

TEST(FsaFromString, OpenFstTransducerNonZeroStart) {
  std::string s = R"(
    1 0 0 0 0.1
    0 0 4 40 0.3
    0 0 3 30 0.2
    0 0.4
  )";
  Array2<int32_t> aux_labels_array;
  Fsa fsa = FsaFromString(s, true, 1, &aux_labels_array);
  CheckArrayData(aux_labels_array.Row(0), {0, 40, 30, -1});
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 0, -0.1f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 1, 4, -0.3f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 1, 3, -0.2f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 2, -1, -0.4f}));
}

TEST(FsaFromString, OpenFstTransducerNonZeroStartRagged1) {
  std::string s = R"(
    1 0 0 0 [10] 0.1
    0 0 4 40 [] 0.3
    0 0 3 30 [11 12] 0.2
    0 0.4
  )";
  Ragged<int32_t> r,
      r2("[ [10] []  [ 11 12 ] []]");
  Array2<int32_t> aux_labels_array;
  Fsa fsa = FsaFromString(s, true, 1, &aux_labels_array, 1, &r);
  CheckArrayData(aux_labels_array.Row(0), {0, 40, 30, -1});
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 0, -0.1f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 1, 4, -0.3f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 1, 3, -0.2f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 2, -1, -0.4f}));
  EXPECT_EQ(Equal(r, r2), true);
}


TEST(FsaFromString, OpenFstTransducerZeroStartRagged1) {
  std::string s = R"(
    0 1 0 0 [10] 0.1
    1 1 4 40 [] 0.3
    1 1 3 30 [11 12] 0.2
    1 0.4
  )";
  Ragged<int32_t> r,
      r2("[ [10] []  [ 11 12 ] []]");
  Array2<int32_t> aux_labels_array;
  Fsa fsa = FsaFromString(s, true, 1, &aux_labels_array, 1, &r);
  CheckArrayData(aux_labels_array.Row(0), {0, 40, 30, -1});
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 0, -0.1f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 1, 4, -0.3f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 1, 3, -0.2f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 2, -1, -0.4f}));
  EXPECT_EQ(Equal(r, r2), true);
}


TEST(FsaFromString, OpenFstTransducerNonZeroStartCase2) {
  std::string s = R"(
    1 3 10 100 0.1
    1 3 20 200 0.2
    1 0 90 8 0.8
    3 0 30 300 0.3
    3 0 40 400 0.4
    3 1 6 8 0.33
    0 4 50 500 0.5
    0 3 0 3 0.55
    0 1 9 10 0.9
    4 0.6
  )";
  Array2<int32_t> aux_labels;
  Fsa fsa = FsaFromString(s, true, 1, &aux_labels);
  CheckArrayData(aux_labels.Row(0), {100, 200, 8, 500, 3, 10, 300, 400, 8, -1});
  EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 3, 10, -0.1f}));
  EXPECT_EQ((fsa[{0, 1}]), (Arc{0, 3, 20, -0.2f}));
  EXPECT_EQ((fsa[{0, 2}]), (Arc{0, 1, 90, -0.8f}));
  EXPECT_EQ((fsa[{1, 0}]), (Arc{1, 4, 50, -0.5f}));
  EXPECT_EQ((fsa[{1, 1}]), (Arc{1, 3, 0, -0.55f}));
  EXPECT_EQ((fsa[{1, 2}]), (Arc{1, 0, 9, -0.9f}));
  EXPECT_EQ((fsa[{3, 0}]), (Arc{3, 1, 30, -0.3f}));
  EXPECT_EQ((fsa[{3, 1}]), (Arc{3, 1, 40, -0.4f}));
  EXPECT_EQ((fsa[{3, 2}]), (Arc{3, 0, 6, -0.33f}));
  EXPECT_EQ((fsa[{4, 0}]), (Arc{4, 5, -1, -0.6f}));
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
  Array2<int32_t> aux_labels_array;
  auto fsa = FsaFromString(s, false, 1, &aux_labels_array);
  Array1<int32_t> aux_labels = aux_labels_array.Row(0);
  auto str = FsaToString(fsa, false, 1, &aux_labels);
  K2_LOG(INFO) << "\n" << str;

  str = FsaToString(fsa, true, 1, &aux_labels);
  K2_LOG(INFO) << "\n---negating---\n" << str;
}

TEST(FsaUtilsTest, TestGetDestStates) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    // test with simple case should be good enough
    std::string s1 = R"(0 1 1 0
0 2  1 0
0 3  1 0
0 3  2 0
1 2  1 0
1 3  1 0
3 4  1 0
3 5  -1 0
4 5  -1 0
5
)";

    std::string s2 = R"(0 1 1 0
0 2  1 0
1 2  1 0
1 3  1 0
2 3  1 0
2 4  -1 0
4
)";

    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    {
      // as_idx01 = false
      Array1<int32_t> result = GetDestStates(fsa_vec, false);
      ASSERT_EQ(result.Dim(), fsa_vec.NumElements());
      result = result.To(cpu);
      std::vector<int32_t> cpu_data(result.Data(),
                                    result.Data() + result.Dim());
      EXPECT_THAT(cpu_data, ::testing::ElementsAre(1, 2, 3, 3, 2, 3, 4, 5, 5, 1,
                                                   2, 2, 3, 3, 4));
    }

    {
      // as_idx01 = true
      Array1<int32_t> result = GetDestStates(fsa_vec, true);
      ASSERT_EQ(result.Dim(), fsa_vec.NumElements());
      result = result.To(cpu);
      std::vector<int32_t> cpu_data(result.Data(),
                                    result.Data() + result.Dim());
      EXPECT_THAT(cpu_data, ::testing::ElementsAre(1, 2, 3, 3, 2, 3, 4, 5, 5, 7,
                                                   8, 8, 9, 9, 10));
    }
  }
}

class StatesBatchSuiteTest : public ::testing::Test {
 protected:
  StatesBatchSuiteTest() {
    std::string s1 = R"(0 1 1 1
    0 2  1 2
    0 3  1 2
    0 3  2 3
    1 2  1 4
    1 3  1 5
    3 4  1 6
    3 5  -1 7
    4 5  -1 8
    5
    )";

    std::string s2 = R"(0 1 1 1
    0 2  1 2
    1 2  1 3
    1 3  1 4
    2 3  1 5
    2 4  -1 6
    4
  )";

    std::string s3 = R"(0 2 1 1
    1 2  1 2
    1 3  1 3
    1 4  1 4
    2 3  1 5
    2 4  1 6
    3 4  1 7
    4 5  -1 8
    5
  )";

    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa fsa3 = FsaFromString(s3);
    Fsa *fsa_array[] = {&fsa1, &fsa2, &fsa3};
    fsa_vec_ = CreateFsaVec(3, &fsa_array[0]);
  }

  FsaVec fsa_vec_;
};

// Note states_batches should be indexed with [fsa][batch][state]
void CheckGetStatesBatchesResult(const FsaVec &fsas_vec_in,
                                 const Ragged<int32_t> &states_batches_in,
                                 bool transpose) {
  ContextPtr cpu = GetCpuContext();
  FsaVec fsa_vec = fsas_vec_in.To(cpu);
  Ragged<int32_t> states_batches = states_batches_in.To(cpu);

  int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
          num_batches = states_batches.TotSize(1);
  const int32_t *fsa_row_splits1 = fsa_vec.RowSplits(1).Data();
  const int32_t *row_splits1_data = states_batches.RowSplits(1).Data();
  // num-batches in each fsa should not be greater than num-states
  if (!transpose) {
    for (int32_t n = 0; n < num_fsas; ++n) {
      int32_t num_batches = row_splits1_data[n + 1] - row_splits1_data[n];
      int32_t num_states_this_fsa = fsa_row_splits1[n + 1] - fsa_row_splits1[n];
      EXPECT_LE(num_batches, num_states_this_fsa);
      if (num_states_this_fsa > 0) {
        EXPECT_GT(num_batches, 0);
      }
    }
  }

  // values should be [0,1, ..., num_states - 1]
  Array1<int32_t> states = Range(cpu, num_states, 0, 1);
  CheckArrayData(states_batches.values, states);

  Array1<int32_t> max_states_in_batches(cpu, num_batches);
  MaxPerSublist(states_batches, -1, &max_states_in_batches);
  Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
  const int32_t *batch_ids_data = states_batches.RowIds(2).Data();
  const int32_t *batch_state_data = states_batches.values.Data();
  const int32_t *max_states_data = max_states_in_batches.Data();
  const int32_t *dest_states_data = dest_states.Data();
  const int32_t *fsa_row_splits2_data = fsa_vec.RowSplits(2).Data();
  for (int32_t i = 0; i < num_states; ++i) {
    int32_t batch_idx = batch_ids_data[i];
    int32_t state_idx = batch_state_data[i];
    EXPECT_EQ(state_idx,
              i);  // as state_batches is indexed with [fsa][batch][state]
    int32_t max_state_this_batch = max_states_data[batch_idx];
    EXPECT_LE(state_idx, max_state_this_batch);
    int32_t arc_begin = fsa_row_splits2_data[state_idx];
    int32_t arc_end = fsa_row_splits2_data[state_idx + 1];
    for (int32_t idx = arc_begin; idx != arc_end; ++idx) {
      // states in each batch only have arcs to later numbered batches
      int32_t dest_state_this_arc = dest_states_data[idx];
      EXPECT_GT(dest_state_this_arc, max_state_this_batch);
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestGetStateBatches) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1);
      EXPECT_EQ(num_fsas, 3);

      {
        // no transpose: [fsa_idx][batch_idx][state]
        Ragged<int32_t> result = GetStateBatches(fsa_vec, false);
        EXPECT_EQ(result.Dim0(), num_fsas);
        ASSERT_EQ(result.NumElements(), num_states);
        CheckGetStatesBatchesResult(fsa_vec, result, false);
      }

      {
        // transpose: [batch_index][fsa_index][state]
        Ragged<int32_t> result = GetStateBatches(fsa_vec, true);
        result = result.To(cpu);
        // result.Dim0() is num-batches
        EXPECT_EQ(result.TotSize(1), num_fsas * result.Dim0());
        ASSERT_EQ(result.NumElements(), num_states);
        int32_t *row_splits1_data = result.RowSplits(1).Data();
        for (int32_t n = 0; n <= result.Dim0(); ++n) {
          EXPECT_EQ(row_splits1_data[n], n * num_fsas);
        }
        CheckGetStatesBatchesResult(fsa_vec, Transpose(result), true);
      }
    }
  }
  {
    // random case
    for (int32_t i = 0; i != 2; ++i) {
      for (auto &context : {GetCpuContext(), GetCudaContext()}) {
        FsaVec random_fsas = RandomFsaVec();
        FsaVec fsa_vec = random_fsas.To(context);
        int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1);

        {
          // no transpose: [fsa_idx][batch_idx][state]
          Ragged<int32_t> result = GetStateBatches(fsa_vec, false);
          EXPECT_EQ(result.Dim0(), num_fsas);
          ASSERT_EQ(result.NumElements(), num_states);
          CheckGetStatesBatchesResult(fsa_vec, result, false);
        }

        {
          // transpose: [batch_index][fsa_index][state]
          Ragged<int32_t> result = GetStateBatches(fsa_vec, true);
          result = result.To(cpu);
          // result.Dim0() is num-batches
          EXPECT_EQ(result.TotSize(1), num_fsas * result.Dim0());
          ASSERT_EQ(result.NumElements(), num_states);
          int32_t *row_splits1_data = result.RowSplits(1).Data();
          for (int32_t n = 0; n <= result.Dim0(); ++n) {
            EXPECT_EQ(row_splits1_data[n], n * num_fsas);
          }
          CheckGetStatesBatchesResult(fsa_vec, Transpose(result), true);
        }
      }
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestIncomingArc) {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
      CheckArrayData(
          dest_states,
          std::vector<int32_t>{1, 2, 3,  3,  2,  3,  4,  5,  5,  7,  8, 8,
                               9, 9, 10, 13, 13, 14, 15, 14, 15, 15, 16});
      Ragged<int32_t> result = GetIncomingArcs(fsa_vec, dest_states);
      result = result.To(cpu);
      // check states_num in each fsa
      EXPECT_EQ(result.Dim0(), num_fsas);
      CheckArrayData(result.RowSplits(1), fsa_vec.RowSplits(1));
      // check the number of incoming arcs in each state
      EXPECT_EQ(result.TotSize(1), fsa_vec.TotSize(1));
      CheckArrayData(result.RowSplits(2),
                     std::vector<int32_t>{0, 0, 1, 3, 6, 7, 9, 9, 10, 12, 14,
                                          15, 15, 15, 17, 19, 22, 23});
      // check incoming arc ids
      EXPECT_EQ(result.NumElements(), num_arcs);
      CheckArrayData(
          result.values,
          std::vector<int32_t>{0,  1,  4,  2,  3,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 19, 18, 20, 21, 22});
    }
  }
  // TODO(haowen): add random cases
}

TEST_F(StatesBatchSuiteTest, TestLeavingArcIndexBatches) {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
      Ragged<int32_t> result =
          GetLeavingArcIndexBatches(fsa_vec, state_batches);
      result = result.To(cpu);
      ASSERT_EQ(result.NumAxes(), 4);
      // axes 0,1,2 are same with those of state_batches
      RaggedShape sub_shape = RemoveAxis(result.shape, 3);
      for (int32_t i = 1; i != 3; ++i) {
        CheckArrayData(sub_shape.RowSplits(i), state_batches.RowSplits(i));
      }
      // transpose [batch][fsa][state][arc_list] to
      // [fsa][batch][state][arc_list], the element would be sorted as leaving
      // arc orders as in fsa_vec
      Ragged<int32_t> transposed = Transpose(result);
      std::vector<int32_t> arc_ids(num_arcs);
      std::iota(arc_ids.begin(), arc_ids.end(), 0);
      CheckArrayData(transposed.values, arc_ids);
      // check row_ids
      CheckArrayData(transposed.RowIds(3), fsa_vec.RowIds(2));
    }
  }
  // TODO(haowen): add random cases
}

TEST_F(StatesBatchSuiteTest, TestEnteringArcIndexBatches) {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
      Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
      Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
      Ragged<int32_t> result =
          GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);
      result = result.To(cpu);
      ASSERT_EQ(result.NumAxes(), 4);
      // axes 0,1,2 are same with those of state_batches
      RaggedShape sub_shape = RemoveAxis(result.shape, 3);
      for (int32_t i = 1; i != 3; ++i) {
        CheckArrayData(sub_shape.RowSplits(i), state_batches.RowSplits(i));
      }
      // transpose [batch][fsa][state][arc_list] to
      // [fsa][batch][state][arc_list], the element would be sorted as incoming
      // arc orders as in fsa_vec
      Ragged<int32_t> transposed = Transpose(result);
      CheckArrayData(transposed.values, incoming_arcs.values);
      // check row_ids
      CheckArrayData(transposed.RowIds(3), incoming_arcs.RowIds(2));
    }
  }
  // TODO(haowen): add random cases
}

TEST_F(StatesBatchSuiteTest, TestForwardScores) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
      Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
      Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
      Ragged<int32_t> entering_arc_batches =
          GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);

      {
        // max
        Array1<int32_t> entering_arcs;
        Array1<float> scores = GetForwardScores<float>(fsa_vec, state_batches,
                                                       entering_arc_batches,
                                                       false, &entering_arcs);
        EXPECT_EQ(scores.Dim(), num_states);
        K2_LOG(INFO) << "Scores: " << scores
                     << "\n,Entering arcs: " << entering_arcs;
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores = GetForwardScores<float>(cpu_fsa_vec, false);
        CheckArrayData(scores, cpu_scores);
        //  [ 0 1 5 6 12 20 0 1 4 9 10 0 -inf 1 6 13 21 ]
      }
      {
        // logsum
        Array1<double> scores = GetForwardScores<double>(
            fsa_vec, state_batches, entering_arc_batches, true);
        EXPECT_EQ(scores.Dim(), num_states);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<double> cpu_scores = GetForwardScores<double>(cpu_fsa_vec, true);
        CheckArrayData(scores, cpu_scores);
        // [ 0 1 5.04859 6.06588 12.0659 20.0668 0 1 4.12693 9.14293 10.1269 0
        // -inf 1 6 13.0025 21.0025 ]
      }
    }
  }
  {
    // random case
    for (int32_t i = 0; i != 2; ++i) {
      for (auto &context : {GetCpuContext(), GetCudaContext()}) {
        FsaVec random_fsas = RandomFsaVec();
        FsaVec fsa_vec = random_fsas.To(context);
        int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
                num_arcs = fsa_vec.NumElements();

        Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
        Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
        Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
        Ragged<int32_t> entering_arc_batches =
            GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);

        {
          // max
          Array1<int32_t> entering_arcs;
          Array1<float> scores = GetForwardScores<float>(fsa_vec, state_batches,
                                                         entering_arc_batches,
                                                         false, &entering_arcs);
          EXPECT_EQ(scores.Dim(), num_states);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetForwardScores<float>(cpu_fsa_vec, false);
          CheckArrayData(scores, cpu_scores);
        }
        {
          // logsum
          Array1<double> scores = GetForwardScores<double>(
              fsa_vec, state_batches, entering_arc_batches, true);
          EXPECT_EQ(scores.Dim(), num_states);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<double> cpu_scores =
              GetForwardScores<double>(cpu_fsa_vec, true);
          CheckArrayData(scores, cpu_scores);
        }
      }
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestGetTotScores) {
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
      Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
      Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
      Ragged<int32_t> entering_arc_batches =
          GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);

      {
        // max
        Array1<float> scores = GetForwardScores<float>(
            fsa_vec, state_batches, entering_arc_batches, false);
        Array1<float> tot_scores = GetTotScores(fsa_vec, scores);
        EXPECT_EQ(tot_scores.Dim(), num_fsas);
        K2_LOG(INFO) << tot_scores;
        //  [ 20 10 21 ]
      }
      {
        // logsum
        Array1<float> scores = GetForwardScores<float>(
            fsa_vec, state_batches, entering_arc_batches, true);
        Array1<float> tot_scores = GetTotScores(fsa_vec, scores);
        EXPECT_EQ(tot_scores.Dim(), num_fsas);
        K2_LOG(INFO) << tot_scores;
        // [ 20.0668 10.1269 21.0025 ]
      }
    }
  }
  // TODO(haowen): add random cases
}

TEST_F(StatesBatchSuiteTest, TestBackwardScores) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
      Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
      Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
      Ragged<int32_t> entering_arc_batches =
          GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);
      Ragged<int32_t> leaving_arc_batches =
          GetLeavingArcIndexBatches(fsa_vec, state_batches);

      {
        // max
        Array1<float> scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, false);
        EXPECT_EQ(scores.Dim(), num_states);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores =
            GetBackwardScores<float>(cpu_fsa_vec, nullptr, false);
        CheckArrayData(scores, cpu_scores);
        // [ 20 19 -inf 14 8 0 10 9 6 -inf 0 21 22 20 15 8 0 ]
      }
      {
        // logsum
        Array1<float> scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, true);
        EXPECT_EQ(scores.Dim(), num_states);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores =
            GetBackwardScores<float>(cpu_fsa_vec, nullptr, true);
        CheckArrayData(scores, cpu_scores);
        // [ 20.0668 19.0009 -inf 14.0009 8 0 10.1269 9 6 -inf
        // 0 21.0025 22.0206 20.0025 15 8 0 ]
      }
    }
  }
  {
    // random case
    for (int32_t i = 0; i != 2; ++i) {
      for (auto &context : {GetCpuContext(), GetCudaContext()}) {
        FsaVec random_fsas = RandomFsaVec();
        FsaVec fsa_vec = random_fsas.To(context);
        int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
                num_arcs = fsa_vec.NumElements();

        Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
        Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
        Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
        Ragged<int32_t> entering_arc_batches =
            GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);
        Ragged<int32_t> leaving_arc_batches =
            GetLeavingArcIndexBatches(fsa_vec, state_batches);

        {
          // max
          Array1<float> scores = GetBackwardScores<float>(
              fsa_vec, state_batches, leaving_arc_batches, false);
          EXPECT_EQ(scores.Dim(), num_states);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetBackwardScores<float>(cpu_fsa_vec, nullptr, false);
          CheckArrayData(scores, cpu_scores);
        }
        {
          // logsum
          Array1<float> scores = GetBackwardScores<float>(
              fsa_vec, state_batches, leaving_arc_batches, true);
          EXPECT_EQ(scores.Dim(), num_states);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetBackwardScores<float>(cpu_fsa_vec, nullptr, true);
          CheckArrayData(scores, cpu_scores);
        }
      }
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestArcPost) {
  {
    // simple case
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      FsaVec fsa_vec = fsa_vec_.To(context);
      int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
              num_arcs = fsa_vec.NumElements();
      EXPECT_EQ(num_fsas, 3);

      Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
      Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
      Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
      Ragged<int32_t> entering_arc_batches =
          GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);
      Ragged<int32_t> leaving_arc_batches =
          GetLeavingArcIndexBatches(fsa_vec, state_batches);

      {
        // max
        Array1<float> forward_scores = GetForwardScores<float>(
            fsa_vec, state_batches, entering_arc_batches, false);
        Array1<float> backward_scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, false);
        Array1<float> arc_scores =
            GetArcPost(fsa_vec, forward_scores, backward_scores);
        EXPECT_EQ(arc_scores.Dim(), num_arcs);
        K2_LOG(INFO) << arc_scores;
        // [ 20 -inf 16 17 -inf 20 20 13 20 10 8 10 -inf -inf 10 21 -inf -inf
        // -inf 21 15 21 21 ]
      }
    }
  }
  // TODO(haowen): add random cases
}

template <typename FloatType>
void TestBackpropGetForwardScores(FsaVec &fsa_vec_in) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsa_vec = fsa_vec_in.To(context);
    int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
            num_arcs = fsa_vec.NumElements();

    Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
    Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
    Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
    Ragged<int32_t> entering_arc_batches =
        GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);
    Ragged<int32_t> leaving_arc_batches =
        GetLeavingArcIndexBatches(fsa_vec, state_batches);

    {
      // max
      Array1<int32_t> entering_arcs;
      Array1<FloatType> forward_scores = GetForwardScores<FloatType>(
          fsa_vec, state_batches, entering_arc_batches, false, &entering_arcs);
      Array1<FloatType> backward_scores = GetBackwardScores<FloatType>(
          fsa_vec, state_batches, leaving_arc_batches, false);
      // set the forward_scores_deriv_in to all zeros except for a 1 at the
      // final-state. Then the returned derivative will be non-zero values
      // only for those arcs along the best path.
      Array1<FloatType> forward_scores_deriv_in(context, num_states, 0);
      FloatType *forward_scores_deriv_in_data = forward_scores_deriv_in.Data();
      const int32_t *fsa_row_splits1 = fsa_vec.RowSplits(1).Data();
      K2_EVAL(
          context, num_fsas, lambda_set_forward_derivs_in,
          (int32_t fsa_idx)->void {
            int32_t start_state = fsa_row_splits1[fsa_idx],
                    start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
            if (start_state_next_fsa - start_state > 0)
              forward_scores_deriv_in_data[start_state_next_fsa - 1] = 1;
          });
      Array1<FloatType> arc_derivs = BackpropGetForwardScores(
          fsa_vec, state_batches, leaving_arc_batches, false, &entering_arcs,
          forward_scores, forward_scores_deriv_in);
      entering_arcs = entering_arcs.To(cpu);
      Array1<FloatType> expected_arc_deriv(cpu, num_arcs, 0);
      FloatType *expected_arc_deriv_data = expected_arc_deriv.Data();
      Array1<int32_t> fsa_row_splits1_cpu = fsa_vec.RowSplits(1).To(cpu);
      Array1<Arc> cpu_arcs = fsa_vec.values.To(cpu);
      for (int32_t fsa_idx = 0; fsa_idx != num_fsas; ++fsa_idx) {
        int32_t start_state = fsa_row_splits1_cpu[fsa_idx],
                start_state_next_fsa = fsa_row_splits1_cpu[fsa_idx + 1];
        if (start_state_next_fsa > start_state) {
          for (int32_t state = start_state_next_fsa - 1; state > start_state;) {
            if (entering_arcs[state] != -1) {
              int32_t arc_id = entering_arcs[state];
              const Arc &arc = cpu_arcs[arc_id];
              int32_t src_state = arc.src_state + start_state;
              int32_t dst_state = arc.dest_state + start_state;
              ASSERT_EQ(state, dst_state);
              expected_arc_deriv_data[arc_id] = 1;
              state = src_state;
            }
          }
        }
      }
      ASSERT_EQ(arc_derivs.Dim(), num_arcs);
      CheckArrayData(arc_derivs, expected_arc_deriv);
    }
    {
      // logsum
      Array1<FloatType> forward_scores = GetForwardScores<FloatType>(
          fsa_vec, state_batches, entering_arc_batches, true);
      Array1<FloatType> backward_scores = GetBackwardScores<FloatType>(
          fsa_vec, state_batches, leaving_arc_batches, true);
      // set the forward_scores_deriv_in to all zeros except for a 1 at the
      // final-state. Then the returned derivative of
      // BackpropGetForwardScores() should be identical to the posterior as
      // obtained by doing GetArcPost() and exponentiating
      Array1<FloatType> forward_scores_deriv_in(context, num_states, 0);
      FloatType *forward_scores_deriv_in_data = forward_scores_deriv_in.Data();
      const int32_t *fsa_row_splits1 = fsa_vec.RowSplits(1).Data();
      K2_EVAL(
          context, num_fsas, lambda_set_forward_derivs_in,
          (int32_t fsa_idx)->void {
            int32_t start_state = fsa_row_splits1[fsa_idx],
                    start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
            if (start_state_next_fsa - start_state > 0)
              forward_scores_deriv_in_data[start_state_next_fsa - 1] = 1;
          });
      Array1<FloatType> arc_derivs = BackpropGetForwardScores(
          fsa_vec, state_batches, leaving_arc_batches, true, nullptr,
          forward_scores, forward_scores_deriv_in);
      ASSERT_EQ(arc_derivs.Dim(), num_arcs);
      Array1<FloatType> arc_post =
          GetArcPost(fsa_vec, forward_scores, backward_scores);
      FloatType *arc_post_data = arc_post.Data();
      K2_EVAL(
          context, num_arcs, lambda_exp_arc_post,
          (int32_t i)->void { arc_post_data[i] = exp(arc_post_data[i]); });
      CheckArrayData(arc_derivs, arc_post);
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestBackpropForwardScores) {
  // simple case
  TestBackpropGetForwardScores<float>(fsa_vec_);
  TestBackpropGetForwardScores<double>(fsa_vec_);
  // random case
  for (int32_t i = 0; i != 2; ++i) {
    FsaVec random_fsas = RandomFsaVec();
    // make the fsa connected for easy testing for tropical version, the
    // algorithm should work for non-connected fsa as well.
    FsaVec connected;
    Connect(random_fsas, &connected);
    TestBackpropGetForwardScores<float>(connected);
    TestBackpropGetForwardScores<double>(connected);
  }
}

template <typename FloatType>
void TestBackpropGetBackwardScores(FsaVec &fsa_vec_in) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
                                     // random case
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsa_vec = fsa_vec_in.To(context);
    int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
            num_arcs = fsa_vec.NumElements();

    Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
    Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
    Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
    Ragged<int32_t> entering_arc_batches =
        GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);
    Ragged<int32_t> leaving_arc_batches =
        GetLeavingArcIndexBatches(fsa_vec, state_batches);

    {
      // max
      Array1<FloatType> forward_scores = GetForwardScores<FloatType>(
          fsa_vec, state_batches, entering_arc_batches, false);
      Array1<FloatType> backward_scores = GetBackwardScores<FloatType>(
          fsa_vec, state_batches, leaving_arc_batches, false);
      // set the backward_scores_deriv_in to all zeros except for a 1 at the
      // start-state. Then the returned derivative will be non-zero values
      // only for those arcs along the best path.
      Array1<FloatType> backward_scores_deriv_in(context, num_states, 0);
      FloatType *backward_scores_deriv_in_data =
          backward_scores_deriv_in.Data();
      const int32_t *fsa_row_splits1 = fsa_vec.RowSplits(1).Data();
      K2_EVAL(
          context, num_fsas, lambda_set_backward_derivs_in,
          (int32_t fsa_idx)->void {
            int32_t start_state = fsa_row_splits1[fsa_idx],
                    start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
            if (start_state_next_fsa - start_state > 0)
              backward_scores_deriv_in_data[start_state] = 1;
          });
      Array1<FloatType> arc_derivs = BackpropGetBackwardScores(
          fsa_vec, state_batches, entering_arc_batches, false, backward_scores,
          backward_scores_deriv_in);
      ASSERT_EQ(arc_derivs.Dim(), num_arcs);
      Array1<int32_t> fsa_row_splits1_cpu = fsa_vec.RowSplits(1).To(cpu);
      Array1<int32_t> fsa_row_splits2_cpu = fsa_vec.RowSplits(2).To(cpu);
      Array1<Arc> cpu_arcs = fsa_vec.values.To(cpu);
      backward_scores = backward_scores.To(cpu);
      Array1<FloatType> expected_arc_deriv(cpu, num_arcs, 0);
      FloatType *expected_arc_deriv_data = expected_arc_deriv.Data();
      const FloatType negative_infinity =
          -std::numeric_limits<FloatType>::infinity();
      for (int32_t fsa_idx = 0; fsa_idx != num_fsas; ++fsa_idx) {
        int32_t start_state = fsa_row_splits1_cpu[fsa_idx],
                start_state_next_fsa = fsa_row_splits1_cpu[fsa_idx + 1];
        if (start_state_next_fsa > start_state) {
          for (int32_t state = start_state;
               state != start_state_next_fsa - 1;) {
            FloatType score = negative_infinity;
            int32_t arc_id = -1;
            int32_t dest_state = -1;
            // get best leaving arc id for the state, i.e. the arc contributes
            // to the backward scores of the this state when computing backward
            // scores for tropical semiring.
            int32_t arc_begin = fsa_row_splits2_cpu[state],
                    arc_end = fsa_row_splits2_cpu[state + 1];
            ASSERT_GT(arc_end, arc_begin);  // as we suppose the input fsa
                                            // connected for the test suite
            for (int32_t cur_arc_id = arc_begin; cur_arc_id != arc_end;
                 ++cur_arc_id) {
              const Arc &arc = cpu_arcs[cur_arc_id];
              ASSERT_EQ(state, start_state + arc.src_state);
              int32_t cur_dest_state = start_state + arc.dest_state;
              FloatType cur_score = arc.score + backward_scores[cur_dest_state];
              // note below we use `>=` instead of `>` as we prefer larger
              // arc_id which is consistent with BackpropGetBackwardScores.
              if (cur_score >= score) {
                score = cur_score;
                arc_id = cur_arc_id;
                dest_state = cur_dest_state;
              }
            }
            expected_arc_deriv_data[arc_id] = 1;
            state = dest_state;
          }
        }
      }
      ASSERT_EQ(arc_derivs.Dim(), num_arcs);
      CheckArrayData(arc_derivs, expected_arc_deriv);
    }
    {
      // logsum
      Array1<FloatType> forward_scores = GetForwardScores<FloatType>(
          fsa_vec, state_batches, entering_arc_batches, true);
      Array1<FloatType> backward_scores = GetBackwardScores<FloatType>(
          fsa_vec, state_batches, leaving_arc_batches, true);
      Array1<FloatType> arc_post =
          GetArcPost(fsa_vec, forward_scores, backward_scores);
      // set the backward_scores_deriv_in to all zeros except for a 1 at the
      // start-state. Then the returned derivative of
      // BackpropGetBackwardScores() should be identical to the posterior as
      // obtained by doing GetArcPost() and exponentiating
      Array1<FloatType> backward_scores_deriv_in(context, num_states, 0);
      FloatType *backward_scores_deriv_in_data =
          backward_scores_deriv_in.Data();
      const int32_t *fsa_row_splits1 = fsa_vec.RowSplits(1).Data();
      K2_EVAL(
          context, num_fsas, lambda_set_backward_derivs_in,
          (int32_t fsa_idx)->void {
            int32_t start_state = fsa_row_splits1[fsa_idx],
                    start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
            if (start_state_next_fsa - start_state > 0)
              backward_scores_deriv_in_data[start_state] = 1;
          });
      Array1<FloatType> arc_derivs = BackpropGetBackwardScores(
          fsa_vec, state_batches, entering_arc_batches, true, backward_scores,
          backward_scores_deriv_in);
      ASSERT_EQ(arc_derivs.Dim(), num_arcs);
      FloatType *arc_post_data = arc_post.Data();
      K2_EVAL(
          context, num_arcs, lambda_exp_arc_post,
          (int32_t i)->void { arc_post_data[i] = exp(arc_post_data[i]); });
      CheckArrayData(arc_derivs, arc_post);
    }
  }
}
TEST_F(StatesBatchSuiteTest, TestBackpropBackwardScores) {
  // simple case
  TestBackpropGetBackwardScores<float>(fsa_vec_);
  TestBackpropGetBackwardScores<double>(fsa_vec_);
  for (int32_t i = 0; i != 2; ++i) {
    // random case
    FsaVec random_fsas = RandomFsaVec();
    // make the fsa connected for easy testing for tropical version, the
    // algorithm should work for non-connected fsa as well.
    FsaVec connected;
    Connect(random_fsas, &connected);
    TestBackpropGetBackwardScores<float>(connected);
    TestBackpropGetBackwardScores<double>(connected);
  }
}

template <typename FloatType>
void TestRandomPaths(FsaVec &fsa_vec_in) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data

  Ragged<int32_t> cpu_paths[2];  // indexed by i below.


  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsas = fsa_vec_in.To(context);
    int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
            num_arcs = fsas.NumElements();

    Ragged<int32_t> state_batches = GetStateBatches(fsas, true);
    Array1<int32_t> dest_states = GetDestStates(fsas, true);
    Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsas, dest_states);
    Ragged<int32_t> entering_arc_batches =
        GetEnteringArcIndexBatches(fsas, incoming_arcs, state_batches);
    Ragged<int32_t> leaving_arc_batches =
        GetLeavingArcIndexBatches(fsas, state_batches);

    for (int32_t i = 0; i < 2; ++i) {
      bool log_semiring = (i != 0);

      Array1<FloatType> forward_scores = GetForwardScores<FloatType>(
          fsas, state_batches, entering_arc_batches, log_semiring, nullptr);
      Array1<FloatType> backward_scores = GetBackwardScores<FloatType>(
          fsas, state_batches, leaving_arc_batches, log_semiring);

      Array1<FloatType> arc_post =
          GetArcPost(fsas, forward_scores, backward_scores);

      Array1<FloatType> arc_cdf = GetArcCdf(fsas, arc_post);
      const FloatType *arc_cdf_data = arc_cdf.Data(),
                      *arc_post_data = arc_post.Data();
      const int32_t *fsas_row_splits2_data = fsas.RowSplits(2).Data(),
                    *fsas_row_ids2_data = fsas.RowIds(2).Data();

      K2_EVAL(
          context, fsas.NumElements(), lambda_check_cdf,
          (int32_t arc_idx012) {
            int32_t state_idx01 = fsas_row_ids2_data[arc_idx012];
            FloatType cdf_val = arc_cdf_data[arc_idx012];
            K2_CHECK_GE(cdf_val, 0.0);
            K2_CHECK_LE(cdf_val, 1.0);
            if (arc_idx012 > fsas_row_splits2_data[state_idx01]) {
              K2_CHECK_GE(cdf_val, arc_cdf_data[arc_idx012 - 1]);
            }
          });

      Array1<FloatType> tot_scores = GetTotScores(fsas, forward_scores);

      int32_t num_paths = 10;
      Ragged<int32_t> paths =
          RandomPaths(fsas, arc_cdf, num_paths, tot_scores, state_batches);

      if (context->GetDeviceType() == kCpu) {
        cpu_paths[i] = paths;
      } else {
        Ragged<int32_t> other_paths = cpu_paths[i].To(context);
        if (!Equal(paths, other_paths)) {
          K2_LOG(WARNING) << "Paths differ: " << other_paths << " vs. "
                          << paths;
        }
      }


      int32_t *paths_row_ids2 = paths.RowIds(2).Data(),
          *paths_row_splits2 = paths.RowSplits(2).Data(),
          *paths_row_splits1 = paths.RowSplits(1).Data(),
          *paths_row_ids1 = paths.RowIds(1).Data(),
              *paths_data = paths.values.Data(),
              *fsas_row_splits1_data = fsas.RowSplits(1).Data(),
              *fsas_row_ids1_data = fsas.RowIds(1).Data();
      const Arc *arcs_data = fsas.values.Data();

      K2_EVAL(
          context, paths.NumElements(), lambda_check_arcs,
          (int32_t path_idx012) {
            int32_t arc_idx012 = paths_data[path_idx012],
                    state_idx01 = fsas_row_ids2_data[arc_idx012],
                    fsa_idx0 = fsas_row_ids1_data[state_idx01],
                    state_idx0x = fsas_row_splits1_data[fsa_idx0];
            int32_t path_idx01 = paths_row_ids2[path_idx012],
                    path_idx01x = paths_row_splits2[path_idx01],
                    path_idx2 = path_idx012 - path_idx01x;
            if (path_idx2 > 0) {
              int32_t prev_arc_idx012 = paths_data[path_idx012 - 1],
                      prev_dest_state_idx1 =
                          arcs_data[prev_arc_idx012].dest_state,
                      prev_dest_state_idx01 =
                          state_idx0x + prev_dest_state_idx1;
              K2_CHECK_EQ(state_idx01, prev_dest_state_idx01);
            } else {
              K2_CHECK_EQ(state_idx01, state_idx0x);
            }
            if (path_idx012 + 1 == paths_row_splits2[path_idx01 + 1]) {
              int32_t dest_state_idx01 =
                  state_idx0x + arcs_data[arc_idx012].dest_state;
              K2_CHECK_EQ(dest_state_idx01,
                          fsas_row_splits1_data[fsa_idx0 + 1] - 1);
            }
          });

      // the paths (sequences of arcs) should be in lexicographical order.
      K2_EVAL(
          context, paths.TotSize(1), lambda_check_order,
          (int32_t path_idx01) {
            int32_t fsa_idx0 = paths_row_ids1[path_idx01],
                path_idx1 = path_idx01 - paths_row_splits1[fsa_idx0];
            if (path_idx1 > 0) {
              int32_t path_idx01x_prev = paths_row_splits2[path_idx01 - 1],
                  path_idx01x = paths_row_splits2[path_idx01],
                  path_idx01x_next = paths_row_splits2[path_idx01 + 1];
              int32_t len_prev = path_idx01x - path_idx01x_prev,
                  len = path_idx01x_next - path_idx01x;
              int32_t min_len = min(len, len_prev);
              for (int32_t i = 0; i < min_len; i++) {
                int32_t prev_arc = paths_data[path_idx01x_prev + i],
                    arc = paths_data[path_idx01x + i];
                if (arc > prev_arc) {
                  break;
                } else {
                  // arc should not be < prev_arc.
                  K2_CHECK_GE(arc, prev_arc)
                      << ", arc_cdf_data[prev_arc,arc] = "
                      << arc_cdf_data[prev_arc] << "," << arc_cdf_data[arc];
                }
              }
            }
          });
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestRandomPaths) {
  // simple case
  TestRandomPaths<float>(fsa_vec_);
  TestRandomPaths<double>(fsa_vec_);
  for (int32_t i = 0; i != 50; ++i) {
    // random case
    FsaVec random_fsas = RandomFsaVec();
    // make the fsa connected for easy testing for tropical version, the
    // algorithm should work for non-connected fsa as well.
    FsaVec connected;
    Connect(random_fsas, &connected);
    TestRandomPaths<float>(connected);
    TestRandomPaths<double>(connected);
  }
}

template <typename FloatType>
void TestBackpropGetArcPost(FsaVec &fsa_vec_in) {
  ContextPtr cpu = GetCpuContext();  // will be used to copy data
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsa_vec = fsa_vec_in.To(context);
    int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1),
            num_arcs = fsa_vec.NumElements();

    Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
    Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
    Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);

    Array1<FloatType> arc_post_deriv(context, num_arcs, 1);
    Array1<FloatType> forward_scores_deriv, backward_scores_deriv;
    BackpropGetArcPost(fsa_vec, incoming_arcs, arc_post_deriv,
                       &forward_scores_deriv, &backward_scores_deriv);
    fsa_vec = fsa_vec.To(cpu);
    Array1<int32_t> fsa_row_splits1 = fsa_vec.RowSplits(1),
                    fsa_row_splits2 = fsa_vec.RowSplits(2);
    incoming_arcs = incoming_arcs.To(cpu);
    Array1<int32_t> incoming_arcs_row_splits1 = incoming_arcs.RowSplits(1),
                    incoming_arcs_row_splits2 = incoming_arcs.RowSplits(2);
    forward_scores_deriv = forward_scores_deriv.To(cpu);
    backward_scores_deriv = backward_scores_deriv.To(cpu);
    for (int32_t fsa_idx = 0; fsa_idx != num_fsas; ++fsa_idx) {
      int32_t start_state = fsa_row_splits1[fsa_idx],
              start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
      if (start_state_next_fsa > start_state) {
        int32_t num_arcs_this_fsa = fsa_row_splits2[start_state_next_fsa] -
                                    fsa_row_splits2[start_state];
        FloatType sum_deriv = -0.5 * num_arcs_this_fsa;
        EXPECT_EQ(backward_scores_deriv[start_state], sum_deriv);
        EXPECT_EQ(forward_scores_deriv[start_state_next_fsa - 1], sum_deriv);
        for (int32_t state = start_state; state != start_state_next_fsa;
             ++state) {
          if (state != start_state)
            EXPECT_EQ(backward_scores_deriv[state],
                      incoming_arcs_row_splits2[state + 1] -
                          incoming_arcs_row_splits2[state]);
          if (state != start_state_next_fsa - 1)
            EXPECT_EQ(forward_scores_deriv[state],
                      fsa_row_splits2[state + 1] - fsa_row_splits2[state]);
        }
      }
    }
  }
}
TEST_F(StatesBatchSuiteTest, TestBackpropArcPost) {
  // simple case
  TestBackpropGetArcPost<float>(fsa_vec_);
  TestBackpropGetArcPost<double>(fsa_vec_);
  for (int32_t i = 0; i != 2; ++i) {
    // random case
    FsaVec random_fsas = RandomFsaVec();
    TestBackpropGetArcPost<float>(random_fsas);
    TestBackpropGetArcPost<double>(random_fsas);
  }
}

TEST(FsaUtils, ConvertDenseToFsaVec) {
  /*
    -inf  0    1
      0 -inf -inf
    -inf  2    3
    -inf  4    5
      0 -inf  -inf
    -inf  6    7
    -inf  8    9
    -inf  10   11
      0 -inf  -inf
  */
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  // clang-format off
  std::vector<float> data = {
    kNegInf, 0, 1,
    0, kNegInf, kNegInf,
    kNegInf, 2, 3,
    kNegInf, 4, 5,
    0, kNegInf, kNegInf,
    kNegInf, 6, 7,
    kNegInf, 8, 9,
    kNegInf, 10, 11,
    0, kNegInf, kNegInf,
  };
  // clang-format on

  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> row_splits(context, std::vector<int32_t>{0, 2, 5, 9});
    RaggedShape shape = RaggedShape2(&row_splits, nullptr, 9);
    Array1<float> tmp(context, data);
    Array2<float> score(tmp, 9, 3);

    DenseFsaVec dense_fsa_vec{shape, score};
    FsaVec fsa_vec = ConvertDenseToFsaVec(dense_fsa_vec);
    ASSERT_EQ(fsa_vec.Dim0(), 3);  // there are 3 FSAs

    fsa_vec = fsa_vec.To(GetCpuContext());  // for testing

    CheckArrayData(fsa_vec.RowSplits(1), std::vector<int32_t>{0, 3, 7, 12});
    CheckArrayData(fsa_vec.RowSplits(2),
                   std::vector<int32_t>{
                       0, 2,        // fsa 0, state 0
                       3, 3,        // fsa 0, state 1, final state
                       5, 7,        // fsa 1, state 0, state 1
                       8, 8,        // fsa 1, state 2, final state
                       10, 12, 14,  // fsa 2, state 0, 1, 2
                       15, 15       // fsa 2, state 3, final state
                   });
    //             [{fsa, state, arc}]
    EXPECT_EQ((fsa_vec[{0, 0, 0}]), (Arc{0, 1, 0, 0}));
    EXPECT_EQ((fsa_vec[{0, 0, 1}]), (Arc{0, 1, 1, 1}));
    EXPECT_EQ((fsa_vec[{0, 1, 0}]), (Arc{1, 2, -1, 0}));

    EXPECT_EQ((fsa_vec[{1, 0, 0}]), (Arc{0, 1, 0, 2}));
    EXPECT_EQ((fsa_vec[{1, 0, 1}]), (Arc{0, 1, 1, 3}));
    EXPECT_EQ((fsa_vec[{1, 1, 0}]), (Arc{1, 2, 0, 4}));
    EXPECT_EQ((fsa_vec[{1, 1, 1}]), (Arc{1, 2, 1, 5}));
    EXPECT_EQ((fsa_vec[{1, 2, 0}]), (Arc{2, 3, -1, 0}));

    EXPECT_EQ((fsa_vec[{2, 0, 0}]), (Arc{0, 1, 0, 6}));
    EXPECT_EQ((fsa_vec[{2, 0, 1}]), (Arc{0, 1, 1, 7}));
    EXPECT_EQ((fsa_vec[{2, 1, 0}]), (Arc{1, 2, 0, 8}));
    EXPECT_EQ((fsa_vec[{2, 1, 1}]), (Arc{1, 2, 1, 9}));
    EXPECT_EQ((fsa_vec[{2, 2, 0}]), (Arc{2, 3, 0, 10}));
    EXPECT_EQ((fsa_vec[{2, 2, 1}]), (Arc{2, 3, 1, 11}));
    EXPECT_EQ((fsa_vec[{2, 3, 0}]), (Arc{3, 4, -1, 0}));
  }
}

TEST(FsaUtils, ComposeArcMapsTest) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    {
      // simple case
      const std::vector<int32_t> arc_map1_row_splits = {0, 2, 2, 3, 6};
      Array1<int32_t> arc_map1_row_splits_array(context, arc_map1_row_splits);
      RaggedShape arc_map1_shape =
          RaggedShape2(&arc_map1_row_splits_array, nullptr, -1);
      const std::vector<int32_t> arc_map1_values = {1, 5, 4, 8, -1, 0};
      Array1<int32_t> arc_map1_values_array(context, arc_map1_values);
      Ragged<int32_t> arc_map1(arc_map1_shape, arc_map1_values_array);

      const std::vector<int32_t> arc_map2_row_splits = {0, 1, 3, 3, 8};
      Array1<int32_t> arc_map2_row_splits_array(context, arc_map2_row_splits);
      RaggedShape arc_map2_shape =
          RaggedShape2(&arc_map2_row_splits_array, nullptr, -1);
      const std::vector<int32_t> arc_map2_values = {2, 0, 1, 2, 3, 0, 1, 3};
      Array1<int32_t> arc_map2_values_array(context, arc_map2_values);
      Ragged<int32_t> arc_map2(arc_map2_shape, arc_map2_values_array);

      Ragged<int> ans = ComposeArcMaps(arc_map1, arc_map2);
      EXPECT_EQ(ans.NumAxes(), 2);
      EXPECT_EQ(ans.Dim0(), arc_map2.Dim0());
      const std::vector<int32_t> expected_row_splits = {0, 1, 3, 3, 12};
      const std::vector<int32_t> expected_values = {4, 1, 5, 4, 8,  -1,
                                                    0, 1, 5, 8, -1, 0};
      CheckArrayData(ans.RowSplits(1), expected_row_splits);
      CheckArrayData(ans.values, expected_values);
    }

    {
      // test with random size
      ContextPtr cpu = GetCpuContext();
      for (auto &context : {GetCpuContext(), GetCudaContext()}) {
        for (int32_t i = 0; i < 2; ++i) {
          Ragged<int32_t> arc_map1 =
              RandomRagged<int32_t>(-1, 100, 2, 2, 0, 1000).To(context);
          RaggedShape arc_map2_shape =
              RandomRaggedShape(false, 2, 2, 0, 1000).To(context);
          int32_t arc_map1_dim0 = arc_map1.Dim0(),
                  arc_map2_value_dim = arc_map2_shape.NumElements();
          if (arc_map1_dim0 == 0) continue;
          Array1<int32_t> arc_map2_values = RandUniformArray1(
              context, arc_map2_value_dim, 0, arc_map1_dim0 - 1);
          Ragged<int32_t> arc_map2(arc_map2_shape, arc_map2_values);

          Ragged<int32_t> ans = ComposeArcMaps(arc_map1, arc_map2);
          EXPECT_EQ(ans.NumAxes(), 2);
          EXPECT_EQ(ans.Dim0(), arc_map2.Dim0());
          ans = ans.To(cpu);
          arc_map1 = arc_map1.To(cpu);
          arc_map2 = arc_map2.To(cpu);
          const int32_t *arc_map1_row_splits = arc_map1.RowSplits(1).Data(),
                        *arc_map2_row_splits = arc_map2.RowSplits(1).Data(),
                        *ans_row_splits = ans.RowSplits(1).Data();
          const int32_t *arc_map1_value = arc_map1.values.Data(),
                        *arc_map2_value = arc_map2.values.Data(),
                        *ans_value = ans.values.Data();
          int32_t ans_tot_size = 0;
          int32_t ans_idx01 = 0;
          for (int32_t i = 0; i != arc_map2.Dim0(); ++i) {
            int32_t arc_map2_row_begin = arc_map2_row_splits[i],
                    arc_map2_row_end = arc_map2_row_splits[i + 1];
            for (int32_t j = arc_map2_row_begin; j != arc_map2_row_end; ++j) {
              int32_t arc_map1_index = arc_map2_value[j];
              ASSERT_GE(arc_map1_index, 0);
              ASSERT_LT(arc_map1_index, arc_map1_dim0);
              int32_t arc_map1_row_begin = arc_map1_row_splits[arc_map1_index],
                      arc_map1_row_end =
                          arc_map1_row_splits[arc_map1_index + 1];
              ans_tot_size += arc_map1_row_end - arc_map1_row_begin;
              for (int32_t n = arc_map1_row_begin; n != arc_map1_row_end; ++n) {
                int32_t cur_value = arc_map1_value[n];
                int32_t cur_ans_value = ans_value[ans_idx01++];
                EXPECT_EQ(cur_value, cur_ans_value);
              }
            }
            // check row_splits of ans
            EXPECT_EQ(ans_tot_size, ans_row_splits[i + 1]);
          }
        }
      }
    }
  }
}

TEST(FixNumStates, FixNumStates) {
  FsaVec f("[ [ [] []  ] [ [] [] ] ]"), g("[ [ []  ] [ [] [] ] ]"),
      h("[ [ ] [ [] [] ] ]");

  FsaVec f2(f), g2(g), h2(h);

  FixNumStates(&f2);
  FixNumStates(&g2);
  FixNumStates(&h2);

  EXPECT_EQ(Equal(f, f2), true);
  EXPECT_EQ(Equal(h, g2), true);
  EXPECT_EQ(Equal(h, h2), true);
}

TEST(FixFinalLabels, FixFinalLabels) {
  // src_state dst_state label cost
  std::string s = R"(0 1 10 -1.2
    0 2  6 -2.2
    0 3  9 -2.2
    1 2  8  -3.2
    1 3  6  -4.2
    2 3  5 -5.2
    2 4  4  -6.2
    3 5 -1  -7.2
    5
    )";
  for (auto &context : {GetCudaContext(), GetCpuContext()}) {
    Fsa fsa = FsaFromString(s);
    fsa = fsa.To(context);
    std::vector<Fsa*> fsa_ptrs { &fsa, &fsa };
    K2_CHECK_EQ(fsa_ptrs.size(), 2);
    FsaVec fsas = Stack(0, 2, fsa_ptrs.data(), nullptr);


    Array1<int32_t> labels(context, "[ 0 -1 1 2 3 4 5 0 ]"),
        correct_labels(context, "[ 0 0 1 2 3 4 5 -1 ]");

    std::vector<const Array1<int32_t>*> labels_repeated_ptrs = { &labels
                                                                 , &labels },
        correct_labels_repeated_ptrs = { &correct_labels, &correct_labels };
    Array1<int32_t> labels_repeated = Cat(context, 2,
                                          labels_repeated_ptrs.data()),
        correct_labels_repeated = Cat(context, 2,
                                      correct_labels_repeated_ptrs.data());

    FixFinalLabels(fsa, labels.Data(), 1);
    EXPECT_EQ(true, Equal(labels, correct_labels));

    FixFinalLabels(fsas, labels_repeated.Data(), 1);
    EXPECT_EQ(true, Equal(labels_repeated, correct_labels_repeated));

    // this test is rather weak as it should not change `fsas`.
    FixFinalLabels(fsas, reinterpret_cast<int32_t*>(fsas.values.Data()) + 2, 4);
    int32_t props;
    GetFsaVecBasicProperties(fsas, nullptr, &props);
    K2_CHECK_NE(0, props & kFsaPropertiesValid);
  }
}



}  // namespace k2
