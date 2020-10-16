/**
 * @brief Unittest for fsa utils.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "k2/csrc/fsa.h"
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

template <DeviceType d>
void TestGetDestStates() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

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
    std::vector<int32_t> cpu_data(result.Data(), result.Data() + result.Dim());
    EXPECT_THAT(cpu_data, ::testing::ElementsAre(1, 2, 3, 3, 2, 3, 4, 5, 5, 1,
                                                 2, 2, 3, 3, 4));
  }

  {
    // as_idx01 = true
    Array1<int32_t> result = GetDestStates(fsa_vec, true);
    ASSERT_EQ(result.Dim(), fsa_vec.NumElements());
    result = result.To(cpu);
    std::vector<int32_t> cpu_data(result.Data(), result.Data() + result.Dim());
    EXPECT_THAT(cpu_data, ::testing::ElementsAre(1, 2, 3, 3, 2, 3, 4, 5, 5, 7,
                                                 8, 8, 9, 9, 10));
  }
}

TEST(FsaUtilsTest, TestGetDestStates) {
  TestGetDestStates<kCpu>();
  TestGetDestStates<kCuda>();
}

template <DeviceType d>
void TestGetStateBatches() {
  ContextPtr cpu = GetCpuContext();  // will use to copy data
  ContextPtr context = nullptr;
  if (d == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(d, kCuda);
    context = GetCudaContext();
  }

  {
    // simple case
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

    std::string s3 = R"(0 2 1 0
    1 2  1 0
    1 3  1 0
    1 4  1 0
    2 3  1 0
    2 4  1 0
    3 4  1 0
    4 5  -1 0
    5
  )";

    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa fsa3 = FsaFromString(s3);
    std::vector<int32_t> states_num = {fsa1.Dim0(), fsa2.Dim0(), fsa3.Dim0()};
    Fsa *fsa_array[] = {&fsa1, &fsa2, &fsa3};
    FsaVec fsa_vec = CreateFsaVec(3, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);
    int32_t num_fsas = fsa_vec.Dim0(), num_states = fsa_vec.TotSize(1);
    EXPECT_EQ(num_fsas, 3);

    {
      // no transpose: [fsa_idx][batch_idx][state]
      Ragged<int32_t> result = GetStateBatches(fsa_vec, false);
      result = result.To(cpu);
      EXPECT_EQ(result.Dim0(), num_fsas);
      ASSERT_EQ(result.NumElements(), num_states);
      int32_t *row_splits1_data = result.RowSplits(1).Data();
      for (int32_t n = 0; n < num_fsas; ++n) {
        int32_t num_batches = row_splits1_data[n + 1] - row_splits1_data[n];
        // num-batches in each fsa should not be greater num-states
        EXPECT_LE(num_batches, states_num[n]);
        if (states_num[n] > 0) {
          EXPECT_GT(num_batches, 0);
        }
      }
      // check values
      std::vector<int32_t> states(num_states);
      std::iota(states.begin(), states.end(), 0);
      Array1<int32_t> values = result.values;
      ASSERT_EQ(values.Dim(), num_states);
      std::vector<int32_t> cpu_values(values.Data(),
                                      values.Data() + values.Dim());
      EXPECT_EQ(cpu_values, states);
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
    }
  }

  // TODO(haowen): add random cases
}

TEST(FsaUtilsTest, TestGetStateBatches) {
  TestGetStateBatches<kCpu>();
  TestGetStateBatches<kCuda>();
}

}  // namespace k2
