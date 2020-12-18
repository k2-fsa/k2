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

#include <limits>
#include <numeric>
#include <vector>

#include "k2/csrc/fsa.h"
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
    EXPECT_EQ(aux_labels[8], -1);
    EXPECT_EQ(aux_labels[9], -1);
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
            fsa_vec, state_batches, leaving_arc_batches, nullptr, false);
        EXPECT_EQ(scores.Dim(), num_states);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores =
            GetBackwardScores<float>(cpu_fsa_vec, nullptr, false);
        CheckArrayData(scores, cpu_scores);
        // [ 20 19 -inf 14 8 0 10 9 6 -inf 0 21 22 20 15 8 0 ]
      }
      {
        // max with tot_scores provided
        Array1<float> forward_scores = GetForwardScores<float>(
            fsa_vec, state_batches, entering_arc_batches, false);
        Array1<float> tot_scores = GetTotScores(fsa_vec, forward_scores);
        Array1<float> scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, &tot_scores, false);
        EXPECT_EQ(scores.Dim(), num_states);
        Array1<float> cpu_tot_scores = tot_scores.To(cpu);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores =
            GetBackwardScores<float>(cpu_fsa_vec, &cpu_tot_scores, false);
        CheckArrayData(scores, cpu_scores);
        // [ 0 -1 -inf -6 -12 -20 0 -1 -4 -inf -10 0 1 -1 -6 -13 -21 ]
      }
      {
        // logsum
        Array1<float> scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, nullptr, true);
        EXPECT_EQ(scores.Dim(), num_states);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores =
            GetBackwardScores<float>(cpu_fsa_vec, nullptr, true);
        CheckArrayData(scores, cpu_scores);
        // [ 20.0668 19.0009 -inf 14.0009 8 0 10.1269 9 6 -inf
        // 0 21.0025 22.0206 20.0025 15 8 0 ]
      }
      {
        // logsum with tot_scores provided
        Array1<float> forward_scores = GetForwardScores<float>(
            fsa_vec, state_batches, entering_arc_batches, true);
        Array1<float> tot_scores = GetTotScores(fsa_vec, forward_scores);
        Array1<float> scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, &tot_scores, true);
        EXPECT_EQ(scores.Dim(), num_states);
        Array1<float> cpu_tot_scores = tot_scores.To(cpu);
        FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
        Array1<float> cpu_scores =
            GetBackwardScores<float>(cpu_fsa_vec, &cpu_tot_scores, true);
        CheckArrayData(scores, cpu_scores);
        // [ -0.00200483 -1.06789 -inf -6.06789 -12.0688 -20.0688 2.82824e-05
        // -1.1269 -4.1269 -inf -10.1269 -2.47955e-05 1.01813 -1.00002 -6.0025
        // -13.0025 -21.0025 ]
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
              fsa_vec, state_batches, leaving_arc_batches, nullptr, false);
          EXPECT_EQ(scores.Dim(), num_states);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetBackwardScores<float>(cpu_fsa_vec, nullptr, false);
          CheckArrayData(scores, cpu_scores);
        }
        {
          // max with tot_scores provided
          Array1<float> forward_scores = GetForwardScores<float>(
              fsa_vec, state_batches, entering_arc_batches, false);
          Array1<float> tot_scores = GetTotScores(fsa_vec, forward_scores);
          Array1<float> scores = GetBackwardScores<float>(
              fsa_vec, state_batches, leaving_arc_batches, &tot_scores, false);
          EXPECT_EQ(scores.Dim(), num_states);
          Array1<float> cpu_tot_scores = tot_scores.To(cpu);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetBackwardScores<float>(cpu_fsa_vec, &cpu_tot_scores, false);
          CheckArrayData(scores, cpu_scores);
        }
        {
          // logsum
          Array1<float> scores = GetBackwardScores<float>(
              fsa_vec, state_batches, leaving_arc_batches, nullptr, true);
          EXPECT_EQ(scores.Dim(), num_states);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetBackwardScores<float>(cpu_fsa_vec, nullptr, true);
          CheckArrayData(scores, cpu_scores);
        }
        {
          // logsum with tot_scores provided
          Array1<float> forward_scores = GetForwardScores<float>(
              fsa_vec, state_batches, entering_arc_batches, true);
          Array1<float> tot_scores = GetTotScores(fsa_vec, forward_scores);
          Array1<float> scores = GetBackwardScores<float>(
              fsa_vec, state_batches, leaving_arc_batches, &tot_scores, true);
          EXPECT_EQ(scores.Dim(), num_states);
          Array1<float> cpu_tot_scores = tot_scores.To(cpu);
          FsaVec cpu_fsa_vec = fsa_vec.To(cpu);
          Array1<float> cpu_scores =
              GetBackwardScores<float>(cpu_fsa_vec, &cpu_tot_scores, true);
          CheckArrayData(scores, cpu_scores);
        }
      }
    }
  }
}

TEST_F(StatesBatchSuiteTest, TestArcScores) {
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
            fsa_vec, state_batches, leaving_arc_batches, nullptr, false);
        Array1<float> arc_scores =
            GetArcScores(fsa_vec, forward_scores, backward_scores);
        EXPECT_EQ(arc_scores.Dim(), num_arcs);
        K2_LOG(INFO) << arc_scores;
        // [ 20 -inf 16 17 -inf 20 20 13 20 10 8 10 -inf -inf 10 21 -inf -inf
        // -inf 21 15 21 21 ]
      }
      {
        // logsum with tot_scores provided
        Array1<float> forward_scores = GetForwardScores<float>(
            fsa_vec, state_batches, entering_arc_batches, true);
        Array1<float> tot_scores = GetTotScores(fsa_vec, forward_scores);
        Array1<float> backward_scores = GetBackwardScores<float>(
            fsa_vec, state_batches, leaving_arc_batches, &tot_scores, true);
        Array1<float> arc_scores =
            GetArcScores(fsa_vec, forward_scores, backward_scores);
        EXPECT_EQ(arc_scores.Dim(), num_arcs);
        K2_LOG(INFO) << arc_scores;
        // [ -0.0658841 -inf -4.06588 -3.06588 -inf -0.0658841 -0.000911713
        // -7.00091 -0.000911713 -0.126928 -2.12693 -0.126928 -inf -inf 0 0 -inf
        // -inf -inf -0.00247574 -6.00248 -0.00247574 0 ]
      }
    }
  }
  // TODO(haowen): add random cases
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
  FsaVec f("[ [ [] []  ] [ [] [] ] ]"),
      g("[ [ []  ] [ [] [] ] ]"),
      h("[ [ ] [ [] [] ] ]");

  FsaVec f2(f), g2(g), h2(h);

  FixNumStates(&f2);
  FixNumStates(&g2);
  FixNumStates(&h2);

  EXPECT_EQ(Equal(f, f2), true);
  EXPECT_EQ(Equal(h, g2), true);
  EXPECT_EQ(Equal(h, h2), true);
}

}  // namespace k2
