/**
 * @brief Unittest for TopSort.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/math.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(TopSort, SingleFsa) {
  std::string s = R"(0 1 1 1
    0 2 2 2
    1 3 -1 13
    2 1 1 21
    2 3 -1 23
    3
  )";

  auto fsa = FsaFromString(s);

  int32_t gt = kFsaPropertiesTopSorted | kFsaPropertiesTopSortedAndAcyclic;
  int32_t p = GetFsaBasicProperties(fsa);
  EXPECT_NE(p & gt, gt);

  Fsa sorted;
  Array1<int32_t> arc_map;
  TopSort(fsa, &sorted, &arc_map);
  p = GetFsaBasicProperties(sorted);
  EXPECT_EQ(p & gt, gt);
  /* top sorted fsa is
  0 2 1 1    // src arc 0
  0 1 2 2    // src arc 1
  1 2 1 21   // src arc 3
  1 3 -1 23  // src arc 4
  2 3 -1 13  // src arc 2
  3
  */

  CheckArrayData(arc_map, {0, 1, 3, 4, 2});
}

TEST(TopSort, VectorOfFsas) {
  std::string s1 = R"(0 1 1 1
    0 2 2 2
    1 3 -1 13
    2 1 1 21
    2 3 -1 23
    3
  )";

  std::string s2 = R"(0 2 2 2
    1 3 -1 13
    2 1 1 21
    2 3 -1 23
    3
  )";

  auto fsa1 = FsaFromString(s1);
  auto fsa2 = FsaFromString(s2);
  Fsa *fsa_array[] = {&fsa1, &fsa2};
  FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);

  int32_t gt = kFsaPropertiesTopSorted | kFsaPropertiesTopSortedAndAcyclic;
  Array1<int32_t> properties;
  int32_t p;
  GetFsaVecBasicProperties(fsa_vec, &properties, &p);

  EXPECT_NE(p & gt, gt);
  EXPECT_NE(properties[0] & gt, gt);
  EXPECT_NE(properties[1] & gt, gt);

  FsaVec sorted;
  Array1<int32_t> arc_map;
  TopSort(fsa_vec, &sorted, &arc_map);
  GetFsaVecBasicProperties(sorted, &properties, &p);

  EXPECT_EQ(p & gt, gt);
  EXPECT_EQ(properties[0] & gt, gt);
  EXPECT_EQ(properties[1] & gt, gt);

  /* top sorted fsa1 is
  0 2 1 1    // src arc 0
  0 1 2 2    // src arc 1
  1 2 1 21   // src arc 3
  1 3 -1 23  // src arc 4
  2 3 -1 13  // src arc 2
  3

  top sorted fsa2 is
  0 1 2 2     // src arc 0
  1 2 1 21    // src arc 2
  1 3 -1 23   // src arc 3
  2 3 -1 13   // src arc 1
  */

  CheckArrayData(arc_map, {0, 1, 3, 4, 2, 5, 7, 8, 6});
}

TEST(TopSort, RandomSingleFsa) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa = GetRandFsa();
    fsa = fsa.To(context);

    int32_t gt = kFsaPropertiesTopSorted | kFsaPropertiesTopSortedAndAcyclic;
    int32_t p = GetFsaBasicProperties(fsa);
    EXPECT_NE(p & gt, gt);

    Fsa sorted;
    Array1<int32_t> arc_map;
    TopSort(fsa, &sorted, &arc_map);

    p = GetFsaBasicProperties(sorted);
    EXPECT_EQ(p & gt, gt);

    Array1<Arc> arcs = sorted.values.To(GetCpuContext());
    arc_map = arc_map.To(GetCpuContext());
    int32_t num_arcs = arcs.Dim();
    for (int32_t i = 0; i != num_arcs; ++i) {
      EXPECT_EQ(arcs[i].score, arc_map[i]);
    }
  }
}

TEST(TopSort, RandomVectorOfFsas) {
  int num_fsas = 1 + RandInt(0, 100);
  ContextPtr cpu = GetCpuContext();
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    std::vector<Fsa> fsas(num_fsas);
    for (int32_t i = 0; i != num_fsas; ++i) {
      fsas[i] = GetRandFsa();
    }

    int32_t offset = fsas[0].TotSize(1);
    for (int32_t i = 1; i != num_fsas; ++i) {
      Array1<Arc> &arcs = fsas[i].values;
      Arc *arcs_data = arcs.Data();
      int32_t num_arcs = arcs.Dim();
      EXPECT_GT(num_arcs, 1);
      for (int32_t k = 0; k != num_arcs; ++k) {
        arcs_data[k].score += offset;
      }
      offset += num_arcs;
    }

    std::vector<Fsa *> fsa_array(num_fsas);
    for (int32_t i = 0; i != num_fsas; ++i) {
      fsa_array[i] = &fsas[i];
    }

    FsaVec fsa_vec = CreateFsaVec(num_fsas, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    int32_t gt = kFsaPropertiesTopSorted | kFsaPropertiesTopSortedAndAcyclic;
    Array1<int32_t> properties;
    int32_t p;
    GetFsaVecBasicProperties(fsa_vec, &properties, &p);

    /*
    EXPECT_NE(p & gt, gt);
    properties = properties.To(cpu);
    for (int32_t i = 0; i != num_fsas; ++i) {
      EXPECT_NE(properties[i] & gt, gt);
      }*/

    FsaVec sorted;
    Array1<int32_t> arc_map;
    TopSort(fsa_vec, &sorted, &arc_map);

    GetFsaVecBasicProperties(sorted, &properties, &p);

    EXPECT_EQ(p & gt, gt);
    properties = properties.To(cpu);
    for (int32_t i = 0; i != num_fsas; ++i) {
      EXPECT_EQ(properties[i] & gt, gt);
    }

    Array1<Arc> arcs = sorted.values.To(cpu);
    arc_map = arc_map.To(cpu);

    int32_t num_arcs = sorted.TotSize(2);
    for (int32_t i = 0; i != num_arcs; ++i) {
      EXPECT_EQ(arcs[i].score, arc_map[i]);
    }
  }
}

// another random test which uses IsRandEquivalent to check the result
TEST(TopSort, RandomVectorOfFsas1) {
  ContextPtr cpu = GetCpuContext();
  for (int32_t n = 0; n != 1; ++n) {
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      // Here we generate acyclic Fsas for testing as current
      // implementation of GPU TopSort supposes the input Fsas are acyclic.
      FsaVec random_fsas = RandomFsaVec(1, 1000, true);
      int32_t num_fsas = random_fsas.Dim0();
      {
        // reorder states (except start and final state) in each Fsa to make
        // them non-top-sorted (but still acyclic)
        int32_t num_states = random_fsas.TotSize(1);
        Array1<int32_t> order = Range(cpu, num_states, 0);
        const int32_t *fsas_row_splits1_data = random_fsas.RowSplits(1).Data();
        std::random_device rd;
        std::mt19937 g(rd());
        for (int32_t i = 0; i != num_fsas; ++i) {
          int32_t start_state_idx01 = fsas_row_splits1_data[i];
          int32_t next_start_state_idx01 = fsas_row_splits1_data[i + 1];
          if (next_start_state_idx01 > start_state_idx01) {  // non-empty Fsa
            K2_CHECK_GE(next_start_state_idx01 - start_state_idx01, 2);
            // get states except start_state and final_state
            Array1<int32_t> states =
                order.Arange(start_state_idx01 + 1, next_start_state_idx01 - 1);
            if (states.Dim() == 0) continue;
            int32_t *states_data = states.Data();
            std::shuffle(states_data, states_data + states.Dim(), g);
          }
        }
        random_fsas = RenumberFsaVec(random_fsas, order, nullptr);
      }
      {
        // randomly change some arcs to self-loops
        std::random_device rd;
        std::mt19937 g(rd());
        std::bernoulli_distribution d(0.1);
        int32_t num_arcs = random_fsas.NumElements();
        Arc *arcs = random_fsas.values.Data();
        for (int32_t i = 0; i != num_arcs; ++i) {
          if (d(g)) arcs[i].dest_state = arcs[i].src_state;
        }
      }

      FsaVec fsa_vec = random_fsas.To(context);
      FsaVec cpu_fsa_vec = fsa_vec.To(cpu);

      int32_t gt = kFsaPropertiesTopSorted;
      Array1<int32_t> properties;
      int32_t p;
      GetFsaVecBasicProperties(fsa_vec, &properties, &p);

      ASSERT_EQ(properties.Dim(), num_fsas);
      properties = properties.To(cpu);
      Array1<bool> is_top_sorted = IsTopSorted(cpu_fsa_vec);
      ASSERT_EQ(is_top_sorted.Dim(), num_fsas);
      for (int32_t i = 0; i != num_fsas; ++i) {
        // We randomly reorder `random_fsas` which is top-sorted, so
        // `is_top_sorted_this_fsa` below could be either true or false.
        bool is_top_sorted_this_fsa = (properties[i] & gt) == gt;
        EXPECT_EQ(is_top_sorted_this_fsa, is_top_sorted[i]);
      }

      FsaVec sorted;
      TopSort(fsa_vec, &sorted);

      GetFsaVecBasicProperties(sorted, &properties, &p);
      ASSERT_EQ(properties.Dim(), num_fsas);
      properties = properties.To(cpu);
      FsaVec cpu_sorted = sorted.To(cpu);
      is_top_sorted = IsTopSorted(cpu_sorted);
      ASSERT_EQ(is_top_sorted.Dim(), num_fsas);
      for (int32_t i = 0; i != num_fsas; ++i) {
        Fsa cpu_fsa_vec_i = cpu_fsa_vec.Index(0, i);
        Fsa host_sorted_i;
        TopSortHost(cpu_fsa_vec_i, &host_sorted_i);
        Fsa sorted_i = cpu_sorted.Index(0, i);
        EXPECT_EQ(properties[i] & gt, gt);
        EXPECT_TRUE(is_top_sorted[i]);
        EXPECT_TRUE(IsRandEquivalent(sorted_i, host_sorted_i, false));
      }
    }
  }
}
}  // namespace k2
