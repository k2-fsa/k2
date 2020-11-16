/**
 * @brief Unittest for fsa algorithms.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/test_utils.h"

namespace k2 {
TEST(ArcSort, EmptyFsa) {
  Fsa fsa;
  ArcSort(&fsa);
  EXPECT_LT(fsa.NumAxes(), 2);
}

TEST(ArcSort, NonEmptyFsa) {
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
    int32_t prop = GetFsaBasicProperties(fsa);
    EXPECT_NE(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_NE(prop & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    Fsa sorted;
    ArcSort(fsa, &sorted);
    prop = GetFsaBasicProperties(sorted);
    EXPECT_EQ(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_EQ(prop & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    prop = GetFsaBasicProperties(fsa);
    EXPECT_NE(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_NE(prop & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    // now in-place sort

    ArcSort(&fsa);
    prop = GetFsaBasicProperties(fsa);
    EXPECT_EQ(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_EQ(prop & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);
  }
}

TEST(ArcSort, NonEmptyFsaVec) {
  // src_state dst_state label cost
  std::string s1 = R"(0 1 10 -1.2
    0 2  6 -2.2
    0 3  9 -2.2
    0 3  9 -2.2
    1 2  8  -3.2
    1 3  6  -4.2
    2 3  5 -5.2
    2 4  4  -6.2
    3 5 -1  -7.2
    5
  )";

  std::string s2 = R"(0 1 9 -1.2
    0 2  10 -2.2
    0 3  8 -2.2
    1 2  8  -3.2
    1 4  5  -4.2
    1 3  6  -4.2
    2 3  5 -5.2
    2 4  4  -6.2
    3 2 3  -7.2
    3 5 -1  -7.2
    5
  )";

  Fsa fsa1 = FsaFromString(s1);
  Fsa fsa2 = FsaFromString(s2);
  Fsa *fsa_array[] = {&fsa1, &fsa2};

  for (auto &context : {GetCudaContext(), GetCpuContext()}) {
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);
    Array1<int32_t> properties;
    int32_t p;
    GetFsaVecBasicProperties(fsa_vec, &properties, &p);

    EXPECT_NE(properties[0] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_NE(properties[0] & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    EXPECT_NE(properties[1] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_NE(properties[1] & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    EXPECT_NE(p & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);

    FsaVec sorted;
    ArcSort(fsa_vec, &sorted);
    GetFsaVecBasicProperties(sorted, &properties, &p);
    EXPECT_EQ(properties[0] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);

    EXPECT_EQ(properties[1] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_EQ(properties[1] & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    EXPECT_EQ(p & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);

    GetFsaVecBasicProperties(fsa_vec, &properties, &p);

    EXPECT_NE(properties[0] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_NE(properties[0] & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    EXPECT_NE(properties[1] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_NE(properties[1] & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    EXPECT_NE(p & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);

    // now in-place sort
    ArcSort(&fsa_vec);
    GetFsaVecBasicProperties(fsa_vec, &properties, &p);

    EXPECT_EQ(properties[0] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);

    EXPECT_EQ(properties[1] & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
    EXPECT_EQ(properties[1] & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);

    EXPECT_EQ(p & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
  }
}

TEST(FsaAlgo, LinearFsa) {
  for (auto &context : {GetCudaContext(), GetCpuContext()}) {
    Array1<int32_t> symbols(context, std::vector<int32_t>{10, 20, 30});
    int32_t num_symbols = symbols.Dim();
    Fsa fsa = LinearFsa(symbols);
    ASSERT_EQ(fsa.NumAxes(), 2);
    EXPECT_EQ(fsa.TotSize(0), num_symbols + 2);  // num_states
    EXPECT_EQ(fsa.TotSize(1), num_symbols + 1);  // num_arcs

    fsa = fsa.To(GetCpuContext());  // for testing
    EXPECT_EQ((fsa[{0, 0}]), (Arc{0, 1, 10, 0.f}));
    EXPECT_EQ((fsa[{0, 1}]), (Arc{1, 2, 20, 0.f}));
    EXPECT_EQ((fsa[{0, 2}]), (Arc{2, 3, 30, 0.f}));
    EXPECT_EQ((fsa[{0, 3}]), (Arc{3, 4, -1, 0.f}));
  }
}

TEST(FsaAlgo, LinearFsaVec) {
  /*
   [
    [10, 20],
    [100, 200, 300]
   ]
   */
  for (auto &context : {GetCudaContext(), GetCpuContext()}) {
    Array1<int32_t> row_splits1(context, std::vector<int32_t>{0, 2, 5});
    Array1<int32_t> values(context,
                           std::vector<int32_t>{10, 20, 100, 200, 300});
    RaggedShape shape = RaggedShape2(&row_splits1, nullptr, -1);
    Ragged<int32_t> symbols(shape, values);

    int32_t num_fsas = symbols.Dim0();
    int32_t num_symbols = values.Dim();

    FsaVec fsa = LinearFsas(symbols);
    ASSERT_EQ(fsa.NumAxes(), 3);
    EXPECT_EQ(fsa.TotSize(0), num_fsas);                    // num_fsas
    EXPECT_EQ(fsa.TotSize(1), num_symbols + num_fsas * 2);  // num_states
    EXPECT_EQ(fsa.TotSize(2), num_symbols + num_fsas);      // num_arcs

    fsa = fsa.To(GetCpuContext());  // for testing
    EXPECT_EQ((fsa[{0, 0, 0}]), (Arc{0, 1, 10, 0.f}));
    EXPECT_EQ((fsa[{0, 0, 1}]), (Arc{1, 2, 20, 0.f}));
    EXPECT_EQ((fsa[{0, 0, 2}]), (Arc{2, 3, -1, 0.f}));

    EXPECT_EQ((fsa[{1, 0, 0}]), (Arc{0, 1, 100, 0.f}));
    EXPECT_EQ((fsa[{1, 0, 1}]), (Arc{1, 2, 200, 0.f}));
    EXPECT_EQ((fsa[{1, 0, 2}]), (Arc{2, 3, 300, 0.f}));
    EXPECT_EQ((fsa[{1, 0, 3}]), (Arc{3, 4, -1, 0.f}));
  }
}

TEST(FsaAlgo, IntersectFsaVec) {
  /* Given symbol table
   * <eps> 0
   *  a 1
   *  b 2
   *  c 3
   */

  // ab|ac
  std::string s1 = R"(0 1 1 0.1
    0 2 1 0.2
    1 3 2 0.3
    2 3 3 0.4
    3 4 -1 0.5
    4
  )";
  // ab
  std::string s2 = R"( 0 1 1 10
  1 2 2 20
  2 3 -1 30
  3
  )";
  Fsa fsa1 = FsaFromString(s1);
  Fsa fsa2 = FsaFromString(s2);

  Fsa fsa_vec;
  Array1<int32_t> arc_map_a;
  Array1<int32_t> arc_map_b;
  bool treat_epsilons_specially = true;
  Intersect(fsa1, fsa2, treat_epsilons_specially, &fsa_vec, &arc_map_a,
            &arc_map_b);
  /* fsa_vec is
    0 1 1 10.1      // (0), a_arc_0 + b_arc_0
    0 2 1 10.2      // (1)  a_arc_1 + b_arc_0
    1 2 2 20.3      // (2), a_arc_2 + b_arc_1
    2 3 -1 30.5     // (3), a_arc_4 + b_arc_2
    3
   */
  CheckArrayData(arc_map_a, std::vector<int32_t>{0, 1, 2, 4});
  CheckArrayData(arc_map_b, std::vector<int32_t>{0, 0, 1, 2});

  Fsa intersected_fsa = GetFsaVecElement(fsa_vec, 0);
  Fsa out;
  Array1<int32_t> arc_map;
  Connect(intersected_fsa, &out, &arc_map);
  /* out fsa is
    0 1 1 10.1      // 0 -> in_arc_0
    1 2 2 20.3      // 1 -> in_arc_2
    2 3 -1 30.5     // 2 -> in_arc_3
    3
   */
  CheckArrayData(arc_map, std::vector<int32_t>{0, 2, 3});
}

TEST(FsaAlgo, AddEpsilonSelfLoopsFsa) {
  std::string s1 = R"(0 1 1 0.1
    0 2 1 0.2
    1 3 2 0.3
    2 3 3 0.4
    3 4 -1 0.5
    4
  )";
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i < 3; i++) {
      Fsa fsa1 = FsaFromString(s1).To(context);
      if (i > 0) {
        Fsa fsa2 = Fsa("[ ]").To(context);
        Fsa *fsa_array[] = {&fsa2, &fsa1};
        // note: i below will be 1 or 2
        FsaVec fsa_vec = CreateFsaVec(i, &fsa_array[0]);
        fsa1 = fsa_vec;
      }
      Array1<int32_t> arc_map;
      Fsa fsa2;
      AddEpsilonSelfLoops(fsa1, &fsa2, &arc_map);
      K2_LOG(INFO) << "fsa1 = " << fsa1 << ", fsa1+self-loops = " << fsa2
                   << ", arc-map = " << arc_map;
    }
  }
}

TEST(FsaAlgo, ShortestPath) {
  // best path:
  //   states: 0 -> 1 -> 3 -> 7 -> 9
  //   arcs:     1 -> 3 -> 5 -> 10
  std::string s1 = R"(0 4 1 1
    0 1 1 1
    1 2 1 2
    1 3 1 3
    2 7 1 4
    3 7 1 5
    4 6 1 2
    4 8 1 3
    5 9 -1 4
    6 9 -1 3
    7 9 -1 5
    8 9 -1 6
    9
  )";

  // best path:
  //  states: 0 -> 2 -> 3 -> 4 -> 5
  //  arcs:     1 -> 4 -> 5 -> 7
  //  we add 12 to the arcs to get its indexes in the fsa_vec
  std::string s2 = R"(0 1 1 1
    0 2 2 6
    1 2 3 3
    1 3 4 2
    2 3 5 4
    3 4 6 3
    3 5 -1 2
    4 5 -1 0
    5
  )";

  // best path:
  //   states: 0 -> 2 -> 3
  //   arcs:     1 -> 3
  // we add 20 to the arcs to get its indexes in the fsa_vec
  std::string s3 = R"(0 1 1 10
  0 2 2 100
  1 3 -1 3.5
  2 3 -1 5.5
  3
  )";

  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa fsa3 = FsaFromString(s3);

    Fsa *fsa_array[] = {&fsa1, &fsa2, &fsa3};
    FsaVec fsa_vec = CreateFsaVec(3, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    Ragged<int32_t> state_batches = GetStateBatches(fsa_vec, true);
    Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
    Ragged<int32_t> incoming_arcs = GetIncomingArcs(fsa_vec, dest_states);
    Ragged<int32_t> entering_arc_batches =
        GetEnteringArcIndexBatches(fsa_vec, incoming_arcs, state_batches);

    bool log_semiring = false;
    Array1<int32_t> entering_arcs;
    GetForwardScores<float>(fsa_vec, state_batches, entering_arc_batches,
                            log_semiring, &entering_arcs);

    Ragged<int32_t> best_path_arc_indexes =
        ShortestPath(fsa_vec, entering_arcs);
    CheckArrayData(best_path_arc_indexes.values,
                   std::vector<int32_t>{1, 3, 5, 10, 13, 16, 17, 19, 21, 23});

    FsaVec ans = FsaVecFromArcIndexes(fsa_vec, best_path_arc_indexes);
    ASSERT_EQ(ans.NumAxes(), 3);
    ASSERT_EQ(ans.Dim0(), 3);

    ans = ans.To(GetCpuContext());  // for testing
    EXPECT_EQ((ans[{0, 0, 0}]), (Arc{0, 1, 1, 1.f}));
    EXPECT_EQ((ans[{0, 1, 0}]), (Arc{1, 2, 1, 3.f}));
    EXPECT_EQ((ans[{0, 2, 0}]), (Arc{2, 3, 1, 5.f}));
    EXPECT_EQ((ans[{0, 3, 0}]), (Arc{3, 4, -1, 5.f}));

    EXPECT_EQ((ans[{1, 0, 0}]), (Arc{0, 1, 2, 6.f}));
    EXPECT_EQ((ans[{1, 1, 0}]), (Arc{1, 2, 5, 4.f}));
    EXPECT_EQ((ans[{1, 2, 0}]), (Arc{2, 3, 6, 3.f}));
    EXPECT_EQ((ans[{1, 3, 0}]), (Arc{3, 4, -1, 0.f}));

    EXPECT_EQ((ans[{2, 0, 0}]), (Arc{0, 1, 2, 100.f}));
    EXPECT_EQ((ans[{2, 1, 0}]), (Arc{1, 2, -1, 5.5f}));
  }
}

TEST(FsaAlgo, Union) {
  std::string s1 = R"(0 1 1 0.1
    0 2 2 0.2
    1 3 -1 0.3
    2 3 -1 0.4
    3
  )";
  std::string s2 = R"(0 1 -1 0.5
    1
  )";

  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);
    Array1<int32_t> arc_map;
    Fsa fsa = Union(fsa_vec, &arc_map);
    CheckArrayData(arc_map, std::vector<int32_t>{-1, -1, 0, 1, 2, 3, 4});
  }
}

TEST(FsaAlgo, UnionRandomFsas) {
  int32_t min_num_fsas = 2;
  int32_t max_num_fsas = 5;
  bool acyclic = false;
  int32_t max_symbol = 100;
  int32_t min_num_arcs = 10;
  int32_t max_num_arcs = 100;
  FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic, max_symbol,
                             min_num_arcs, max_num_arcs);

  K2_LOG(INFO) << fsas.Dim0();
  K2_LOG(INFO) << fsas.TotSize(1);
  K2_LOG(INFO) << fsas.TotSize(2);
  Array1<int32_t> arc_map;
  Fsa fsa = Union(fsas, &arc_map);
  ASSERT_EQ(arc_map.Dim(), fsas.NumElements() + fsas.Dim0());
  // Add more tests
}

}  // namespace k2
