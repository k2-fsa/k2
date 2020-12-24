/**
 * @brief Unittest for fsa algorithms.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
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
  Intersect(fsa1, -1, fsa2, -1, treat_epsilons_specially, &fsa_vec, &arc_map_a,
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
  int32_t min_num_fsas = 1;
  int32_t max_num_fsas = 100;
  bool acyclic = false;
  int32_t max_symbol = 100;
  int32_t min_num_arcs = max_num_fsas * 2;
  int32_t max_num_arcs = 10000;
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic, max_symbol,
                               min_num_arcs, max_num_arcs);
    fsas = fsas.To(context);

    Array1<int32_t> arc_map;
    Fsa fsa = Union(fsas, &arc_map);
    ASSERT_EQ(arc_map.Dim(), fsas.NumElements() + fsas.Dim0());

    ContextPtr cpu = GetCpuContext();
    arc_map = arc_map.To(cpu);
    for (int32_t i = 0; i != fsas.Dim0(); ++i) EXPECT_EQ(arc_map[i], -1);

    std::vector<int32_t> arc_old2new(fsas.NumElements());
    for (int32_t i = fsas.Dim0(); i != arc_map.Dim(); ++i) {
      EXPECT_EQ(arc_map[i], i - fsas.Dim0());
      arc_old2new[i - fsas.Dim0()] = i;
    }

    fsas = fsas.To(cpu);
    const int32_t *fsas_row_splits1_data = fsas.RowSplits(1).Data();
    const int32_t *fsas_row_ids1_data = fsas.RowIds(1).Data();
    const int32_t *fsas_row_ids2_data = fsas.RowIds(2).Data();

    for (int32_t i = 0; i != fsas.NumElements(); ++i) {
      Arc old_arc = fsas.values[i];
      Arc new_arc = fsa.values[arc_old2new[i]];

      EXPECT_NEAR(old_arc.score, new_arc.score, 1e-6);
      EXPECT_EQ(old_arc.label, new_arc.label);

      int32_t state_idx01 = fsas_row_ids2_data[i];
      int32_t fsas_idx0 = fsas_row_ids1_data[state_idx01];
      int32_t state_offset = 1 - fsas_idx0;
      int32_t state_idx0x = fsas_row_splits1_data[fsas_idx0];
      EXPECT_EQ(old_arc.src_state + state_offset + state_idx0x,
                new_arc.src_state);
      if (old_arc.label != -1)
        EXPECT_EQ(old_arc.dest_state + state_offset + state_idx0x,
                  new_arc.dest_state);
      else
        EXPECT_EQ(new_arc.dest_state, fsa.Dim0() - 1);
    }

    // now check the new start state
    for (int32_t i = 0; i != fsas.Dim0(); ++i) {
      int32_t state_offset = 1 - i;
      Arc arc = fsa.values[i];
      EXPECT_EQ(arc.src_state, 0);
      EXPECT_EQ(arc.dest_state, fsas_row_splits1_data[i] + state_offset);
      EXPECT_EQ(arc.label, 0);
      EXPECT_EQ(arc.score, 0);
    }
  }
}

TEST(FsaAlgo, RemoveEpsilons) {
  {
    // simple case
    std::string s = R"(0 4 1 1
    0 1 1 1
    1 2 0 2
    1 3 0 3
    1 4 0 2
    2 7 0 4
    3 7 0 5
    4 6 1 2
    4 6 0 3
    4 8 1 3
    4 9 -1 2
    5 9 -1 4
    6 9 -1 3
    7 9 -1 5
    8 9 -1 6
    9
    )";
    Fsa src = FsaFromString(s);
    int32_t prop = GetFsaBasicProperties(src);
    EXPECT_NE(prop & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
    Fsa dest;
    Ragged<int32_t> arc_derivs;
    RemoveEpsilon(src, &dest, &arc_derivs);
    prop = GetFsaBasicProperties(dest);
    EXPECT_EQ(prop & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
    bool log_semiring = false;
    EXPECT_TRUE(IsRandEquivalent(src, dest, log_semiring));
    // TODO(haowen): check arc dervis
    K2_LOG(INFO) << arc_derivs;
  }

  {
    // random case
    int32_t min_num_fsas = 1;
    int32_t max_num_fsas = 1000;
    bool acyclic = true;
    // set max_symbol=10 so that we have a high probability
    // to create Fsas with epsilon arcs.
    int32_t max_symbol = 10;
    int32_t min_num_arcs = 0;
    int32_t max_num_arcs = 10000;
    FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic, max_symbol,
                               min_num_arcs, max_num_arcs);
    FsaVec dest;
    RemoveEpsilon(fsas, &dest);
    Array1<int32_t> properties;
    int32_t p;
    GetFsaVecBasicProperties(dest, &properties, &p);
    EXPECT_EQ(p & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
    bool log_semiring = false;
    float beam = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(IsRandEquivalent(fsas, dest, log_semiring, beam, true, 0.01));
  }
}

TEST(FsaAlgo, Determinize) {
  {
    // simple case
    std::string s = R"(0 4 1 1
    0 1 1 1
    1 2 2 2
    1 3 3 3
    2 7 1 4
    3 7 1 5
    4 6 1 2
    4 6 1 3
    4 5 1 3
    4 8 -1 2
    5 8 -1 4
    6 8 -1 3
    7 8 -1 5
    8
    )";
    Fsa src = FsaFromString(s);
    int32_t prop = GetFsaBasicProperties(src);
    EXPECT_NE(prop & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);
    Fsa dest;
    Ragged<int32_t> arc_derivs;
    Determinize(src, &dest, &arc_derivs);
    K2_LOG(INFO) << arc_derivs;
    bool log_semiring = false;
    EXPECT_TRUE(IsRandEquivalent(src, dest, log_semiring));
    Fsa sorted;
    ArcSort(dest, &sorted);
    prop = GetFsaBasicProperties(sorted);
    EXPECT_EQ(prop & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);
  }

  {
    // random case
    int32_t min_num_fsas = 1;
    int32_t max_num_fsas = 1000;
    bool acyclic = true;
    // set max_symbol=10 so that we have a high probability
    // to create non-deterministic Fsas.
    int32_t max_symbol = 10;
    int32_t min_num_arcs = 0;
    int32_t max_num_arcs = 100000;
    FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic, max_symbol,
                               min_num_arcs, max_num_arcs);
    FsaVec connected;
    Connect(fsas, &connected);
    FsaVec dest;
    Determinize(connected, &dest);
    bool log_semiring = false;
    float beam = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(
        IsRandEquivalent(connected, dest, log_semiring, beam, true, 0.01));
    Fsa sorted;
    ArcSort(dest, &sorted);
    Array1<int32_t> properties;
    int32_t p;
    GetFsaVecBasicProperties(sorted, &properties, &p);
    EXPECT_EQ(p & kFsaPropertiesArcSortedAndDeterministic,
              kFsaPropertiesArcSortedAndDeterministic);
  }
}

TEST(FsaAlgo, ClosureSimpleCase) {
  // 0 -> 1 -> 2 -> 3
  std::string s = R"(0 1 1 0.1
    1 2 2 0.2
    2 3 -1 0.3
    3
  )";

  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa = FsaFromString(s).To(context);
    Array1<int32_t> arc_map;
    Fsa ans = Closure(fsa, &arc_map);

    arc_map = arc_map.To(GetCpuContext());
    CheckArrayData(arc_map, std::vector<int32_t>{0, -1, 1, 2});

    ans = ans.To(GetCpuContext());
    EXPECT_EQ(ans.TotSize(0), 4);  // number of states
    EXPECT_EQ(ans.TotSize(1), 4);  // number of arcs

    EXPECT_EQ((ans[{0, 0}]), (Arc{0, 1, 1, 0.1f}));   // state 0, arc 0
    EXPECT_EQ((ans[{0, 1}]), (Arc{0, 3, -1, 0.0f}));  // state 0, arc 1
    EXPECT_EQ((ans[{1, 0}]), (Arc{1, 2, 2, 0.2f}));   // state 1, arc 0
    EXPECT_EQ((ans[{2, 0}]), (Arc{2, 0, 0, 0.3f}));   // state 2, arc 0
  }
}

TEST(FsaAlgo, ClosureStartStateWithoutLeavingArcs) {
  // the start state has no leaving arcs
  // 1 -> 2 -> 3
  std::string s = R"(1 2 2 0.2
    2 3 -1 0.3
    3
  )";

  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa = FsaFromString(s).To(context);
    Array1<int32_t> arc_map;
    Fsa ans = Closure(fsa, &arc_map);

    arc_map = arc_map.To(GetCpuContext());
    CheckArrayData(arc_map, std::vector<int32_t>{-1, 0, 1});

    ans = ans.To(GetCpuContext());
    EXPECT_EQ(ans.TotSize(0), 4);  // number of states
    EXPECT_EQ(ans.TotSize(1), 3);  // number of arcs

    EXPECT_EQ((ans[{0, 0}]), (Arc{0, 3, -1, 0.0f}));  // state 0, arc 0
    EXPECT_EQ((ans[{1, 0}]), (Arc{1, 2, 2, 0.2f}));   // state 1, arc 0
    EXPECT_EQ((ans[{2, 0}]), (Arc{2, 0, 0, 0.3f}));   // state 2, arc 0
  }
}

TEST(FsaAlgo, ClosureRandomCase) {
  bool acyclic = false;
  int32_t max_symbol = 50;
  int32_t min_num_arcs = 1;
  int32_t max_num_arcs = 1000;

  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa = RandomFsa(acyclic, max_symbol, min_num_arcs, max_num_arcs);
    fsa = fsa.To(context);

    Array1<int32_t> arc_map;
    Fsa ans = Closure(fsa, &arc_map);
    fsa = fsa.To(GetCpuContext());
    ans = ans.To(GetCpuContext());
    arc_map = arc_map.To(GetCpuContext());

    int start_state_num_arcs = fsa.RowSplits(1)[1] - ans.RowSplits(1)[0];
    if (fsa.TotSize(0) < 2) {
      // this is an empty FSA
      EXPECT_EQ(ans.NumElements(), 0);
      EXPECT_EQ(arc_map.Dim(), 0);
      continue;
    }

    // for non-empty FSAs

    const Arc *src_arcs_data = fsa.values.Data();
    const Arc *ans_arcs_data = ans.values.Data();
    int32_t src_num_states = fsa.Dim0();
    for (int32_t i = 0; i != start_state_num_arcs; ++i) {
      EXPECT_EQ(arc_map[i], i);
      if (src_arcs_data[i].dest_state != src_num_states - 1) {
        EXPECT_EQ(src_arcs_data[i], ans_arcs_data[i]);
      } else {
        EXPECT_EQ(src_arcs_data[i].src_state, ans_arcs_data[i].src_state);
        EXPECT_EQ(src_arcs_data[i].score, ans_arcs_data[i].score);
        EXPECT_EQ(ans_arcs_data[i].dest_state, 0);
        EXPECT_EQ(src_arcs_data[i].label, -1);
        EXPECT_EQ(ans_arcs_data[i].label, 0);
      }
    }

    EXPECT_EQ(arc_map[start_state_num_arcs], -1);  // this arc is added by us
    EXPECT_EQ(ans_arcs_data[start_state_num_arcs],
              Arc(0, src_num_states - 1, -1, 0.0f));

    int32_t ans_num_arcs = ans.NumElements();
    for (int32_t i = start_state_num_arcs + 1; i != ans_num_arcs; ++i) {
      EXPECT_EQ(arc_map[i], i - 1);
      if (src_arcs_data[i - 1].dest_state != src_num_states - 1) {
        EXPECT_EQ(src_arcs_data[i - 1], ans_arcs_data[i]);
      } else {
        EXPECT_EQ(src_arcs_data[i - 1].src_state, ans_arcs_data[i].src_state);
        EXPECT_EQ(src_arcs_data[i - 1].score, ans_arcs_data[i].score);
        EXPECT_EQ(ans_arcs_data[i].dest_state, 0);
        EXPECT_EQ(src_arcs_data[i - 1].label, -1);
        EXPECT_EQ(ans_arcs_data[i].label, 0);
      }
    }
  }
}

TEST(FsaAlgo, TestExpandArcsA) {
  FsaVec fsa1("[ [ [ ] [ ] ] ]");
  RaggedShape labels_shape("[]");
  FsaVec fsa1_expanded = ExpandArcs(fsa1, labels_shape, nullptr, nullptr);
  K2_CHECK(Equal(fsa1_expanded, fsa1));
}

TEST(FsaAlgo, TestExpandArcsB) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsa1(c, "[ [ [ 0  1  -1  0.0  ] [ ] ] ]");
    RaggedShape labels_shape(c, "[ [ x ] ]");
    Array1<int32_t> fsa_arc_map, labels_arc_map;
    FsaVec fsa1_expanded =
        ExpandArcs(fsa1, labels_shape, &fsa_arc_map, &labels_arc_map);

    Array1<int32_t> fsa_arc_map_ref(c, "[ 0 ]"), labels_arc_map_ref(c, "[ 0 ]");
    K2_CHECK(Equal(fsa1_expanded, fsa1));
    K2_CHECK(Equal(fsa_arc_map, fsa_arc_map_ref));
    K2_CHECK(Equal(labels_arc_map, labels_arc_map_ref));
  }
}

TEST(FsaAlgo, TestExpandArcsC) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsa1(c, "[ [ [ 0  1  -1  2.0  ] [ ] ] ]");
    RaggedShape labels_shape(c, "[ [ x x ] ]");
    Array1<int32_t> fsa_arc_map, labels_arc_map;
    FsaVec fsa1_expanded =
        ExpandArcs(fsa1, labels_shape, &fsa_arc_map, &labels_arc_map);
    K2_LOG(INFO) << "fsa1_expanded = " << fsa1_expanded;
    FsaVec fsa1_expanded_ref(c, "[ [ [ 0  1 0 2.0 ] [ 1 2 -1 0.0 ] [ ] ] ]");

    Array1<int32_t> fsa_arc_map_ref(c, "[ 0 -1 ]"),
        labels_arc_map_ref(c, "[ 0 1 ]");
    K2_CHECK(Equal(fsa1_expanded, fsa1_expanded_ref));
    K2_CHECK(Equal(fsa_arc_map, fsa_arc_map_ref));
    K2_CHECK(Equal(labels_arc_map, labels_arc_map_ref));
  }
}

TEST(FsaAlgo, TestExpandArcsD) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    FsaVec fsa1(c, "[ [ [ 0  1  -1  2.0  0 1 -1 1.0 ] [ ] ] ]");
    RaggedShape labels_shape(c, "[ [ x x ] [ x ] ]");
    Array1<int32_t> fsa_arc_map, labels_arc_map;
    FsaVec fsa1_expanded =
        ExpandArcs(fsa1, labels_shape, &fsa_arc_map, &labels_arc_map);
    K2_LOG(INFO) << "fsa1_expanded = " << fsa1_expanded;
    FsaVec fsa1_expanded_ref(
        c, "[ [ [ 0  1 0 2.0  0 2 -1 1.0 ] [ 1 2 -1 0.0 ] [ ] ] ]");

    Array1<int32_t> fsa_arc_map_ref(c, "[ 0 1 -1 ]"),
        labels_arc_map_ref(c, "[ 0 2 1 ]");

    K2_LOG(INFO) << "labels_arc_map = " << labels_arc_map
                 << ", fsa_arc_map = " << fsa_arc_map;

    K2_CHECK(Equal(fsa1_expanded, fsa1_expanded_ref));
    K2_CHECK(Equal(fsa_arc_map, fsa_arc_map_ref));
    K2_CHECK(Equal(labels_arc_map, labels_arc_map_ref));
  }
}

TEST(FsaAlgo, TestExpandArcsRandom) {
  int32_t min_num_fsas = 1;
  int32_t max_num_fsas = 100;
  bool acyclic = true;  // so IsRandEquivalent() can work.
  int32_t max_symbol = 100;
  int32_t min_num_arcs = max_num_fsas * 2;
  int32_t max_num_arcs = 10000;
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i < 4; i++) {
      FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic,
                                 max_symbol, min_num_arcs, max_num_arcs)
                        .To(context);
      int32_t num_arcs = fsas.NumElements();
      Array1<int32_t> rand = RandUniformArray1(context, num_arcs + 1, 0, 4);
      ExclusiveSum(rand.Arange(0, num_arcs), &rand);
      RaggedShape labels_shape = RaggedShape2(&rand, nullptr, -1);
      Array1<int32_t> fsa_arc_map, labels_arc_map;
      FsaVec fsas_expanded =
          ExpandArcs(fsas, labels_shape, &fsa_arc_map, &labels_arc_map);
      // note: by default, IsRandEquivalent() does treat epsilons specially,
      // which is what we want.
      K2_CHECK(IsRandEquivalent(fsas, fsas_expanded, false));
      // K2_LOG(INFO) << "fsa_arc_map = " << fsa_arc_map
      ///                   << ", labels_arc_map = " << labels_arc_map;
    }
  }
}

TEST(FsaAlgo, InvertHostTest) {
  // top-sorted FSA
  std::string s1 = R"(0 1 1 0 
    0 1 0 0
    0 3 2 0
    1 2 3 0
    1 3 4 0
    1 5 -1 0
    2 3 0 0
    2 5 -1 0
    4 5 -1 0
    5
    )";
  // non-top-sorted FSA
  std::string s2 = R"(0 1 1 0 
    0 1 0 0
    0 3 2 0
    1 2 3 0
    1 3 4 0
    2 1 5 0
    2 5 -1 0
    3 1 6 0
    4 5 -1 0
    5
    )";
  Fsa fsa1 = FsaFromString(s1);
  Fsa fsa2 = FsaFromString(s2);
  Fsa *fsa_array[] = {&fsa1, &fsa2};
  FsaVec src = CreateFsaVec(2, &fsa_array[0]);
  Ragged<int32_t> aux_labels(
      "[ [1 2] [3] [] [5 6 7] [] [-1] [] [-1] [-1] [1 2] [3] [] [5 6 7] [] "
      "[8] [-1] [9 10] [-1] ]");
  FsaVec dest;
  Ragged<int32_t> dest_aux_labels;
  InvertHost(src, aux_labels, &dest, &dest_aux_labels);
  std::vector<Arc> expected_arcs = {
      {0, 1, 1, 0},  {0, 2, 3, 0},  {0, 6, 0, 0},  {1, 2, 2, 0}, {2, 3, 5, 0},
      {2, 6, 0, 0},  {2, 8, -1, 0}, {3, 4, 6, 0},  {4, 5, 7, 0}, {5, 6, 0, 0},
      {5, 8, -1, 0}, {7, 8, -1, 0}, {0, 1, 1, 0},  {0, 3, 3, 0}, {0, 7, 0, 0},
      {1, 3, 2, 0},  {2, 3, 10, 0}, {3, 4, 5, 0},  {3, 7, 0, 0}, {4, 5, 6, 0},
      {5, 6, 7, 0},  {6, 3, 8, 0},  {6, 9, -1, 0}, {7, 2, 9, 0}, {8, 9, -1, 0}};
  CheckArrayData(dest.values, expected_arcs);
  Ragged<int32_t> expected_aux_labels(
      "[ [] [] [2] [1] [] [4] [-1] [] [3] [] [-1] [-1] [] [] [2] [1] [6] [] "
      "[4] [] [3] [5] [-1] [] [-1]]");
  CheckArrayData(dest_aux_labels.RowSplits(1),
                 expected_aux_labels.RowSplits(1));
  CheckArrayData(dest_aux_labels.values, expected_aux_labels.values);
}

TEST(FsaAlgo, InvertTest) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    // top-sorted FSA
    std::string s1 = R"(0 1 1 1
    0 1 0 2
    0 3 2 3
    1 2 3 4
    1 3 4 5
    1 5 -1 6
    2 3 0 7
    2 5 -1 8
    4 5 -1 9
    5
    )";
    // non-top-sorted FSA
    std::string s2 = R"(0 1 1 1
    0 1 0 2
    0 3 2 3
    1 2 3 4
    1 3 4 5
    2 0 5 6
    2 5 -1 7
    3 1 6 8
    4 5 -1 9
    5
    )";
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec src = CreateFsaVec(2, &fsa_array[0]);
    src = src.To(c);
    Ragged<int32_t> aux_labels(
        c,
        "[ [1 2] [3] [] [5 6 7] [] [8 9 -1] [] [10 -1] [-1] [1 2] [3] [] [5 6 "
        "7] "
        "[] "
        "[8 9] [-1] [10 11] [12 -1] ]");
    FsaVec dest;
    Ragged<int32_t> dest_aux_labels;
    Array1<int32_t> arc_map;
    Invert(src, aux_labels, &dest, &dest_aux_labels, &arc_map);
    FsaVec expected_fsa(
        "[ [ [ 0 1 1 1 0 2 3 2 0 9 0 3 ] [ 1 2 2 0 ] [ 2 3 5 4 2 9 0 5 2 5 8 6 "
        "] "
        "[ 3 4 6 0 ] [ 4 7 7 0 ] [ 5 6 9 0 ] [ 6 11 -1 0 ] [ 7 9 0 7 7 8 10 8 "
        "] "
        "[ 8 11 -1 0 ] [ ] [ 10 11 -1 9 ] [ ] ] [ [ 0 1 1 1 0 2 3 2 0 7 0 3 ] "
        "[ "
        "1 2 2 0 ] [ 2 3 5 4 2 7 0 5 ] [ 3 4 6 0 ] [ 4 5 7 0 ] [ 5 6 8 6 5 11 "
        "-1 "
        "7 ] [ 6 0 9 0 ] [ 7 8 10 8 ] [ 8 2 11 0 ] [ 9 10 12 9 ] [ 10 11 -1 0 "
        "] "
        "[ ] ] ]");
    Ragged<int32_t> expected_aux_labels(
        "[ [ 1 ] [ ] [ 2 ] [ ] [ 3 ] [ 4 ] [ ] [ ] [ ] [ ] [ -1 ] [ ] [ ] [ -1 "
        "] "
        "[ -1 ] [ 1 ] [ ] [ 2 ] [ ] [ 3 ] [ 4 ] [ ] [ ] [ 5 ] [ -1 ] [ ] [ 6 ] "
        "[ "
        "] [ ] [ -1 ] ]");
    Array1<int32_t> expected_arc_map(
        "[ 0 1 2 -1 3 4 5 -1 -1 -1 -1 6 7 -1 8 9 10 11 -1 12 13 -1 -1 14 15 -1 "
        "16 -1 17 -1 ]");
    CheckArrayData(dest.values, expected_fsa.values);
    CheckArrayData(dest_aux_labels.RowSplits(1),
                   expected_aux_labels.RowSplits(1));
    CheckArrayData(dest_aux_labels.values, expected_aux_labels.values);
    CheckArrayData(arc_map, expected_arc_map);

    ContextPtr cpu = GetCpuContext();
    src = src.To(cpu);
    dest = dest.To(cpu);
    aux_labels = aux_labels.To(cpu);
    FsaVec cpu_dest;
    Ragged<int32_t> cpu_dest_aux_labels;
    InvertHost(src, aux_labels, &cpu_dest, &cpu_dest_aux_labels);
    // as fsa2 is not top-sorted
    EXPECT_TRUE(IsRandEquivalentUnweighted(dest, cpu_dest, true));
  }
}

TEST(FsaAlgo, TestInvertRandom) {
  ContextPtr cpu = GetCpuContext();
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i < 1; i++) {
      int32_t min_num_fsas = 1;
      int32_t max_num_fsas = 100;
      bool acyclic = true;  // so IsRandEquivalent() can work.
      int32_t max_symbol = 100;
      int32_t min_num_arcs = max_num_fsas * 2;
      int32_t max_num_arcs = 10000;
      FsaVec src = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic, max_symbol,
                                min_num_arcs, max_num_arcs);
      int32_t num_arcs = src.NumElements();
      Array1<int32_t> rand = RandUniformArray1(cpu, num_arcs + 1, 0, 4);
      ExclusiveSum(rand.Arange(0, num_arcs), &rand);
      RaggedShape aux_labels_shape = RaggedShape2(&rand, nullptr, -1);
      // so that final-arcs always have at least one aux_label (which must be
      // -1)
      aux_labels_shape = ChangeSublistSize(aux_labels_shape, 1);
      Array1<int32_t> aux_labels_value =
          RandUniformArray1(cpu, aux_labels_shape.NumElements(), 0, 50);
      Ragged<int32_t> aux_labels(aux_labels_shape, aux_labels_value);
      // set the last aux_labels to -1 if it's a final-arc
      const int32_t *src_row_splits1 = src.RowSplits(1).Data(),
                    *src_row_ids1 = src.RowIds(1).Data(),
                    *src_row_splits2 = src.RowSplits(2).Data(),
                    *src_row_ids2 = src.RowIds(2).Data(),
                    *aux_labels_row_splits = aux_labels.RowSplits(1).Data();
      int32_t *aux_labels_value_data = aux_labels.values.Data();
      const Arc *arcs_data = src.values.Data();
      for (int32_t idx012 = 0; idx012 != num_arcs; ++idx012) {
        int32_t dest_state_idx1 = arcs_data[idx012].dest_state,
                src_state_idx01 = src_row_ids2[idx012],
                fsa_idx0 = src_row_ids1[src_state_idx01],
                start_state_idx0x = src_row_splits1[fsa_idx0],
                next_start_state_idx0x = src_row_splits1[fsa_idx0 + 1],
                dest_state_idx01 = start_state_idx0x + dest_state_idx1;
        if (next_start_state_idx0x > start_state_idx0x &&
            dest_state_idx01 == next_start_state_idx0x - 1) {
          int32_t aux_labels_idx0x = aux_labels_row_splits[idx012],
                  next_aux_labels_idx0x = aux_labels_row_splits[idx012 + 1];
          if (next_aux_labels_idx0x > aux_labels_idx0x)
            aux_labels_value_data[next_aux_labels_idx0x - 1] = -1;
        }
      }
      FsaVec cpu_dest;
      Ragged<int32_t> cpu_dest_aux_labels;
      InvertHost(src, aux_labels, &cpu_dest, &cpu_dest_aux_labels);

      src.To(context);
      aux_labels.To(context);
      FsaVec dest;
      Ragged<int32_t> dest_aux_labels;
      Invert(src, aux_labels, &dest, &dest_aux_labels);
      dest = dest.To(cpu);
      EXPECT_TRUE(IsRandEquivalent(dest, cpu_dest, true));
    }
  }
}
}  // namespace k2
