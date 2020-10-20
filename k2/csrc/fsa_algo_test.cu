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
    3 5 -1  -7.2
    3 2 3  -7.2
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
  Intersect(fsa1, fsa2, &fsa_vec, &arc_map_a, &arc_map_b);
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

}  // namespace k2
