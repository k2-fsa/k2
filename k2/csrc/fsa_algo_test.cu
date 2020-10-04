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

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"

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
  Fsa fsa = FsaFromString(s);
  int32_t prop = GetFsaBasicProperties(fsa);
  EXPECT_NE(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
  EXPECT_NE(prop & kFsaPropertiesArcSortedAndDeterministic,
            kFsaPropertiesArcSortedAndDeterministic);

  fsa = fsa.To(GetCudaContext());
  ArcSort(&fsa);

  prop = GetFsaBasicProperties(fsa);
  EXPECT_EQ(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
  EXPECT_EQ(prop & kFsaPropertiesArcSortedAndDeterministic,
            kFsaPropertiesArcSortedAndDeterministic);
}

TEST(ArcSort, NonEmptyFsaVec) {
  // src_state dst_state label cost
  std::string s1 = R"(0 1 10 -1.2
    0 2  6 -2.2
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
  const Fsa *fsa_array[] = {&fsa1, &fsa2};

  // FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);  // crash!
#if 0
  Array1<int32_t> properties;
  int32_t p;
  K2_LOG(INFO) << FsaPropertiesAsString(properties[0]);
  K2_LOG(INFO) << FsaPropertiesAsString(properties[1]);
#endif
}

}  // namespace k2
