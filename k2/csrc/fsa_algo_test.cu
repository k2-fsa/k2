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
  std::string s = R"(0 1 10   -1.2
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

  fsa = fsa.To(GetCudaContext());
  ArcSort(&fsa);

  prop = GetFsaBasicProperties(fsa);
  EXPECT_EQ(prop & kFsaPropertiesArcSorted, kFsaPropertiesArcSorted);
}

}  // namespace k2
