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

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
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
  // TODO(fangjun)
}

TEST(TopSort, RandomVectorOfFsas) {
  // TODO(fangjun)
}

}  // namespace k2
