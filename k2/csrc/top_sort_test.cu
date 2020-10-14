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

  int32_t prop = GetFsaBasicProperties(fsa);
  EXPECT_NE(prop & kFsaPropertiesTopSorted, kFsaPropertiesTopSorted);

  Fsa sorted;
  Array1<int32_t> arc_map;
  TopSort(fsa, &sorted, &arc_map);
  prop = GetFsaBasicProperties(sorted);
  EXPECT_EQ(prop & kFsaPropertiesTopSorted, kFsaPropertiesTopSorted);
  /* top sorted fst is
  0 2 1 1    // src arc 0
  0 1 2 2    // src arc 1
  1 2 1 21   // src arc 3
  1 3 -1 23  // src arc 4
  2 3 -1 13  // src arc 2
  3
  */

  CheckArrayData(arc_map, {0, 1, 3, 4, 2});
}

}  // namespace k2
