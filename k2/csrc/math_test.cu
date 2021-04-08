/**
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include "gtest/gtest.h"
#include "k2/csrc/algorithms.h"
#include "k2/csrc/math.h"

namespace k2 {
TEST(Math, HighestBitSet) {
  K2_CHECK_EQ(HighestBitSet(int32_t(0)), -1);
  K2_CHECK_EQ(HighestBitSet(int32_t(1)), 0);
  K2_CHECK_EQ(HighestBitSet(int32_t(2)), 1);
  K2_CHECK_EQ(HighestBitSet(int32_t(3)), 1);
  K2_CHECK_EQ(HighestBitSet(int32_t(4)), 2);
  K2_CHECK_EQ(HighestBitSet(int32_t(5)), 2);

  K2_CHECK_EQ(HighestBitSet(int32_t(1) << 29), 29);
  K2_CHECK_EQ(HighestBitSet((int32_t(1) << 29) + 1), 29);
  K2_CHECK_EQ(HighestBitSet(int32_t(1) << 30), 30);
  K2_CHECK_EQ(HighestBitSet((int32_t(1) << 30) + 1), 30);

  K2_CHECK_EQ(HighestBitSet(int64_t(0)), -1);
  K2_CHECK_EQ(HighestBitSet(int64_t(1)), 0);
  K2_CHECK_EQ(HighestBitSet(int64_t(2)), 1);
  K2_CHECK_EQ(HighestBitSet(int64_t(3)), 1);
  K2_CHECK_EQ(HighestBitSet(int64_t(4)), 2);
  K2_CHECK_EQ(HighestBitSet(int64_t(5)), 2);

  K2_CHECK_EQ(HighestBitSet(int64_t(1) << 50), 50);
  K2_CHECK_EQ(HighestBitSet((int64_t(1) << 50) + 1), 50);
}
}  // namespace k2
