/**
 * @brief
 * ragged_utils_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu, Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/ragged_utils.h"

namespace k2 {

TEST(RaggedUtilsTest, CheckAxisEqual) {

  RaggedShape shape1(" [[ x x x ] [ x x ]]"),
      shape1b(" [[ x x x ] [ x x ]]"),
      shape2("[[ x x x ] [ x ]]");

  RaggedShape *array[] = { &shape1, &shape1b, &shape1, &shape2, &shape2 };
  int32_t axis = 0;
  CheckAxisEqual(0, axis, array);
  CheckAxisEqual(1, axis, array);
  CheckAxisEqual(2, axis, array);
  CheckAxisEqual(3, axis, array);
#ifndef NDEBUG
  // this won't actualy die if we compiled with NDEBUG.
  ASSERT_DEATH(CheckAxisEqual(4, axis, array), "");
#endif
  CheckAxisEqual(2, axis, array + 3);
}

TEST(RaggedUtilsTest, GetLayer) {

  RaggedShape shape1(" [[[ x x x ] [ x x ]]]"),
      shape2(" [[ x x x ] [ x x ]]"),
      shape3("[[x x]]");

  RaggedShape shape2b = GetLayer(shape1, 1),
              shape3b = GetLayer(shape1, 0);
  ASSERT_TRUE(Equal(shape2, shape2b));
  ASSERT_TRUE(Equal(shape3, shape3b));
}



}  // namespace k2
