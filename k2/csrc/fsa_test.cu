/**
 * @brief Unittest for fsa.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include "k2/csrc/fsa.h"

namespace k2 {

TEST(FsaPropertiesAsString, Empty) {
  auto s = FsaPropertiesAsString(0);
  EXPECT_EQ(s, "\"\"");
}

TEST(FsaPropertiesAsString, NonEmpty) {
  auto s = FsaPropertiesAsString(kFsaPropertiesValid);
  EXPECT_EQ(s, "\"Valid\"");

  s = FsaPropertiesAsString(kFsaPropertiesNonempty | kFsaPropertiesValid);
  EXPECT_EQ(s, "\"Valid|Nonempty\"");

  s = FsaPropertiesAsString(kFsaPropertiesTopSorted | kFsaPropertiesValid |
                            kFsaPropertiesSerializable);
  EXPECT_EQ(
      s,
      "\"Valid|TopSorted|Serializable\"");
}

}  // namespace k2
