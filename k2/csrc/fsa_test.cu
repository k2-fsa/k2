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
  EXPECT_TRUE(s.empty());
}

TEST(FsaPropertiesAsString, NonEmpty) {
  auto s = FsaPropertiesAsString(kFsaPropertiesValid);
  EXPECT_EQ(s, "kFsaPropertiesValid");

  s = FsaPropertiesAsString(kFsaPropertiesNonempty | kFsaPropertiesValid);
  EXPECT_EQ(s, "kFsaPropertiesValid|kFsaPropertiesNonempty");

  s = FsaPropertiesAsString(kFsaPropertiesTopSorted | kFsaPropertiesValid |
                            kFsaPropertiesSerializable);
  EXPECT_EQ(
      s,
      "kFsaPropertiesValid|kFsaPropertiesTopSorted|kFsaPropertiesSerializable");
}

}  // namespace k2
