/**
 * @brief Unittest for fsa utils.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include "k2/csrc/fsa_utils.h"

namespace k2 {

TEST(FsaFromString, Acceptor) {
  // src_state dst_state label cost
  std::string s = R"(0 1 2   -1.2
    0 2  10 -2.2
    1 3  3  -3.2
    1 6 -1  -4.2
    2 6 -1  -5.2
    2 4  2  -6.2
    5 0  1  -7.2
    6
  )";

  FsaFromString(s);
}

}  // namespace k2
