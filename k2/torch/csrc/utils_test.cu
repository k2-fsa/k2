/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {
TEST(SplitStringToVector, DelimIsSpace) {
  std::string s = "ab c   d e";
  auto ans = SplitStringToVector(s, " ");
  EXPECT_EQ(ans.size(), 4u);
  EXPECT_EQ(ans[0], "ab");
  EXPECT_EQ(ans[1], "c");
  EXPECT_EQ(ans[2], "d");
  EXPECT_EQ(ans[3], "e");
}

TEST(SplitStringToVector, EmptyInput) {
  std::string s = "";
  auto ans = SplitStringToVector(s, " ");
  EXPECT_EQ(ans.size(), 0u);
}

TEST(SplitStringToVector, OnlyOneField) {
  std::string s = "abc";
  auto ans = SplitStringToVector(s, " ");
  EXPECT_EQ(ans.size(), 1u);
  EXPECT_EQ(ans[0], "abc");
}

}  // namespace k2
