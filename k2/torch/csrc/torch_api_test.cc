/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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
#if HAVE_K2_TORCH_API_H == 1
#include "k2/torch_api.h"  // for third party library
#else
#include "k2/torch/csrc/torch_api.h"
#endif

#include "gtest/gtest.h"

namespace k2 {

TEST(ExclusiveSum, SharedMemory) {
  auto src = torch::tensor({2, 3, 0}, torch::kInt);
  ExclusiveSum(src, &src);

  auto expected = torch::tensor({0, 2, 5}, torch::kInt);
  EXPECT_TRUE(torch::allclose(src, expected));
}

TEST(ExclusiveSum, NotSharedMemory) {
  auto src = torch::tensor({2, 3, 0}, torch::kInt);
  auto dst = torch::empty_like(src);
  ExclusiveSum(src, &dst);

  auto expected = torch::tensor({0, 2, 5}, torch::kInt);
  EXPECT_TRUE(torch::allclose(dst, expected));
}

TEST(RaggedShape, TestAllWrappedMethods) {
  auto sizes = torch::tensor({1, 3, 2, 0}, torch::kInt);
  auto row_splits = torch::empty_like(sizes);
  ExclusiveSum(sizes, &row_splits);

  auto shape = RaggedShape2(row_splits, torch::Tensor(), 6);
  EXPECT_EQ(TotSize(shape, 0), 3);
  EXPECT_EQ(TotSize(shape, 1), 6);

  auto expected_row_ids = torch::tensor({0, 1, 1, 1, 2, 2}, torch::kInt);
  EXPECT_TRUE(torch::allclose(RowIds(shape, 1), expected_row_ids));

  EXPECT_TRUE(torch::allclose(RowSplits(shape, 1), row_splits));
}

}  // namespace k2
