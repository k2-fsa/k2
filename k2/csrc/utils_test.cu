/**
 * @brief
 * utils_test
 *
 * @copyright
 * Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "k2/csrc/utils.h"

namespace k2 {

TEST(UtilsTest, CpuExclusiveSum) {
  void *deleter_context;
  ContextPtr c = GetCpuContext();
  int32_t n = 5;
  // [0, 1, 2, 3, 4]
  // the exclusive prefix sum is [0, 0, 1, 3, 6]
  auto *src = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));
  std::iota(src, src + n, 0);

  auto *dst = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));
  ExclusiveSum(c, n, src, dst);

  EXPECT_THAT(std::vector<int32_t>(dst, dst + n),
              ::testing::ElementsAre(0, 0, 1, 3, 6));

  c->Deallocate(dst, deleter_context);
  c->Deallocate(src, deleter_context);
}

TEST(UtilsTest, CudaExclusiveSum) {
  void *deleter_context;
  ContextPtr c = GetCudaContext();
  int32_t n = 5;
  auto *src = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));

  std::vector<int32_t> h(n);
  std::iota(h.begin(), h.end(), 0);
  cudaMemcpy(src, h.data(), sizeof(int32_t) * n, cudaMemcpyHostToDevice);

  auto *dst = reinterpret_cast<int32_t *>(
      c->Allocate(n * sizeof(int32_t), &deleter_context));
  ExclusiveSum(c, n, src, dst);

  cudaMemcpy(h.data(), dst, sizeof(int32_t) * n, cudaMemcpyDeviceToHost);

  EXPECT_THAT(h, ::testing::ElementsAre(0, 0, 1, 3, 6));

  c->Deallocate(dst, deleter_context);
  c->Deallocate(src, deleter_context);
}

}  // namespace k2
