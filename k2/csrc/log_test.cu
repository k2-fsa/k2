/**
 * @brief
 * log
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace k2 {

TEST(Log, Cpu) {
  K2_LOG(TRACE) << "Trace message";
  K2_LOG(DEBUG) << "Debug message";
  K2_LOG(INFO) << "Info message";
  K2_LOG(WARNING) << "Warning message";
  K2_LOG(ERROR) << "Error message";

  K2_DLOG(INFO) << "This is printed only in debug mode";

  int32_t a = 10;
  int32_t b = 20;
  int32_t c = 30;
  int32_t d = a;

  K2_DCHECK_EQ(a, d) << "This is checked only in debug mode";

  K2_CHECK(a == d) << "failed";

  K2_CHECK_EQ(a, d) << "failed";
  K2_CHECK_NE(a, b) << "failed";

  K2_CHECK_LE(a, b) << "failed";
  K2_CHECK_LT(a, b) << "failed";

  K2_CHECK_GE(c, b) << "failed";
  K2_CHECK_GT(c, b) << "failed";
}

__global__ void DummyKernel(int32_t *b, int32_t a) {
  K2_DLOG(INFO) << "In kernel";
  K2_DCHECK_LT(*b, a);  // enabled only in debug mode
  *b += 1;
  K2_CHECK_EQ(*b, a);
  K2_DLOG(DEBUG) << "Done";
}

TEST(Log, Cuda) {
  K2_LOG(TRACE) << "Trace message for cuda";
  K2_LOG(INFO) << "Test log for cuda";
  if (GetCudaContext()->GetDeviceType() == kCpu) return;
  int32_t a = 10;
  int32_t *b = nullptr;
  auto ret = cudaMalloc(&b, sizeof(a));
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to allocate memory";

  ret = cudaMemcpy(b, &a, sizeof(a), cudaMemcpyHostToDevice);
  K2_CHECK_CUDA_ERROR(ret) << "Failed to copy memory to gpu";

  DummyKernel<<<1, 1>>>(b, a + 1);

  int32_t c = 0;
  ret = cudaMemcpy(&c, b, sizeof(a), cudaMemcpyDeviceToHost);
  K2_CHECK_CUDA_ERROR(ret) << "Failed to copy memory from gpu";
  K2_CHECK_EQ(a + 1, c) << "Error in the kernel!";

  ret = cudaFree(b);
  K2_CHECK_CUDA_ERROR(ret) << "Failed to free gpu memory";
}

TEST(LogDeathTest, NegativeCases) {
  ASSERT_DEATH(K2_LOG(FATAL) << "This will crash the program", "");

  int32_t a = 10;
  int32_t b = 20;
  int32_t c = a;
  ASSERT_DEATH(K2_CHECK_EQ(a, b), "");
  ASSERT_DEATH(K2_CHECK_NE(a, c), "");

  ASSERT_DEATH(K2_CHECK_LE(b, a), "");
  ASSERT_DEATH(K2_CHECK_LT(b, a), "");

  ASSERT_DEATH(K2_CHECK_GE(a, b), "");
  ASSERT_DEATH(K2_CHECK_GT(a, b), "");

  auto ret = cudaErrorMemoryAllocation;
  ASSERT_DEATH(K2_CHECK_CUDA_ERROR(ret), "");

  ret = cudaErrorAssert;
  ASSERT_DEATH(K2_CHECK_CUDA_ERROR(ret), "");

  // NOTE: normally we do not need to
  // check if NDEBUG is defined in order
  // to use K2_DCHECK_*. ASSERT_DEATH
  // expects that the statement will make
  // the program crash and this is only
  // possible for the debug build,
  // so we have to add a guard here.
#if !defined(NDEBUG)
  K2_LOG(INFO) << "Check for debug build";
  ASSERT_DEATH(K2_DLOG(FATAL) << "This will crash the program", "");

  ASSERT_DEATH(K2_DCHECK_EQ(a, b), "");
  ASSERT_DEATH(K2_DCHECK_NE(a, c), "");

  ASSERT_DEATH(K2_DCHECK_LE(b, a), "");
  ASSERT_DEATH(K2_DCHECK_LT(b, a), "");

  ASSERT_DEATH(K2_DCHECK_GE(a, b), "");
  ASSERT_DEATH(K2_DCHECK_GT(a, b), "");

  ret = cudaErrorInitializationError;
  ASSERT_DEATH(K2_DCHECK_CUDA_ERROR(ret), "");
#endif
}

}  // namespace k2
