/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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

#include <gtest/gtest.h>

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace k2 {

TEST(Log, Cpu) {
  K2_LOG(TRACE) << "Trace message";
  K2_LOG(DEBUG) << "Debug message";
  K2_LOG(INFO) << "Info message";
  K2_LOG(WARNING) << "Warning message";
#ifndef _MSC_VER
  // It fails on Windows with the following error:
  // k2/csrc/log_test.cu(31): error : expected a ")"
  K2_LOG(ERROR) << "Error message";
#endif

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
#ifdef K2_WITH_CUDA
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
#endif

TEST(LogDeathTest, NegativeCases) {
  ASSERT_THROW(K2_LOG(FATAL) << "This will crash the program",
               std::runtime_error);

  int32_t a = 10;
  int32_t b = 20;
  int32_t c = a;
  ASSERT_THROW(K2_CHECK_EQ(a, b), std::runtime_error);
  ASSERT_THROW(K2_CHECK_NE(a, c), std::runtime_error);

  ASSERT_THROW(K2_CHECK_LE(b, a), std::runtime_error);
  ASSERT_THROW(K2_CHECK_LT(b, a), std::runtime_error);

  ASSERT_THROW(K2_CHECK_GE(a, b), std::runtime_error);
  ASSERT_THROW(K2_CHECK_GT(a, b), std::runtime_error);

#ifdef K2_WITH_CUDA
  auto ret = cudaErrorMemoryAllocation;
  ASSERT_THROW(K2_CHECK_CUDA_ERROR(ret), std::runtime_error);

  ret = cudaErrorAssert;
  ASSERT_THROW(K2_CHECK_CUDA_ERROR(ret), std::runtime_error);
#endif

  // NOTE: normally we do not need to
  // check if NDEBUG is defined in order
  // to use K2_DCHECK_*. ASSERT_DEATH
  // expects that the statement will make
  // the program crash and this is only
  // possible for the debug build,
  // so we have to add a guard here.
#if !defined(NDEBUG)
  K2_LOG(INFO) << "Check for debug build";
  ASSERT_THROW(K2_DLOG(FATAL) << "This will crash the program",
               std::runtime_error);

  ASSERT_THROW(K2_DCHECK_EQ(a, b), std::runtime_error);
  ASSERT_THROW(K2_DCHECK_NE(a, c), std::runtime_error);

  ASSERT_THROW(K2_DCHECK_LE(b, a), std::runtime_error);
  ASSERT_THROW(K2_DCHECK_LT(b, a), std::runtime_error);

  ASSERT_THROW(K2_DCHECK_GE(a, b), std::runtime_error);
  ASSERT_THROW(K2_DCHECK_GT(a, b), std::runtime_error);

#ifdef K2_WITH_CUDA
  ret = cudaErrorInitializationError;
  ASSERT_THROW(K2_DCHECK_CUDA_ERROR(ret), std::runtime_error);
#endif
#endif
}

}  // namespace k2
