// k2/csrc/cuda/log.h

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "gtest/gtest.h"
#include "k2/csrc/cuda/log.h"

namespace k2 {

TEST(Log, Cpu) {
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
  K2_LOG(INFO) << "Test log for cuda";
  int32_t a = 10;
  int32_t *b;
  auto ret = cudaMalloc(&b, sizeof(a));
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to allocate memory";

  ret = cudaMemcpy(b, &a, sizeof(a), cudaMemcpyHostToDevice);
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to copy memory to gpu";

  DummyKernel<<<1, 1>>>(b, a + 1);

  int32_t c = 0;
  ret = cudaMemcpy(&c, b, sizeof(a), cudaMemcpyDeviceToHost);
  K2_CHECK_CUDA_ERROR(ret) << "Failed to copy memory from gpu";
  K2_CHECK_EQ(a + 1, c) << "Error in the kernel!";

  ret = cudaFree(b);
  K2_DCHECK_CUDA_ERROR(ret) << "Failed to free gpu memory";
}

}  // namespace k2
