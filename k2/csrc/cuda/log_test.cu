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

  int a = 10;
  int b = 20;
  int c = 30;
  int d = a;

  K2_CHECK(a == d) << "failed";

  K2_CHECK_EQ(a, d) << "failed";
  K2_CHECK_NE(a, b) << "failed";

  K2_CHECK_LE(a, b) << "failed";
  K2_CHECK_LT(a, b) << "failed";

  K2_CHECK_GE(c, b) << "failed";
  K2_CHECK_GT(c, b) << "failed";
}

__global__ void DummyKernel(int *b, int a) {
  K2_LOG(INFO) << "In kernel";
  K2_CHECK_LT(*b, a) << K2_KERNEL_DEBUG_STR;
  *b += 1;
  K2_CHECK_EQ(*b, a) << K2_KERNEL_DEBUG_STR;
}

TEST(Log, Cuda) {
  K2_LOG(INFO) << "Test log for cuda";
  int a = 10;
  int *b;
  auto ret = cudaMalloc(&b, sizeof(a));
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to allocate memory";

  ret = cudaMemcpy(b, &a, sizeof(a), cudaMemcpyHostToDevice);
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to copy memory to gpu";

  DummyKernel<<<1, 1>>>(b, a + 1);

  int c = 0;
  ret = cudaMemcpy(&c, b, sizeof(a), cudaMemcpyDeviceToHost);
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to copy memory from gpu";
  K2_CHECK_EQ(a + 1, c) << "Error in the kernel!";

  ret = cudaFree(b);
  K2_CHECK_EQ(ret, cudaSuccess) << "Failed to free gpu memory";
}

}  // namespace k2
