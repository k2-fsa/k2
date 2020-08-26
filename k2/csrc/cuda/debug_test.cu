// k2/csrc/cuda/debug_test.cu

// Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors

#include <gtest/gtest.h>

#include "k2/csrc/cuda/debug.cuh"

namespace k2 {

// A dummy cuda asynchronous function
__global__ void FillContents(int N, int *output) {
  int correct_index = static_cast<int>(threadIdx.x) * N;

  for (int i = correct_index; i < N; i++) {
    output[i] = (i - correct_index) * 2;
    K2_DLOG("Print from async cuda-api, output[%d] = %d", i, output[i]);
  }
}

TEST(DebugTest, StaticAssert) {
  K2_STATIC_ASSERT(9 % 3 == 0, "The value is unexpected");
}

TEST(DebugTest, K2CudaCheckError) {
  K2_CUDA_CHECK_ERROR(cudaSuccess);
}

TEST(DebugTest, K2CudaApiSafeCall) {
  K2_CUDA_API_SAFE_CALL(cudaDeviceSynchronize());
}

TEST(DebugTest, K2CudaKernelSafeCall) {
  K2_CUDA_KERNEL_SAFE_CALL(cudaDeviceSynchronize());
}

TEST(DebugTest, K2Dlog) {
  int *unallocated_array = static_cast<int *>(malloc(30 * sizeof(int)));
  // device call
  {
    FillContents<<<2, 3>>>(5, unallocated_array);
  }

  // host call
  {
//    FillContents(5, unallocated_array);
  }
}

}  // end namespace k2
