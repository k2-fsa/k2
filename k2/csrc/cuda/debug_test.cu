// k2/csrc/cuda/debug_test.cu

// Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#ifndef K2_PARANOID
#define K2_PARANOID
#endif

#include "k2/csrc/cuda/debug.h"

namespace k2 {

// A dummy cuda asynchronous function
__global__ void FillContents(int n, int *output) {
  int correct_index = static_cast<int>(threadIdx.x) * n;

  for (int i = correct_index; i < n; i++) {
    output[i] = (i - correct_index) * 2;
    K2_DLOG("Print from async cuda kernel, output[%d] = %d\n", i, output[i]);
  }
}

/* number of blocks (max 65,536) */
#define BLOCKS 50000

/* number of threads per block (max 1,024) */
#define THREADS 200

/* N is now the number of blocks times the number of threads
   in this case it is 10 million */
#define N (BLOCKS * THREADS)

/* the kernel fills an array up with square roots */
__global__ void square_roots(double *array) {
  /* find my array index
     this now uses three values:
     blockIdx.x which is the block ID we are on
     blockDim.x which is the number of blocks in the X dimension
     threadIdx.x which is the thread ID we are */
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* compute the square root of this number and store it in the array
     CUDA provides a sqrt function as part of its math library, so this
     call uses that - not the one from math.h which is a CPU function */
  array[idx] = sqrt((double) idx);
}

__global__ void HelloCUDA(float f) {
  if (threadIdx.x == 0)
    K2_DLOG("Hello thread %d, f=%f\n", threadIdx.x, f);
}

// A vector add kernel definition
__global__ void VecAdd(const float *A, const float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
  K2_PARANOID_ASSERT(C[i] == A[i] + B[i],
                     "Error: C[%d] is %f != A[%d] (%f) + B[%d] (%f)\n",
                     i, C[i], i, B[i], i, B[i]);
}

TEST(DebugTest, StaticAssert) {
  K2_STATIC_ASSERT(9 % 3 == 0, "The value is unexpected");
}

TEST(DebugTest, K2Assert) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  K2_ASSERT(2 > 1);
  ASSERT_DEATH(K2_ASSERT(2 < 1), "");
}

TEST(DebugTest, K2CheckEq) {
  K2_CHECK_EQ(2 + 3, 5);
}

TEST(DebugTest, K2CudaCheckError) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(K2_CUDA_CHECK_ERROR(cudaErrorMemoryAllocation, true),
               "cudaErrorMemoryAllocation");
}

TEST(DebugTest, K2CudaSafeCall) {

  /* store the square roots of 0 to (N-1) on the CPU
   * stored on the heap since it's too big for the stack for large values of N */
  double *roots = (double *) malloc(N * sizeof(double));

  /* allocate a GPU array to hold the square roots */
  double *gpu_roots = nullptr;
  K2_CUDA_SAFE_CALL(cudaMalloc((void **) &gpu_roots, N * sizeof(double)));

  /* invoke the GPU to calculate the square roots
     now, we don't run all N blocks, because that may be more than 65,535 */
  K2_CUDA_SAFE_CALL(square_roots<<<BLOCKS, THREADS>>>(gpu_roots));

  /* copy the data back */
  K2_CUDA_SAFE_CALL(cudaMemcpy(roots, gpu_roots, N * sizeof(double),
                        cudaMemcpyDeviceToHost));

  /* free the memory */
  K2_CUDA_SAFE_CALL(cudaFree(gpu_roots));

  // host call K2_DLOG
  {
    /* print out one square root example just to see that it worked */
    unsigned int i = 100000;
    testing::internal::CaptureStdout();
    K2_DLOG("sqrt(%d) = %lf\n", i, roots[i]);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "sqrt(100000) = 316.227766\n");
  }

  /* free the CPU memory */
  free(roots);
}

TEST(DebugTest, K2DLog) {
  int *d_A;
  cudaMalloc(&d_A, 6 * sizeof(int));

  // device call K2_DLOG
  {
    HelloCUDA<<<1, 5>>>(1.2345f);
    FillContents<<<3, 2>>>(1, d_A);
    cudaDeviceSynchronize();
    auto error = K2_CUDA_CHECK_ERROR(cudaGetLastError());
    EXPECT_EQ(error, cudaSuccess);
  }

  cudaFree(d_A);
}

TEST(DebugTest, K2ParanoidAssert) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  {
    int a = 2;
    int b = 1;
    ASSERT_DEATH(
        K2_PARANOID_ASSERT(a < b, "%d unexpectedly smaller than %d\n", a, b),
        "Assertion `a < b' failed");
  }

  {
    int nn = 6;
    size_t size = nn * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float h_A[6] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    float h_B[6] = {0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
    float h_C[6];

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // call K2_PARANOID_ASSERT in kernel
    VecAdd<<<1, 3>>>(d_A, d_B, d_C);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // call K2_PARANOID_ASSERT in host
    K2_PARANOID_ASSERT(h_C[2] == 0.5 && h_C[3] == 0,
                       "h_C[%d] is %f, not 0.6\n", 3, h_C[3]);
    EXPECT_EQ(h_C[2], 0.5);
    EXPECT_EQ(h_C[3], 0.0);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }
}

}  // namespace k2
