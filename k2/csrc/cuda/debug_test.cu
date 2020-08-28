// k2/csrc/cuda/debug_test.cu

// Copyright (c) 2020, Xiaomi Corporation ( authors: Meixu Song )

// See ../../LICENSE for clarification regarding multiple authors

#include <gtest/gtest.h>

#include "k2/csrc/cuda/debug.h"

namespace k2 {

// A dummy cuda asynchronous function
__global__ void FillContents(int n, int *output) {
  int correct_index = static_cast<int>(threadIdx.x) * n;

  for (int i = correct_index; i < n; i++) {
    output[i] = (i - correct_index) * 2;
    K2_DLOG("Print from async cuda-api, output[%d] = %d", i, output[i]);
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
__global__ void square_roots(double* array) {
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

TEST(DebugTest, StaticAssert) {
  K2_STATIC_ASSERT(9 % 3 == 0, "The value is unexpected");
}

TEST(DebugTest, K2CudaCheckError) {
  K2_CUDA_CHECK_ERROR(cudaSuccess);
}

TEST(DebugTest, K2CudaSafeCall) {

  /* store the square roots of 0 to (N-1) on the CPU
   * stored on the heap since it's too big for the stack for large values of N */
  double* roots = (double*) malloc(N * sizeof(double));

  /* allocate a GPU array to hold the square roots */
  double *gpu_roots = nullptr;
  K2_CUDA_API_SAFE_CALL(cudaMalloc((void**) &gpu_roots, N * sizeof(double)));

  /* invoke the GPU to calculate the square roots
     now, we don't run all N blocks, because that may be more than 65,535 */
  K2_CUDA_KERNEL_SAFE_CALL(square_roots<<<BLOCKS, THREADS>>>(gpu_roots));

  /* copy the data back */
  K2_CUDA_API_SAFE_CALL(cudaMemcpy(roots, gpu_roots, N * sizeof(double),
                        cudaMemcpyDeviceToHost));

  /* free the memory */
  K2_CUDA_API_SAFE_CALL(cudaFree(gpu_roots));

  // host call K2_DLOG
  {
    /* print out 100 evenly spaced square roots just to see that it worked */
    unsigned int i;
    for (i = 0; i < N; i += (N / 100)) {
      K2_DLOG("sqrt(%d) = %lf\n", i, roots[i]);
    }
  }

  /* free the CPU memory */
  free(roots);
}

TEST(DebugTest, K2DLog) {
  int *unallocated_array = static_cast<int *>(malloc(30 * sizeof(int)));
  // device call K2_DLOG
  {
    HelloCUDA<<<1, 5>>>(1.2345f);
    FillContents<<<2, 3>>>(5, unallocated_array);
  }
}

}  // namespace k2
