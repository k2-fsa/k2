/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
 *                      Xiaomi Corporation (author: Haowen Qiu
 *                                                  Meixu Song)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cstdlib>

#include "k2/csrc/context.h"

namespace k2 {

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  return std::make_shared<CudaContext>(gpu_id);
}

template <typename LambdaT>
__global__ void eval_lambda(int32_t n, LambdaT lambda) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lambda(i);
  }
}

template <typename LambdaT>
__global__ void eval_lambda2(int32_t m, int32_t n, LambdaT lambda) {
  // actually threadIdx.y will always be 1 for now so we could drop that part of
  // setting i..
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    lambda(i, j);
  }
}

/* Eval() will evaluate lambda(i) for 0 <= i < n, on the appropriate
   device (CPU or GPU). */
template <typename LambdaT>
void Eval(cudaStream_t stream, int32_t n, LambdaT &lambda) {
  if (n <= 0) return;  // actually it would be an error if n < 0.
  if (stream == kCudaStreamInvalid) {
    // TODO: if n is very large, we'll eventually support running this with
    // multiple threads.
    for (int32_t i = 0; i < n; ++i) {
      lambda(i);
    }
  } else {
    int32_t block_size = 256;
    int32_t grid_size = NumBlocks(n, block_size);
    eval_lambda<LambdaT><<<grid_size, block_size, 0, stream>>>(n, lambda);
    auto err = cudaGetLastError();
    K2_DCHECK_CUDA_ERROR(err);
  }
}

/*
  This is a form of Eval() where the lambda takes  two arguments.

  Eval2() will evaluate lambda(i, j) for 0 <= i < m and 0 <= j < n,
  on the appropriate  device (CPU or GPU).  The second index, n,
  is supposed to be the faster-varying one, the index for which
  threads in the same warp will tend to have different values.
  (Of course this doesn't affect the semantics of the operation).
*/
template <typename LambdaT>
void Eval2(cudaStream_t stream, int32_t m, int32_t n, LambdaT &lambda) {
  if (m <= 0 || n <= 0)
    return;  // actually it would be an error if m < 0 or n < 0.
  if (stream == kCudaStreamInvalid) {
    // TODO: if n is very large, we'll eventually support running this with
    // multiple threads.
    for (int32_t i = 0; i < m; ++i) {
      for (int32_t j = 0; j < n; ++j) {
        lambda(i, j);
      }
    }
  } else {
    // this way of choosing block and grid sizes is of course not very smart, we
    // can look at this later on, possibly referring to Kaldi's
    // GetBlockSizesForSimpleMatrixOperation().
    dim3 block_size(16, 16, 1);
    dim3 grid_size(NumBlocks(n, 16), NumBlocks(m, 16));
    eval_lambda2<LambdaT><<<grid_size, block_size, 0, stream>>> (m, n, lambda);
    auto err = cudaGetLastError();
    K2_DCHECK_CUDA_ERROR(err);
  }
}

}  // namespace k2
