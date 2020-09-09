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

#include "k2/csrc/context.cuh"

#include <cstdlib>

#include "k2/csrc/utils.cuh"

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
  int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    lambda(i, j);
  }
}

}  // namespace k2
