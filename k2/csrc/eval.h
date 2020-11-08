/**
 * @brief
 * eval
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_EVAL_H_
#define K2_CSRC_EVAL_H_

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <ostream>
#include <type_traits>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace k2 {

template <typename LambdaT>
__global__ void eval_lambda(int32_t n, LambdaT lambda) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lambda(i);
  }
}

template <typename T, typename LambdaT>
__global__ void eval_lambda_large(int32_t n, LambdaT lambda) {
  int32_t i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    lambda(i);
  }
}

template <typename T, typename LambdaT>
__global__ void set_data_with_lambda(T *data, int32_t n, LambdaT lambda) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = lambda(i);
  }
}

template <typename LambdaT>
__global__ void eval_lambda2_simple(int32_t m, int32_t n, LambdaT lambda) {
  int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    lambda(i, j);
  }
}

template <typename LambdaT>
__global__ void eval_lambda2_zm(int32_t m, int32_t n, LambdaT lambda) {
  int32_t i = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
  int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    lambda(i, j);
  }
}
template <typename LambdaT>
__global__ void eval_lambda2_zn(int32_t m, int32_t n, LambdaT lambda) {
  int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t j = (blockIdx.z * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    lambda(i, j);
  }
}

__host__ __device__ __forceinline__ int32_t NumBlocks(int32_t size,
                                                      int32_t block_size) {
  return (size + block_size - 1) / block_size;
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
    // the loop breaks it up into pieces of size less than 65536.
    if (grid_size < 65536) {
      K2_CUDA_SAFE_CALL(eval_lambda<LambdaT>
                        <<<grid_size, block_size, 0, stream>>>(n, lambda));
    } else {
      int32_t x_grid_size = (grid_size < (1 << 20) ? (1 << 10) : 32768),
              y_grid_size = NumBlocks(grid_size, x_grid_size);
      dim3 grid_dim(x_grid_size, y_grid_size, 1), block_dim(block_size, 1, 1);
      K2_CUDA_SAFE_CALL(eval_lambda_large<LambdaT>
                        <<<grid_dim, block_dim, 0, stream>>>(n, lambda));
    }
  }
}

/*
  Utility function used inside Eval2().
 */

enum class Lambda2KernelType { Simple = 1, UseZForM = 2, UseZForN = 3 };
/* Creates the configuration (block_dim, grid_dim and kernel type) for use in
   Eval2, i.e. for how we invoke eval_lambda2().
      @param [in] m   "outer" dimension (the one we want to be more slowly
                       varying)
      @param [in] n   "inner" dimension (the one which should vary within a
                       warp)
 */
void GetBlockSizesForLambda2(int32_t m, int32_t n, dim3 *block_dim,
                             dim3 *grid_dim, Lambda2KernelType *kernel_type);

template <typename ContextPtrType,  // Context*  or ContextPtr ==
                                    // std::shared_ptr<Context>
          typename LambdaT>
void Eval(ContextPtrType c, int32_t n, LambdaT &lambda) {
  Eval(c->GetCudaStream(), n, lambda);
}

template <typename LambdaT>
void EvalDevice(cudaStream_t stream, int32_t n, LambdaT &lambda) {
  if (n <= 0) return;  // actually it would be an error if n < 0.
  K2_CHECK(stream != kCudaStreamInvalid);
  int32_t block_size = 256;
  int32_t grid_size = NumBlocks(n, block_size);
  // the loop breaks it up into pieces of size less than 65536.
  if (grid_size < 65536) {
    K2_CUDA_SAFE_CALL(eval_lambda<LambdaT>
                      <<<grid_size, block_size, 0, stream>>>(n, lambda));
  } else {
    int32_t x_grid_size = (grid_size < (1 << 20) ? (1 << 10) : 32768),
            y_grid_size = NumBlocks(grid_size, x_grid_size);
    dim3 grid_dim(x_grid_size, y_grid_size, 1), block_dim(block_size, 1, 1);
    K2_CUDA_SAFE_CALL(eval_lambda_large<LambdaT>
                      <<<grid_dim, block_dim, 0, stream>>>(n, lambda));
  }
}

// like Eval() but works only for device.
template <typename ContextPtrType,  // Context*  or ContextPtr ==
                                    // std::shared_ptr<Context>
          typename LambdaT>
void EvalDevice(ContextPtrType c, int32_t n, LambdaT &lambda) {
  EvalDevice(c->GetCudaStream(), n, lambda);
}

/* SetData() will do `data[i] = lambda(i)` for 0 <= i < n, on the appropriate
   device (CPU or GPU) */
template <typename T, typename LambdaT>
void SetData(cudaStream_t stream, T *data, int32_t n, LambdaT &lambda) {
  if (n <= 0) return;  // actually it would be an error if n < 0.
  if (stream == kCudaStreamInvalid) {
    // TODO: if n is very large, we'll eventually support running this with
    // multiple threads.
    for (int32_t i = 0; i < n; ++i) {
      data[i] = lambda(i);
    }
  } else {
    int32_t block_size = 256;
    int32_t grid_size = NumBlocks(n, block_size);
    K2_CUDA_SAFE_CALL(set_data_with_lambda<T, LambdaT>
                      <<<grid_size, block_size, 0, stream>>>(data, n, lambda));
  }
}

template <typename ContextPtrType,  // Context*  or ContextPtr ==
                                    // std::shared_ptr<Context>
          typename T, typename LambdaT>
void SetData(ContextPtrType c, T *data, int32_t n, LambdaT &lambda) {
  SetData(c->GetCudaStream(), data, n, lambda);
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
    dim3 block_dim, grid_dim;
    Lambda2KernelType kernel_type;
    GetBlockSizesForLambda2(m, n, &block_dim, &grid_dim, &kernel_type);
    switch (kernel_type) {
      case Lambda2KernelType::Simple:
        K2_CUDA_SAFE_CALL(
            eval_lambda2_simple<<<grid_dim, block_dim, 0, stream>>>(m, n,
                                                                    lambda));
        break;
      case Lambda2KernelType::UseZForM:
        K2_CUDA_SAFE_CALL(
            eval_lambda2_zm<<<grid_dim, block_dim, 0, stream>>>(m, n, lambda));
        break;
      case Lambda2KernelType::UseZForN:
        K2_CUDA_SAFE_CALL(
            eval_lambda2_zn<<<grid_dim, block_dim, 0, stream>>>(m, n, lambda));
        break;
      default:
        K2_LOG(FATAL) << "Unknown kernel type";
    }
  }
}

/*
  This is a device version of Eval2().

  It will will evaluate lambda(i, j) for 0 <= i < m and 0 <= j < n,
  on the GPU with stream `stream`
 */
template <typename LambdaT>
void Eval2Device(cudaStream_t stream, int32_t m, int32_t n, LambdaT &lambda) {
  if (m <= 0 || n <= 0)
    return;  // actually it would be an error if m < 0 or n < 0.
  K2_DCHECK(stream != kCudaStreamInvalid);
  dim3 block_dim, grid_dim;
  Lambda2KernelType kernel_type;
  GetBlockSizesForLambda2(m, n, &block_dim, &grid_dim, &kernel_type);
  switch (kernel_type) {
    case Lambda2KernelType::Simple:
      K2_CUDA_SAFE_CALL(
          eval_lambda2_simple<<<grid_dim, block_dim, 0, stream>>>(m, n,
                                                                  lambda));
      break;
    case Lambda2KernelType::UseZForM:
      K2_CUDA_SAFE_CALL(
          eval_lambda2_zm<<<grid_dim, block_dim, 0, stream>>>(m, n, lambda));
      break;
    case Lambda2KernelType::UseZForN:
      K2_CUDA_SAFE_CALL(
          eval_lambda2_zn<<<grid_dim, block_dim, 0, stream>>>(m, n, lambda));
      break;
    default:
      K2_LOG(FATAL) << "Unknown kernel type";
  }
}




template <typename ContextPtrType,  // Context*  or ContextPtr ==
                                    // std::shared_ptr<Context>
          typename LambdaT>
inline void Eval2(ContextPtrType c, int32_t m, int32_t n, LambdaT &lambda) {
  Eval2(c->GetCudaStream(), m, n, lambda);
}


// device-only version of Eval2
template <typename ContextPtrType,  // Context*  or ContextPtr ==
                                    // std::shared_ptr<Context>
          typename LambdaT>
inline void Eval2Device(ContextPtrType c, int32_t m,
                        int32_t n, LambdaT &lambda) {
  Eval2Device(c->GetCudaStream(), m, n, lambda);
}

// Also see EvalWithRedirect() in utils.h


}  // namespace k2

#endif  // K2_CSRC_EVAL_H_
