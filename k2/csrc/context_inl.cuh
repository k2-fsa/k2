/**
 * @brief
 * context_inl
 * this file is supposed to be included only by context.h
 * It' the implementation of template functions and inlined functions.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu
 *                                                   Meixu Song)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_CONTEXT_INL_H_
#define K2_CSRC_CONTEXT_INL_H_

#ifndef IS_IN_K2_CSRC_CONTEXT_H_
#error "this file is supposed to be included only by context.h"
#endif

namespace k2 {

// Note currently we just support single GPU device, but finally we may need to
// handle different GPU devices on multiple machines, that's also the reason
// that we pass `Context` instead of `DeviceType` as the input parameter here.
inline MemoryCopyKind GetMemoryCopyKind(const Context &src,
                                        const Context &dst) {
  if (src.GetDeviceType() == kCpu && dst.GetDeviceType() == kCpu) {
    return MemcpyHostToHost;
  } else if (src.GetDeviceType() == kCpu && dst.GetDeviceType() == kCuda) {
    return MemcpyHostToDevice;
  } else if (src.GetDeviceType() == kCuda && dst.GetDeviceType() == kCpu) {
    return MemcpyDeviceToHost;
  } else if (src.GetDeviceType() == kCuda && dst.GetDeviceType() == kCuda) {
    return MemcpyDeviceToDevice;
  } else {
    K2_LOG(FATAL) << "Unsupported Context";
    return MemcpyUnknown;
  }
}

inline void MemoryCopy(void *dst, const void *src, std::size_t count,
                       MemoryCopyKind kind) {

  std::map<MemoryCopyKind, cudaMemcpyKind> copy_kind_mappings = {
      {MemcpyHostToHost, cudaMemcpyHostToHost},
      {MemcpyHostToDevice, cudaMemcpyHostToDevice},
      {MemcpyDeviceToHost, cudaMemcpyDeviceToHost},
      {MemcpyDeviceToDevice, cudaMemcpyDeviceToDevice}};
  auto it = copy_kind_mappings.find(kind);
  K2_CHECK_NE(it, copy_kind_mappings.end());
  cudaMemcpy(dst, src, count, it->second);
}

template <typename T1, typename T2>
bool IsCompatible(const T1 &t1, const T2 &t2) {
  // suppose both T1 and T2 have member method `Context`
  return t1.Context()->IsCompatible(*t2.Context());
}

template <typename T>
ContextPtr GetContext(T &t) {
  // suppose T has member method `Context`
  ContextPtr c;
  t.To(c);
  return c;
}

template <typename First, typename... Rest>
ContextPtr GetContext(First &first, Rest &... rest) {
  ContextPtr ans1 = GetContext(first), ans2 = GetContext(rest...);
  K2_DCHECK(ans1->IsCompatible(*ans2)) << "Contexts are not compatible";
  return ans1;
}

/*
  Convenience wrapper for NewRegion() that takes the context from a provided
  region.
 */
inline std::shared_ptr<Region> NewRegion(Region &region, std::size_t num_bytes) {
  return NewRegion(region.context, num_bytes);
}

// Objects from k2 generally have a Context() method, so this template
// will work to get the device-type for pretty arbitrary objects.
template <typename T>
inline DeviceType DeviceOf(const T &t) {
  return t.Context()->GetDeviceType();
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
#ifdef __CUDA_ARCH__
    int32_t block_size = 256;
    int32_t grid_size = NumBlocks(n, block_size);
    eval_lambda<LambdaT><<<grid_size, block_size, 0, stream>>>(n, lambda);
    K2_DCHECK_CUDA_ERROR(cudaGetLastError());
#endif
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
#ifdef __CUDA_ARCH__
    // this way of choosing block and grid sizes is of course not very smart, we
    // can look at this later on, possibly referring to Kaldi's
    // GetBlockSizesForSimpleMatrixOperation().
    dim3 block_size(16, 16, 1);
    dim3 grid_size(NumBlocks(n, 16), NumBlocks(m, 16));
    eval_lambda2<LambdaT><<<grid_size, block_size, 0, stream>>> (m, n, lambda);
    auto err = cudaGetLastError();
    K2_DCHECK_CUDA_ERROR(err);
#endif
  }
}

/* Eval() will evaluate lambda(i) for 0 <= i < n, on the appropriate
   device (CPU or GPU). */
template <typename LambdaT>
void Eval(cudaStream_t stream, int32_t n, LambdaT &lambda);

// Context*  or ContextPtr == std::shared_ptr<Context>
template <typename ContextPtrType, typename LambdaT>
void Eval(ContextPtrType c, int32_t n, LambdaT &lambda) {
  Eval(c->GetCudaStream(), n, lambda);
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
void Eval2(cudaStream_t stream, int32_t m, int32_t n, LambdaT &lambda);

// Context*  or ContextPtr == std::shared_ptr<Context>
template <typename ContextPtrType, typename LambdaT>
void Eval2(ContextPtrType c, int32_t m, int32_t n, LambdaT &lambda) {
  Eval2(c->GetCudaStream(), m, n, lambda);
}

}  // namespace k2

#endif  // K2_CSRC_CONTEXT_INL_H_
