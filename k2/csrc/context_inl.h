/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Meixu Song)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef IS_IN_K2_CSRC_CONTEXT_H_
#error "this file is supposed to be included only by context.h"
#endif

// No header guard for this file since it will only be included
// in ops.h

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

inline int32_t NumBlocks(int32_t size, int32_t block_size) {
  return (size + block_size - 1) / block_size;
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
inline void Eval2(ContextPtrType c, int32_t m, int32_t n, LambdaT &lambda) {
  Eval2(c->GetCudaStream(), m, n, lambda);
}

/**
 * @brief
 *
 * @tparam ContextPtrType
 *
 * @todo: implement this
 */
template <typename ContextPtrType>
void ParallelRunner<ContextPtrType>::Finish() {}

}  // namespace k2
