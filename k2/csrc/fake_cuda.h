/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file is used to compile k2 for CPU only
// The functions in this file are not supposed to be called.
// If they are called by accident, it throws an RuntimeError exception

#ifndef K2_CSRC_FAKE_CUDA_H_
#define K2_CSRC_FAKE_CUDA_H_

#ifdef K2_WITH_CUDA
#error "Don't include this file if k2 is building with CUDA"
#endif

#include "k2/csrc/log.h"

#define __device__
#define __host__
#define __global__
#define __shared__

#ifdef __clang__
// clang does not recognize __forceinline__
#define __forceinline__ inline
#elif defined(__GNUC__)
#define __forceinline__ __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define __forceinline__ __forceinline
#endif

#define K2_NIY  \
  K2_LOG(FATAL) \
      << "Not implemented yet. Don't call me! (Not Compiled with CUDA ?)"

using cudaError_t = int32_t;
using cudaStream_t = int32_t *;
using cudaEvent_t = int32_t *;

#ifndef max
template <typename T>
inline T max(T a, T b) {
  return a < b ? b : a;
}
#endif

#ifndef min
template <typename T>
inline T min(T a, T b) {  // NOLINT
  return b > a ? a : b;
}
#endif

namespace k2 {

struct Dim3 {
  /* implicit */ Dim3(int32_t x = 0, int32_t y = 0, int32_t z = 0) {}  // NOLINT
  int32_t x;
  int32_t y;
  int32_t z;
};
using dim3 = Dim3;

static Dim3 threadIdx;
static Dim3 blockIdx;
static Dim3 blockDim;
static Dim3 gridDim;

enum FakedEnum {
  cudaSuccess = -1,
  cudaErrorNotReady = -1,
  cudaErrorInitializationError = -1,
  cudaMemcpyDeviceToHost = -1,
  cudaMemcpyHostToDevice = -1,
  cudaMemcpyDeviceToDevice = -1,
  cudaEventDisableTiming = -1,
};

using cudaMemcpyKind = FakedEnum;

inline const char *cudaGetErrorString(cudaError_t error) {
  K2_NIY;
  return nullptr;
}

inline cudaError_t cudaGetLastError() {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaDeviceSynchronize() {
  K2_NIY;
  return 0;
}

inline void __syncthreads() { K2_NIY; }

inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                              cudaMemcpyKind kind) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                   cudaMemcpyKind kind,
                                   cudaStream_t stream = 0) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaEventCreate(cudaEvent_t *event) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
                                            unsigned int flags) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaEventQuery(cudaEvent_t event) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                       unsigned int flags = 0) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  K2_NIY;
  return 0;
}

template <typename ContextPtrType, typename T>
void SegmentedExclusiveSum(ContextPtrType context, const T *d_in,
                           int32_t num_elements, const uint32_t *d_iflags,
                           T *d_out) {
  K2_NIY;
}

inline cudaError_t cudaGetDeviceCount(int *count) {
  K2_NIY;
  return 0;
}

inline cudaError_t cudaMallocHost(void **ptr, size_t size) {
  K2_NIY;
  return 0;
}

}  // namespace k2

namespace cub {

namespace DeviceScan {

template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT,
          typename InitValueT>
cudaError_t ExclusiveScan(void *d_temp_storage, size_t &temp_storage_bytes,
                          InputIteratorT d_in, OutputIteratorT d_out,
                          ScanOpT scan_op, InitValueT init_value, int num_items,
                          cudaStream_t stream = 0,
                          bool debug_synchronous = false) {
  K2_NIY;
  return 0;
}

template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT>
cudaError_t InclusiveScan(void *d_temp_storage, size_t &temp_storage_bytes,
                          InputIteratorT d_in, OutputIteratorT d_out,
                          ScanOpT scan_op, int num_items,
                          cudaStream_t stream = 0,
                          bool debug_synchronous = false) {
  K2_NIY;
  return 0;
}

template <typename InputIteratorT, typename OutputIteratorT>
cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes,
                         InputIteratorT d_in, OutputIteratorT d_out,
                         int num_items, cudaStream_t stream = 0,
                         bool debug_synchronous = false) {
  K2_NIY;
  return 0;
}

}  // namespace DeviceScan

namespace DeviceReduce {

template <typename InputIteratorT, typename OutputIteratorT,
          typename ReductionOpT, typename T>
cudaError_t Reduce(void *d_temp_storage, size_t &temp_storage_bytes,
                   InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                   ReductionOpT reduction_op, T init, cudaStream_t stream = 0,
                   bool debug_synchronous = false) {
  K2_NIY;
  return 0;
}

}  // namespace DeviceReduce

template <typename T, int BLOCK_DIM_X>
struct BlockScan {
  struct TempStorage {};
  explicit BlockScan(const TempStorage &) { K2_NIY; }
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  void InclusiveScan(T input, T &output, ScanOp scan_op,
                     BlockPrefixCallbackOp &block_prefix_callback_op) {
    K2_NIY;
  }
};

}  // namespace cub

namespace mgpu {
enum FakedEnum {
  memory_space_device = -1,
};

using memory_space_t = FakedEnum;

class standard_context_t {
 public:
  standard_context_t(bool, cudaStream_t) { K2_NIY; }

  virtual void *alloc(size_t, memory_space_t) = 0;

  virtual void free(void *, memory_space_t) = 0;

  virtual ~standard_context_t() = default;
};

using context_t = standard_context_t;

// Key-value mergesort.
template <typename key_t, typename val_t, typename comp_t>
void mergesort(key_t *keys_input, val_t *vals_input, int count, comp_t comp,
               context_t &context) {
  K2_NIY;
}

// Key-only mergesort
template <typename key_t, typename comp_t>
void mergesort(key_t *keys_input, int count, comp_t comp, context_t &context) {
  K2_NIY;
}

template <typename T>
struct tuple {};  // NOLINT

template <int32_t I, typename T>
T get(const tuple<T> &) {  // NOLINT
  K2_NIY;
  return T();
}

template <typename segments_it, typename output_it>
void load_balance_search(int count, segments_it segments, int num_segments,
                         output_it output, context_t &context) {
  K2_NIY;
}

enum bounds_t { bounds_lower };

template <bounds_t bounds, typename needles_it, typename haystack_it,
          typename indices_it, typename comp_it>
void sorted_search(needles_it needles, int num_needles, haystack_it haystack,
                   int num_haystack, indices_it indices, comp_it comp,
                   context_t &context) {
  K2_NIY;
}

}  // namespace mgpu

#endif  // K2_CSRC_FAKE_CUDA_H_
