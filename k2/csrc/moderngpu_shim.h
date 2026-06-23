/**
 * Copyright (c) 2026  Advanced Micro Devices, Inc. (authors: Jeff Daily <jeff.daily@amd.com>)
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

// HIP replacement for the small slice of moderngpu that k2 uses. moderngpu is
// not portable to ROCm (its intrinsics.hxx #errors under non-nvcc, hardcodes a
// 32-lane warp, and uses inline PTX), so on the HIP build we do NOT compile
// moderngpu at all. Instead this header provides the exact mgpu::-shaped API
// that k2's call sites expect, backed by rocThrust and a couple of small
// kernels, so every existing `mgpu::...` call compiles unchanged. The CUDA
// build never sees this file (moderngpu.h includes the real moderngpu there).
//
// Semantics are matched to moderngpu:
//  - mergesort / segmented_sort* are STABLE (moderngpu's are, and k2's CPU
//    reference uses std::stable_sort; index maps must be reproducible);
//  - the comparators k2 passes are arbitrary device callables (Arc/ArcComparer,
//    device lambdas, LessThan/GreaterThan), which rocThrust handles directly
//    and a radix sort could not.

#ifndef K2_CSRC_MODERNGPU_SHIM_H_
#define K2_CSRC_MODERNGPU_SHIM_H_

#if !defined(K2_WITH_HIP)
#error "moderngpu_shim.h is only for the HIP build"
#endif

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <hipcub/hipcub.hpp>
#include <thrust/binary_search.h>  // NOLINT(build/include_order)
#include <thrust/execution_policy.h>  // NOLINT(build/include_order)
#include <thrust/iterator/counting_iterator.h>  // NOLINT(build/include_order)
#include <thrust/sequence.h>  // NOLINT(build/include_order)
#include <thrust/sort.h>  // NOLINT(build/include_order)

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace mgpu {

// moderngpu's allocator/context base. Here it only needs to carry the k2
// Context (for device allocate/deallocate) and the HIP stream.
struct context_t {
  k2::ContextPtr k2_context;

  context_t() = default;
  explicit context_t(k2::ContextPtr c) : k2_context(std::move(c)) {}

  // Re-query each time so a CudaStreamOverride in effect is honored (matches
  // how the CUDA build reads the stream).
  hipStream_t stream() const { return k2_context->GetCudaStream(); }
};

// k2's GetModernGpuAllocator returns a `standard_context_t`-derived object on
// CUDA; on HIP the plain context_t is enough.
using standard_context_t = context_t;

// ---- transform_lbs caching-tuple support (CatWithOffsets) -----------------
// k2 calls transform_lbs with an optional mgpu::make_tuple(ptr...) whose values
// are loaded per-segment and passed to the lambda as mgpu::tuple<...>.
template <typename... Ts>
struct tuple {
  // Only the single-element case is exercised by k2, but keep it general.
};

template <typename T>
struct tuple<T> {
  T v0;
};

template <typename T>
__host__ __device__ __forceinline__ const T &get_impl(
    const tuple<T> &t, std::integral_constant<int, 0>) {
  return t.v0;
}

template <int I, typename... Ts>
__host__ __device__ __forceinline__ auto get(const tuple<Ts...> &t)
    -> decltype(get_impl(t, std::integral_constant<int, I>())) {
  return get_impl(t, std::integral_constant<int, I>());
}

// A tuple of device pointers; indexing it by segment yields a tuple of values.
template <typename... Ts>
struct ptr_tuple {};

template <typename T>
struct ptr_tuple<T> {
  const T *p0;
  __device__ __forceinline__ tuple<T> at(int32_t seg) const {
    return tuple<T>{p0[seg]};
  }
};

template <typename T>
__host__ __forceinline__ ptr_tuple<T> make_tuple(const T *p0) {
  return ptr_tuple<T>{p0};
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace shim_internal {

constexpr int32_t kBlockSize = 256;

template <typename T>
inline thrust::device_ptr<T> dptr(T *p) {
  return thrust::device_pointer_cast(p);
}

// row_ids[i] = segment of element i = upper_bound(row_splits[1..nsegs], i),
// i.e. the moderngpu load-balance-search result. row_splits has nsegs+1
// entries.
inline void ComputeRowIds(context_t &ctx, int32_t count,
                          const int32_t *row_splits, int32_t num_segments,
                          int32_t *row_ids) {
  if (count <= 0) return;
  auto policy = thrust::hip::par.on(ctx.stream());
  // ends = row_splits + 1 (the per-segment end offsets).
  thrust::upper_bound(
      policy, dptr(const_cast<int32_t *>(row_splits)) + 1,
      dptr(const_cast<int32_t *>(row_splits)) + 1 + num_segments,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(count), dptr(row_ids));
}

// Grid-stride kernel that invokes the user lambda f(index, seg, rank) for the
// plain transform_lbs; row_ids/row_splits give seg and rank.
template <typename Lambda>
__global__ void TransformLbsKernel(int32_t count, const int32_t *row_ids,
                                    const int32_t *row_splits, Lambda f) {
  for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x) {
    int32_t seg = row_ids[index];
    int32_t rank = index - row_splits[seg];
    f(index, seg, rank);
  }
}

// Same, with a per-segment cached tuple passed as the 4th lambda argument.
template <typename Lambda, typename PtrTuple>
__global__ void TransformLbsTupleKernel(int32_t count, const int32_t *row_ids,
                                        const int32_t *row_splits,
                                        PtrTuple cached, Lambda f) {
  for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x) {
    int32_t seg = row_ids[index];
    int32_t rank = index - row_splits[seg];
    f(index, seg, rank, cached.at(seg));
  }
}

// Scratch device buffer tied to a k2 Context (used for row_ids).
struct DeviceScratch {
  k2::ContextPtr c;
  void *data = nullptr;
  void *deleter_context = nullptr;
  DeviceScratch(k2::ContextPtr c, size_t bytes) : c(std::move(c)) {
    data = this->c->Allocate(bytes, &deleter_context);
  }
  ~DeviceScratch() { c->Deallocate(data, deleter_context); }
  template <typename T>
  T *as() {
    return reinterpret_cast<T *>(data);
  }
};

// Copy the nsegs+1 segment offsets to host so we can iterate segments.
inline std::vector<int32_t> SegmentsToHost(context_t &ctx,
                                           const int32_t *segments,
                                           int32_t num_segments) {
  std::vector<int32_t> host(num_segments + 1);
  K2_CHECK_EQ(hipMemcpyAsync(host.data(), segments,
                             (num_segments + 1) * sizeof(int32_t),
                             hipMemcpyDeviceToHost, ctx.stream()),
              hipSuccess);
  K2_CHECK_EQ(hipStreamSynchronize(ctx.stream()), hipSuccess);
  return host;
}

}  // namespace shim_internal

// ---------------------------------------------------------------------------
// transform_lbs: f(index, seg, rank) for each index in [0, count).
// ---------------------------------------------------------------------------
template <typename Lambda>
void transform_lbs(Lambda f, int32_t count, const int32_t *row_splits,
                   int32_t num_segments, context_t &ctx) {
  if (count <= 0) return;
  shim_internal::DeviceScratch row_ids_buf(ctx.k2_context,
                                           count * sizeof(int32_t));
  int32_t *row_ids = row_ids_buf.as<int32_t>();
  shim_internal::ComputeRowIds(ctx, count, row_splits, num_segments, row_ids);

  int32_t grid = (count + shim_internal::kBlockSize - 1) /
                 shim_internal::kBlockSize;
  shim_internal::TransformLbsKernel<<<grid, shim_internal::kBlockSize, 0,
                                      ctx.stream()>>>(count, row_ids,
                                                      row_splits, f);
}

template <typename Lambda, typename... Ts>
void transform_lbs(Lambda f, int32_t count, const int32_t *row_splits,
                   int32_t num_segments, ptr_tuple<Ts...> cached,
                   context_t &ctx) {
  if (count <= 0) return;
  shim_internal::DeviceScratch row_ids_buf(ctx.k2_context,
                                           count * sizeof(int32_t));
  int32_t *row_ids = row_ids_buf.as<int32_t>();
  shim_internal::ComputeRowIds(ctx, count, row_splits, num_segments, row_ids);

  int32_t grid = (count + shim_internal::kBlockSize - 1) /
                 shim_internal::kBlockSize;
  shim_internal::TransformLbsTupleKernel<<<grid, shim_internal::kBlockSize, 0,
                                           ctx.stream()>>>(
      count, row_ids, row_splits, cached, f);
}

// ---------------------------------------------------------------------------
// mergesort: stable sort of keys, optionally permuting an index/value array.
// ---------------------------------------------------------------------------
template <typename Key, typename Comp>
void mergesort(Key *keys, int32_t count, Comp comp, context_t &ctx) {
  if (count <= 0) return;
  auto policy = thrust::hip::par.on(ctx.stream());
  thrust::stable_sort(policy, shim_internal::dptr(keys),
                      shim_internal::dptr(keys) + count, comp);
}

template <typename Key, typename Value, typename Comp>
void mergesort(Key *keys, Value *vals, int32_t count, Comp comp,
               context_t &ctx) {
  if (count <= 0) return;
  auto policy = thrust::hip::par.on(ctx.stream());
  thrust::stable_sort_by_key(policy, shim_internal::dptr(keys),
                             shim_internal::dptr(keys) + count,
                             shim_internal::dptr(vals), comp);
}

// ---------------------------------------------------------------------------
// segmented_sort / segmented_sort_indices: per-segment stable sort. segments is
// a device array of nsegs+1 offsets (row_splits style).
// ---------------------------------------------------------------------------
template <typename Key, typename Comp>
void segmented_sort(Key *keys, int32_t count, const int32_t *segments,
                    int32_t num_segments, Comp comp, context_t &ctx) {
  if (count <= 0) return;
  std::vector<int32_t> off =
      shim_internal::SegmentsToHost(ctx, segments, num_segments);
  auto policy = thrust::hip::par.on(ctx.stream());
  for (int32_t s = 0; s < num_segments; ++s) {
    int32_t begin = off[s], end = off[s + 1];
    if (end - begin > 1)
      thrust::stable_sort(policy, shim_internal::dptr(keys) + begin,
                          shim_internal::dptr(keys) + end, comp);
  }
}

template <typename Key, typename Index, typename Comp>
void segmented_sort_indices(Key *keys, Index *indices, int32_t count,
                            const int32_t *segments, int32_t num_segments,
                            Comp comp, context_t &ctx) {
  if (count <= 0) return;
  // moderngpu's segmented_sort_indices fills `indices` with the GLOBAL identity
  // permutation (0..count-1) and then stable-sorts each segment's slice
  // alongside the keys, so afterwards indices[p] is the original global index
  // of the element now at p. k2 relies on this (it does NOT pre-seed `indices`,
  // unlike mergesort which seeds with Range()): PruneRaggedAxis1 reads
  // order_map[idx01] as a global original index. Seed the identity here.
  auto policy = thrust::hip::par.on(ctx.stream());
  thrust::sequence(policy, shim_internal::dptr(indices),
                   shim_internal::dptr(indices) + count, Index(0));
  std::vector<int32_t> off =
      shim_internal::SegmentsToHost(ctx, segments, num_segments);
  for (int32_t s = 0; s < num_segments; ++s) {
    int32_t begin = off[s], end = off[s + 1];
    if (end - begin > 1)
      thrust::stable_sort_by_key(policy, shim_internal::dptr(keys) + begin,
                                 shim_internal::dptr(keys) + end,
                                 shim_internal::dptr(indices) + begin, comp);
  }
}

// ---------------------------------------------------------------------------
// load_balance_search: out_row_ids[i] = segment of element i.
// ---------------------------------------------------------------------------
inline void load_balance_search(int32_t count, const int32_t *row_splits,
                                int32_t num_segments, int32_t *out_row_ids,
                                context_t &ctx) {
  shim_internal::ComputeRowIds(ctx, count, row_splits, num_segments,
                               out_row_ids);
}

// bounds_lower/bounds_upper are enum constants used as a non-type template
// argument, matching moderngpu's `sorted_search<mgpu::bounds_lower>(...)`.
enum bounds_t { bounds_lower, bounds_upper };

// moderngpu's plus_t<T>; only used as the scan op tag for transform_scan.
template <typename T>
struct plus_t {
  __host__ __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

namespace shim_internal {
template <typename T, typename Lambda>
__global__ void MaterializeKernel(int32_t count, Lambda f, T *out) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    out[i] = f(i);
  }
}
}  // namespace shim_internal

// transform_scan<T>(f, count, output, plus, reduction_out, ctx): exclusive sum
// of f(i) for i in [0,count) into output[0..count-1], with the grand total
// written to reduction_out (which k2 passes as output+count). Implemented as in
// k2's own ExclusiveSum: materialize f(.) into count+1 elements (the last is a
// don't-care input) and run hipcub ExclusiveSum, so output[count] == total.
template <typename T, typename Lambda, typename Op>
void transform_scan(Lambda f, int32_t count, T *output, Op /*op*/,
                    T *reduction_out, context_t &ctx) {
  if (count <= 0) {
    if (count == 0)
      K2_CHECK_EQ(hipMemsetAsync(reduction_out, 0, sizeof(T), ctx.stream()),
                  hipSuccess);
    return;
  }
  // values: count+1 transformed inputs (index `count` is a readable
  // don't-care).
  shim_internal::DeviceScratch values_buf(ctx.k2_context,
                                          (count + 1) * sizeof(T));
  T *values = values_buf.as<T>();
  int32_t grid = (count + shim_internal::kBlockSize - 1) /
                 shim_internal::kBlockSize;
  shim_internal::MaterializeKernel<T, Lambda>
      <<<grid, shim_internal::kBlockSize, 0, ctx.stream()>>>(count, f, values);

  size_t temp_bytes = 0;
  K2_CHECK_EQ(hipcub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, values,
                                               output, count + 1, ctx.stream()),
              hipSuccess);
  shim_internal::DeviceScratch temp_buf(ctx.k2_context, temp_bytes);
  K2_CHECK_EQ(
      hipcub::DeviceScan::ExclusiveSum(temp_buf.data, temp_bytes, values,
                                       output, count + 1, ctx.stream()),
      hipSuccess);
  // output[count] now holds the total; reduction_out is output + count for k2's
  // K2_TRANS_EXCSUM, so it is already populated, but write it explicitly in
  // case reduction_out aliases elsewhere.
  if (reduction_out != output + count)
    K2_CHECK_EQ(hipMemcpyAsync(reduction_out, output + count, sizeof(T),
                               hipMemcpyDeviceToDevice, ctx.stream()),
                hipSuccess);
}

// sorted_search<bounds_lower>: out[i] = lower_bound(haystack, needles[i]).
template <bounds_t Bounds, typename T, typename Comp>
void sorted_search(const T *needles, int32_t num_needles, const T *haystack,
                   int32_t num_haystack, int32_t *out, Comp /*comp*/,
                   context_t &ctx) {
  if (num_needles <= 0) return;
  auto policy = thrust::hip::par.on(ctx.stream());
  auto hay_begin = shim_internal::dptr(const_cast<T *>(haystack));
  auto needle_begin = shim_internal::dptr(const_cast<T *>(needles));
  if (Bounds == bounds_lower)
    thrust::lower_bound(policy, hay_begin, hay_begin + num_haystack,
                        needle_begin, needle_begin + num_needles,
                        shim_internal::dptr(out));
  else
    thrust::upper_bound(policy, hay_begin, hay_begin + num_haystack,
                        needle_begin, needle_begin + num_needles,
                        shim_internal::dptr(out));
}

}  // namespace mgpu

#endif  // K2_CSRC_MODERNGPU_SHIM_H_
