/**
 * @brief
 * array_ops_inl
 *
 * @note
 * Don't include this file directly; it is included by array_ops.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_OPS_INL_H_
#define K2_CSRC_ARRAY_OPS_INL_H_

#ifndef IS_IN_K2_CSRC_ARRAY_OPS_H_
#error "this file is supposed to be included only by array_ops.h"
#endif

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "cub/cub.cuh"
#include "k2/csrc/macros.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "k2/csrc/utils.h"
#include "moderngpu/kernel_mergesort.hxx"

namespace k2 {
namespace internal {
// Will be used in ExclusiveSumDeref to call ExclusiveSum (which calls
// cub::DeviceScan::ExclusiveSum internally).
template <typename T>
struct PtrPtr {
  const T **data;

  explicit PtrPtr(const T **data) : data(data) {}

  // operator[] and operator+ are required by cub::DeviceScan::ExclusiveSum
  __host__ __device__ T operator[](int32_t i) const { return *(data[i]); }
  __host__ __device__ PtrPtr operator+(int32_t n) const {
    PtrPtr tmp(*this);
    tmp.data += n;
    return tmp;
  }
};

// Will be used (as both InputIterator and OutputIterator) in
// MonotonicLowerBound to call cub::DeviceScan::InclusiveScan.
template <typename T>
struct ConstReversedPtr {
  const T *data;

  // data points to the last element now
  explicit ConstReversedPtr(const T *data, int32_t size)
      : data(data + size - 1) {}

  // operator[] and operator+ are required by cub::DeviceScan::InclusiveScan
  __host__ __device__ T operator[](int32_t i) const { return data[-i]; }
  __host__ __device__ ConstReversedPtr operator+(int32_t n) const {
    ConstReversedPtr tmp(*this);
    tmp.data -= n;
    return tmp;
  }
};

template <typename T>
struct ReversedPtr {
  T *data;

  // data points to the last element now
  explicit ReversedPtr(T *data, int32_t size) : data(data + size - 1) {}

  // operator[] and operator+ are required by cub::DeviceScan::InclusiveScan
  __host__ __device__ T &operator[](int32_t i) { return data[-i]; }
  __host__ __device__ ReversedPtr operator+(int32_t n) const {
    ReversedPtr tmp(*this);
    tmp.data -= n;
    return tmp;
  }
};

// TODO(haowen): manage/load block config with some classes? then we can get
// different configuration depending on num_elements and data type.
// block size for matrix transpose.
static constexpr int32_t kTransTileDim = 32;
static constexpr int32_t kTransBlockRows = 8;

template <typename T>
__global__ void TransposeKernel(int32_t rows, int32_t cols,
                                int32_t input_elem_stride0,
                                int32_t output_elem_stride0, const T *input,
                                T *output) {
  // TODO(haowen): here we need to handle different type of T to avoid bank
  // conflicts, the size of cache now is fine for type size with 32bit (e.g.
  // int32_t or float).
  __shared__ T cache[kTransTileDim][kTransTileDim + 1];

  // input index, in a coalesced manner.
  int32_t x = threadIdx.x + blockIdx.x * kTransTileDim;
  int32_t y = threadIdx.y + blockIdx.y * kTransTileDim;

  for (int32_t i = 0; i < kTransTileDim; i += kTransBlockRows) {
    if (x < cols && (y + i) < rows) {
      cache[threadIdx.y + i][threadIdx.x] =
          input[(y + i) * input_elem_stride0 + x];
    }
  }

  __syncthreads();

  // output index, in a coalesced manner
  x = threadIdx.x + blockIdx.y * kTransTileDim;
  y = threadIdx.y + blockIdx.x * kTransTileDim;
  for (int32_t i = 0; i < kTransTileDim; i += kTransBlockRows) {
    if (x < rows && (y + i) < cols) {
      output[(y + i) * output_elem_stride0 + x] =
          cache[threadIdx.x][threadIdx.y + i];
    }
  }
}

// will be called in ExclusiveSum(Array2 &src, Array2 *dest, int32_t axis)
// to compute exclusive sum for each row
template <typename T>
void ExclusiveSumPerRow(const Array2<T> &src, Array2<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  int32_t rows = dest->Dim0();
  // note there may be dest->Dim1() == src.Dim1() + 1
  int32_t cols = dest->Dim1();
  ContextPtr &ctx = src.Context();
  ConstArray2Accessor<T> src_acc = src.Accessor();
  Array2Accessor<T> dest_acc = dest->Accessor();
  // TODO(haowen): parallelized it in case dest_minor_dim is large
  for (int32_t i = 0; i != rows; ++i) {
    ExclusiveSum(ctx, cols, src_acc.Row(i), dest_acc.Row(i));
  }
}

// called in RandUniformArray1
template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              T>::type * = nullptr>
void RandArray1Internal(ContextPtr &c, int32_t dim, T min_value, T max_value,
                        T *data, int32_t seed = 0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if (seed != 0) gen = std::mt19937(seed);
  std::uniform_real_distribution<T> dis(min_value, max_value);
  for (int32_t i = 0; i < dim; ++i) data[i] = dis(gen);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              T>::type * = nullptr>
void RandArray1Internal(ContextPtr &c, int32_t dim, T min_value, T max_value,
                        T *data, int32_t seed = 0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if (seed != 0) gen = std::mt19937(seed);
  // TODO(haowen): uniform_int_distribution does not support bool and char,
  // we may need to add some check here?
  std::uniform_int_distribution<T> dis(min_value, max_value);
  for (int32_t i = 0; i < dim; ++i) data[i] = dis(gen);
}

}  // namespace internal
}  // namespace k2

namespace std {
// vaule_type is required by cub::DeviceScan::ExclusiveSum
template <typename T>
struct iterator_traits<k2::internal::PtrPtr<T>> {
  typedef T value_type;
};
// vaule_type is required by cub::DeviceScan::InclusiveSum
template <typename T>
struct iterator_traits<k2::internal::ConstReversedPtr<T>> {
  typedef T value_type;
};
template <typename T>
struct iterator_traits<k2::internal::ReversedPtr<T>> {
  typedef T value_type;
};
}  // namespace std

namespace k2 {
template <typename T>
void Transpose(ContextPtr &c, const Array2<T> &src, Array2<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(c->IsCompatible(*src.Context()));
  K2_CHECK(c->IsCompatible(*dest->Context()));
  int32_t rows = src.Dim0();
  int32_t cols = src.Dim1();
  // TODO(haowen): limit the number of elements?
  K2_CHECK_EQ(rows, dest->Dim1());
  K2_CHECK_EQ(cols, dest->Dim0());
  if (rows == 0 || cols == 0) return;
  int32_t src_elem_stride0 = src.ElemStride0();
  int32_t dest_elem_stride0 = dest->ElemStride0();
  const T *src_data = src.Data();
  T *dest_data = dest->Data();
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    for (int32_t i = 0; i < cols; ++i) {
      for (int32_t j = 0; j < rows; ++j) {
        dest_data[i * dest_elem_stride0 + j] =
            src_data[j * src_elem_stride0 + i];
      }
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    dim3 block_size(internal::kTransTileDim, internal::kTransBlockRows, 1);
    dim3 grid_size(NumBlocks(cols, internal::kTransTileDim),
                   NumBlocks(rows, internal::kTransTileDim));
    K2_CUDA_SAFE_CALL(
        internal::
            TransposeKernel<<<grid_size, block_size, 0, c->GetCudaStream()>>>(
                rows, cols, src_elem_stride0, dest_elem_stride0, src_data,
                dest_data));
  }
}

template <typename T>
void ExclusiveSumDeref(Array1<const T *> &src, Array1<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(src, *dest));
  int32_t src_dim = src.Dim();
  int32_t dest_dim = dest->Dim();
  K2_CHECK(dest_dim == src_dim || dest_dim == src_dim + 1);
  if (dest_dim == src_dim + 1) {
    const RegionPtr &region = src.GetRegion();
    ssize_t byte_offset = static_cast<ssize_t>(src.ByteOffset());
    K2_CHECK_GE(region->num_bytes - byte_offset, dest_dim * src.ElementSize());
  }
  internal::PtrPtr<T> src_data = internal::PtrPtr<T>(src.Data());
  ExclusiveSum(src.Context(), dest_dim, src_data, dest->Data());
}

template <typename T>
void ExclusiveSum(const Array2<T> &src, Array2<T> *dest, int32_t axis) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(axis == 0 || axis == 1);
  K2_CHECK(IsCompatible(src, *dest));
  int32_t src_major_dim = src.Dim0();  // the axis will be summed
  int32_t src_minor_dim = src.Dim1();
  int32_t dest_major_dim = dest->Dim0();
  int32_t dest_minor_dim = dest->Dim1();
  if (axis == 1) {
    std::swap(src_major_dim, src_minor_dim);
    std::swap(dest_major_dim, dest_minor_dim);
  }
  K2_CHECK_EQ(dest_minor_dim, src_minor_dim);
  K2_CHECK(dest_major_dim == src_major_dim ||
           dest_major_dim == src_major_dim + 1);
  if (dest_major_dim == src_major_dim + 1) {
    const RegionPtr &region = src.GetRegion();
    ssize_t byte_offset = static_cast<ssize_t>(src.ByteOffset());
    K2_CHECK_GE(region->num_bytes - byte_offset,
                (src_major_dim * src_minor_dim + 1) * src.ElementSize());
  }

  if (axis == 1) {
    internal::ExclusiveSumPerRow(src, dest);
  } else {
    ContextPtr &ctx = src.Context();
    int32_t elem_size = src.ElementSize();
    // note here we always allocate an extra element for src_trans
    RegionPtr src_trans_region =
        NewRegion(ctx, (src_major_dim * src_minor_dim + 1) * elem_size);
    Array2<T> src_trans(src_minor_dim, src_major_dim, src_major_dim, 0,
                        src_trans_region);
    Transpose(ctx, src, &src_trans);

    RegionPtr dest_trans_region =
        NewRegion(ctx, dest_major_dim * dest_minor_dim * elem_size);
    Array2<T> dest_trans(dest_minor_dim, dest_major_dim, dest_major_dim, 0,
                         dest_trans_region);
    internal::ExclusiveSumPerRow(src_trans, &dest_trans);
    Transpose(ctx, dest_trans, dest);
  }
}

// CAUTION: if you fix bugs in this code, please also fix the same bugs in
// Splice() in array_ops.cu, since it was modified from this code.
template <typename T>
Array1<T> Append(int32_t num_arrays, const Array1<T> **src) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_arrays, 0);
  ContextPtr &c = src[0]->Context();

  std::vector<int32_t> row_splits_vec(num_arrays + 1);
  int32_t sum = 0, max_dim = 0;
  row_splits_vec[0] = sum;
  for (int32_t i = 0; i < num_arrays; ++i) {
    int32_t dim = src[i]->Dim();
    if (dim > max_dim) max_dim = dim;
    sum += dim;
    row_splits_vec[i + 1] = sum;
  }
  int32_t ans_size = sum;

  Array1<T> ans(c, ans_size);
  T *ans_data = ans.Data();

  if (c->GetDeviceType() == kCpu) {
    // a simple loop is faster, although the other branches should still work on
    // CPU.
    int32_t elem_size = src[0]->ElementSize();
    for (int32_t i = 0; i < num_arrays; ++i) {
      int32_t this_dim = src[i]->Dim();
      const T *this_src_data = src[i]->Data();
      memcpy(static_cast<void *>(ans_data),
             static_cast<const void *>(this_src_data), elem_size * this_dim);
      ans_data += this_dim;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    Array1<int32_t> row_splits(c, row_splits_vec);
    const int32_t *row_splits_data = row_splits.Data();
    std::vector<const T *> src_ptrs_vec(num_arrays);
    for (int32_t i = 0; i < num_arrays; ++i) src_ptrs_vec[i] = src[i]->Data();
    Array1<const T *> src_ptrs(c, src_ptrs_vec);
    const T **src_ptrs_data = src_ptrs.Data();
    int32_t avg_input_size = ans_size / num_arrays;
    if (max_dim < 2 * avg_input_size + 512) {
      // here, 2 is a heuristic factor. We're saying, "if the max length of any
      // of the source arrays is not too much larger than the average length of
      // the source arrays."  The `+ 512` is an additional heuristic factor, as
      // we care less about launching too many GPU threads if the number of
      // elements being processed is small. What we're saying is that the
      // arrays' sizes are fairly balanced, so we launch with a simple
      // rectangular kernel.
      K2_EVAL2(
          c, num_arrays, max_dim, lambda_set_data,
          (int32_t i, int32_t j)->void {
            int32_t row_start = row_splits_data[i],
                    row_end = row_splits_data[i + 1];
            const T *src_ptr = src_ptrs_data[i];
            if (j < row_end - row_start) {
              ans_data[row_start + j] = src_ptr[j];
            }
          });
    } else {
      int32_t block_dim = 256;
      while (block_dim * 4 < avg_input_size && block_dim < 8192) block_dim *= 2;

      // `index_map` will map from 'new index' to 'old index', with 0 <=
      // old_index < num_arrays... we handle each source array with multiple
      // blocks.
      //  The elements of `index_map` will be of the form:
      //    old_index + (block_of_this_array << 32).
      // where `old_index` is an index into `src` and `block_of_this_array`
      // tells us which block it is, as in 0, 1, 2, 3...
      // there won't be very many blocks, so it's not a problem to enumerate
      // them on CPU.
      std::vector<uint64_t> index_map;
      index_map.reserve((2 * ans_size) / block_dim);
      for (int32_t i = 0; i < num_arrays; ++i) {
        int32_t this_array_size = src[i]->Dim();
        int32_t this_num_blocks = NumBlocks(this_array_size, block_dim);
        for (int32_t j = 0; j < this_num_blocks; ++j) {
          index_map.push_back((static_cast<uint64_t>(j) << 32) +
                              static_cast<uint64_t>(i));
        }
      }
      Array1<uint64_t> index_map_gpu(c, index_map);
      const uint64_t *index_map_data = index_map_gpu.Data();

      K2_EVAL2(
          c, index_map_gpu.Dim(), block_dim, lambda_set_data_blocks,
          (int32_t i, int32_t j) {
            uint64_t index = index_map_data[i];
            uint32_t orig_i = static_cast<uint32_t>(index),
                     block_index = static_cast<uint32_t>(index >> 32);
            int32_t row_start = row_splits_data[orig_i],
                    row_end = row_splits_data[orig_i + 1],
                    orig_j = (block_index * block_dim) + j;
            const T *src_ptr = src_ptrs_data[orig_i];
            if (orig_j < row_end - row_start) {
              ans_data[row_start + orig_j] = src_ptr[orig_j];
            }
          });
    }
  }
  return ans;
}

template <typename T>
Array1<T> Append(int32_t src_size, const Array1<T> *src) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(src_size, 0);
  std::vector<const Array1<T> *> srcs(src_size);
  for (int32_t i = 0; i != src_size; ++i) srcs[i] = src + i;
  const Array1<T> **srcs_ptr = srcs.data();
  return Append(src_size, srcs_ptr);
}

template <typename T, typename Op>
void ApplyOpOnArray1(Array1<T> &src, T default_value, Array1<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(src, *dest));
  K2_CHECK_EQ(dest->Dim(), 1);

  ContextPtr &c = src.Context();
  T *src_data = src.Data();
  T *dest_data = dest->Data();
  int32_t size = src.Dim();
  Op op;

  if (c->GetDeviceType() == kCpu) {
    T val = default_value;
    for (int32_t i = 0; i != size; ++i) {
      val = op(src_data[i], val);
    }
    dest_data[0] = val;
  } else {
    K2_CHECK(c->GetDeviceType() == kCuda);

    size_t temp_storage_bytes = 0;
    // the first time is to determine temporary device storage requirements
    K2_CUDA_SAFE_CALL(cub::DeviceReduce::Reduce(
        nullptr, temp_storage_bytes, src_data, dest_data, size, op,
        default_value, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceReduce::Reduce(
        d_temp_storage.Data(), temp_storage_bytes, src_data, dest_data, size,
        op, default_value, c->GetCudaStream()));
  }
}

template <typename T, typename BinaryOp>
void ApplyBinaryOpOnArray1(Array1<T> &src1, Array1<T> &src2, Array1<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(dest, nullptr);

  int32_t dim = src1.Dim();
  K2_CHECK_EQ(dim, src2.Dim());
  K2_CHECK_EQ(dim, dest->Dim());
  ContextPtr c = GetContext(src1, src2, *dest);

  const T *src1_data = src1.Data();
  const T *src2_data = src2.Data();
  T *dest_data = dest->Data();

  BinaryOp op;

  K2_EVAL(
      c, dim, lambda_set_values,
      (int32_t i)->void { dest_data[i] = op(src1_data[i], src2_data[i]); });
}

template <typename T>
Array1<T> RandUniformArray1(ContextPtr c, int32_t dim, T min_value, T max_value,
                            int32_t seed /*= 0*/) {
  NVTX_RANGE(K2_FUNC);
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Only support floating-point and integral type");
  Array1<T> temp(GetCpuContext(), dim);
  T *data = temp.Data();
  K2_CHECK_GE(max_value, min_value);

  std::random_device rd;
  std::mt19937 gen(rd());
  if (max_value == min_value) {
    for (int32_t i = 0; i < dim; ++i) data[i] = min_value;
  } else {
    internal::RandArray1Internal<T>(c, dim, min_value, max_value, data, seed);
  }
  return temp.To(c);
}

template <typename T>
Array2<T> RandUniformArray2(ContextPtr c, int32_t dim0, int32_t dim1,
                            T min_value, T max_value) {
  NVTX_RANGE(K2_FUNC);
  int32_t dim1_extra = RandInt(0, 2),  // make it randomly not contiguous.
      new_dim1 = dim1 + dim1_extra;
  Array1<T> array1temp =
      RandUniformArray1<T>(c, dim0 * new_dim1, min_value, max_value);
  Array2<T> array2temp(array1temp, dim0, new_dim1);

  int32_t offset = RandInt(0, dim1_extra);
  return array2temp.ColArange(offset, offset + dim1);
}

template <typename T>
Array1<T> Range(ContextPtr c, int32_t dim, T first_value, T inc /*=1*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(dim, 0);
  Array1<T> ans = Array1<T>(c, dim);
  T *ans_data = ans.Data();

  K2_EVAL(
      c, dim, lambda_set_values,
      (int32_t i)->void { ans_data[i] = first_value + i * inc; });
  return ans;
}

template <typename T>
Array1<T> Arange(ContextPtr c, T begin, T end, T inc) {
  return Range<T>(c, (end + inc - 1 - begin) / inc, begin, inc);
}

template <typename T>
Array2<T> ToContiguous(const Array2<T> &src) {
  NVTX_RANGE(K2_FUNC);
  int32_t dim0 = src.Dim0();
  int32_t dim1 = src.Dim1();
  int32_t elem_stride0 = src.ElemStride0();
  if (dim1 == elem_stride0) return src;
  Array2<T> ans(src.Context(), src.Dim0(), src.Dim1());
  T *out = ans.Data();
  const T *in = src.Data();
  K2_EVAL2(
      src.Context(), dim0, dim1, lambda_copy_elems,
      (int32_t i, int32_t j)->void {
        out[i * dim1 + j] = in[i * elem_stride0 + j];
      });
  return ans;
}

template <typename T>
bool Equal(const Array1<T> &a, const Array1<T> &b) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(a.Dim(), b.Dim());
  ContextPtr c = GetContext(a, b);
  const T *a_data = a.Data(), *b_data = b.Data();
  if (c->GetDeviceType() == kCpu) {
    return memcmp(reinterpret_cast<const void *>(a_data),
                  reinterpret_cast<const void *>(b_data),
                  sizeof(T) * a.Dim()) == 0;
  } else {
    Array1<int32_t> is_same(c, 1, 1);
    int32_t *is_same_data = is_same.Data();
    auto lambda_test = [=] __device__(int32_t i) -> void {
      if (a_data[i] != b_data[i]) *is_same_data = 0;
    };
    EvalDevice(c, a.Dim(), lambda_test);
    return is_same[0];
  }
}

template <typename T>
bool Equal(const Array2<T> &a, const Array2<T> &b) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(a.Dim0(), b.Dim0());
  K2_CHECK_EQ(a.Dim1(), b.Dim1());
  ContextPtr c = GetContext(a, b);
  const T *a_data = a.Data(), *b_data = b.Data();

  if (a.IsContiguous() && b.IsContiguous()) {
    // use simpler code which might be faster.
    int32_t dim = a.Dim0() * a.Dim1();
    Array1<T> a1(dim, a.GetRegion(), a.ByteOffset()),
        b1(dim, b.GetRegion(), b.ByteOffset());
    return Equal(a1, b1);
  }

  auto a_acc = a.Accessor(), b_acc = b.Accessor();

  if (c->GetDeviceType() == kCpu) {
    size_t row_bytes = a.Dim1() * sizeof(T);
    for (int32_t row = 0; row < a.Dim0(); row++)
      if (memcmp(reinterpret_cast<const void *>(a_acc.Row(row)),
                 reinterpret_cast<const void *>(b_acc.Row(row)), row_bytes))
        return false;
    return true;
  } else {
    Array1<int32_t> is_same(c, 1, 1);
    int32_t *is_same_data = is_same.Data();
    auto lambda_test = [=] __device__(int32_t i, int32_t j) -> void {
      if (a_acc(i, j) != b_acc(i, j)) *is_same_data = 0;
    };
    Eval2Device(c, a.Dim0(), a.Dim1(), lambda_test);
    return is_same[0];
  }
}

template <typename T>
bool IsMonotonic(const Array1<T> &a) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = a.Context();
  int32_t dim = a.Dim();
  const T *data = a.Data();
  if (c->GetDeviceType() == kCpu) {
    for (int i = 0; i + 1 < dim; i++)
      if (data[i + 1] < data[i]) return false;
    return true;
  } else {
    Array1<int32_t> is_monotonic(c, 1, 1);
    int32_t *is_monotonic_data = is_monotonic.Data();
    auto lambda_test = [=] __device__(int32_t i) -> void {
      if (data[i + 1] < data[i]) *is_monotonic_data = 0;
    };
    EvalDevice(c, dim - 1, lambda_test);
    return is_monotonic[0];
  }
}

template <typename T>
bool IsMonotonicDecreasing(const Array1<T> &a) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = a.Context();
  int32_t dim = a.Dim();
  const T *data = a.Data();
  if (c->GetDeviceType() == kCpu) {
    for (int i = 0; i + 1 < dim; i++)
      if (data[i + 1] > data[i]) return false;
    return true;
  } else {
    Array1<int32_t> is_monotonic(c, 1, 1);
    int32_t *is_monotonic_data = is_monotonic.Data();
    auto lambda_test = [=] __device__(int32_t i) -> void {
      if (data[i + 1] > data[i]) *is_monotonic_data = 0;
    };
    EvalDevice(c, dim - 1, lambda_test);
    return is_monotonic[0];
  }
}

template <typename S, typename T>
void MonotonicLowerBound(const Array1<S> &src, Array1<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_STATIC_ASSERT((std::is_convertible<S, T>::value));
  K2_CHECK(IsCompatible(src, *dest));
  int32_t dim = src.Dim();
  K2_CHECK_EQ(dest->Dim(), dim);

  ContextPtr &c = src.Context();
  const S *src_data = src.Data();
  T *dest_data = dest->Data();

  if (c->GetDeviceType() == kCpu) {
    S min_value = std::numeric_limits<S>::max();
    for (int32_t i = dim - 1; i >= 0; --i) {
      min_value = std::min(src_data[i], min_value);
      // we suppose it's safe to assign a value with type `S`
      // to a value with type `T`
      dest_data[i] = min_value;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    MinOp<S> min_op;
    internal::ConstReversedPtr<S> src_ptr =
        internal::ConstReversedPtr<S>(src_data, dim);
    internal::ReversedPtr<T> dest_ptr =
        internal::ReversedPtr<T>(dest_data, dim);
    // The first time is to determine temporary device storage requirements.
    std::size_t temp_storage_bytes = 0;
    K2_CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveScan(
        nullptr, temp_storage_bytes, src_ptr, dest_ptr, min_op, dim,
        c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveScan(
        d_temp_storage.Data(), temp_storage_bytes, src_ptr, dest_ptr, min_op,
        dim, c->GetCudaStream()));
  }
}

template <typename S, typename T>
void MonotonicDecreasingUpperBound(const Array1<S> &src, Array1<T> *dest) {
  NVTX_RANGE(K2_FUNC);
  K2_STATIC_ASSERT((std::is_convertible<S, T>::value));
  K2_CHECK(IsCompatible(src, *dest));
  int32_t dim = src.Dim();
  K2_CHECK_EQ(dest->Dim(), dim);

  ContextPtr &c = src.Context();
  const S *src_data = src.Data();
  T *dest_data = dest->Data();

  if (c->GetDeviceType() == kCpu) {
    S max_value = std::numeric_limits<S>::min();
    for (int32_t i = dim - 1; i >= 0; --i) {
      max_value = std::max(src_data[i], max_value);
      // we suppose it's safe to assign a value with type `S`
      // to a value with type `T`
      dest_data[i] = max_value;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    MaxOp<S> max_op;
    internal::ConstReversedPtr<S> src_ptr =
        internal::ConstReversedPtr<S>(src_data, dim);
    internal::ReversedPtr<T> dest_ptr =
        internal::ReversedPtr<T>(dest_data, dim);
    // The first time is to determine temporary device storage requirements.
    std::size_t temp_storage_bytes = 0;
    K2_CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveScan(
        nullptr, temp_storage_bytes, src_ptr, dest_ptr, max_op, dim,
        c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveScan(
        d_temp_storage.Data(), temp_storage_bytes, src_ptr, dest_ptr, max_op,
        dim, c->GetCudaStream()));
  }
}

template <typename T>
Array1<T> Plus(const Array1<T> &src, T t) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  int32_t dim = src.Dim();
  Array1<T> ans(c, dim);
  const T *data = src.Data();
  T *ans_data = ans.Data();
  K2_EVAL(
      c, dim, lambda_add, (int32_t i)->void { ans_data[i] = data[i] + t; });
  return ans;
}

template <typename T>
Array1<T> Index(const Array1<T> &src, const Array1<int32_t> &indexes,
                bool allow_minus_one) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  K2_CHECK(c->IsCompatible(*indexes.Context()));
  int32_t ans_dim = indexes.Dim();
  Array1<T> ans(c, ans_dim);
  T *ans_data = ans.Data();
  const T *src_data = src.Data();
  const int32_t *index_data = indexes.Data();
  DeviceType d = c->GetDeviceType();
  if (allow_minus_one) {
    if (d == kCpu) {
#pragma unroll(4)
      for (int32_t i = 0; i < ans_dim; i++) {
        int32_t index = index_data[i];
        T value = (index < 0 ? T(0) : src_data[index]);
        ans_data[i] = value;
      }
    } else {
      auto lambda_set_values = [=] __device__(int32_t i) -> void {
        int32_t index = index_data[i];
        T value = (index < 0 ? T(0) : src_data[index]);
        ans_data[i] = value;
      };
      EvalDevice(c, ans_dim, lambda_set_values);
    }
  } else {
    if (d == kCpu) {
#pragma unroll(4)
      for (int32_t i = 0; i < ans_dim; i++)
        ans_data[i] = src_data[index_data[i]];
    } else {
      auto lambda_set_values = [=] __device__(int32_t i) -> void {
        ans_data[i] = src_data[index_data[i]];
      };
      EvalDevice(c, ans_dim, lambda_set_values);
    }
  }
  return ans;
}

template <typename T>
Array2<T> IndexRows(const Array2<T> &src, const Array1<int32_t> &indexes,
                    bool allow_minus_one) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  K2_CHECK(c->IsCompatible(*indexes.Context()));
  int32_t ans_dim0 = indexes.Dim(), dim1 = src.Dim1();
  Array2<T> ans(c, ans_dim0, dim1);
  const int32_t *index_data = indexes.Data();
  auto ans_acc = ans.Accessor();
  auto src_acc = src.Accessor();
  DeviceType d = c->GetDeviceType();
  if (allow_minus_one) {
    if (d == kCpu) {
      for (int32_t i = 0; i < ans_dim0; i++) {
        int32_t index = index_data[i];
        if (index < 0) {
#pragma unroll(4)
          for (int32_t j = 0; j < dim1; j++) ans_acc(i, j) = T(0);
        } else {
#pragma unroll(4)
          for (int32_t j = 0; j < dim1; j++) ans_acc(i, j) = src_acc(index, j);
        }
      }
    } else {
      auto lambda_set_values = [=] __device__(int32_t i, int32_t j) -> void {
        int32_t index = index_data[i];
        ans_acc(i, j) = (index < 0 ? T(0) : src_acc(index, j));
      };
      Eval2Device(c, ans_dim0, dim1, lambda_set_values);
    }
  } else {
    if (d == kCpu) {
      for (int32_t i = 0; i < ans_dim0; i++) {
        int32_t index = index_data[i];
#pragma unroll(4)
        for (int32_t j = 0; j < dim1; j++) ans_acc(i, j) = src_acc(index, j);
      }
    } else {
      auto lambda_set_values = [=] __device__(int32_t i, int32_t j) -> void {
        int32_t index = index_data[i];
        ans_acc(i, j) = src_acc(index, j);
      };
      Eval2Device(c, ans_dim0, dim1, lambda_set_values);
    }
  }
  return ans;
}

template <typename T, typename Compare>
static void SortCpu(Array1<T> *array, Array1<int32_t> *index_map) {
  Compare comp;
  if (index_map != nullptr) {
    Array1<int32_t> tmp_index_map = Range(array->Context(), array->Dim(), 0);
    const T *array_data = array->Data();
    std::sort(tmp_index_map.Data(), tmp_index_map.Data() + tmp_index_map.Dim(),
              [array_data, comp](int32_t i, int32_t j) {
                return comp(array_data[i], array_data[j]);
              });
    *index_map = std::move(tmp_index_map);
  }

  std::sort(array->Data(), array->Data() + array->Dim(), comp);
}

template <typename T, typename Compare /*= LessThan<T>*/>
void Sort(Array1<T> *array, Array1<int32_t> *index_map /*= nullptr*/) {
  if (!array->IsValid()) return;

  ContextPtr &context = array->Context();
  if (context->GetDeviceType() == kCpu)
    return SortCpu<T, Compare>(array, index_map);

  K2_DCHECK_EQ(context->GetDeviceType(), kCuda);

  mgpu::context_t *mgpu_context = GetModernGpuAllocator(context);

  if (index_map != nullptr) {
    *index_map = Range(context, array->Dim(), 0);
    mgpu::mergesort(array->Data(), index_map->Data(), array->Dim(), Compare(),
                    *mgpu_context);
  } else {
    mgpu::mergesort(array->Data(), array->Dim(), Compare(), *mgpu_context);
  }
}

template <typename T>
void Assign(Array2<T> &src, Array2<T> *dest) {
  K2_CHECK_EQ(src.Dim0(), dest->Dim0());
  K2_CHECK_EQ(src.Dim1(), dest->Dim1());
  int32_t dim0 = src.Dim0(), dim1 = src.Dim1(), src_stride = src.ElemStride0(),
          dest_stride = dest->ElemStride0();

  if (src_stride == dim1 && dest_stride == dim1) {
    size_t num_bytes = dim0 * src.ElementSize() * dim1;
    src.Context()->CopyDataTo(num_bytes, src.Data(), dest->Context(),
                              dest->Data());
  } else {
    // this branch does not support cross-device copy.
    ContextPtr c = GetContext(src, *dest);
    T *dest_data = dest->Data();
    const T *src_data = src.Data();
    if (c->GetDeviceType() == kCpu) {
      size_t row_length_bytes = src.ElementSize() * dim1;
      for (int32_t r = 0; r < dim0;
           r++, dest_data += dest_stride, src_data += src_stride) {
        memcpy(static_cast<void *>(dest_data),
               static_cast<const void *>(src_data), row_length_bytes);
      }
    } else {
      auto lambda_copy_data = [=] __device__(int32_t i, int32_t j) {
        dest_data[i * dest_stride + j] = src_data[i * src_stride + j];
      };
      Eval2Device(c, dim0, dim1, lambda_copy_data);
    }
  }
}


template <typename S, typename T>
void Assign(Array1<S> &src, Array1<T> *dest) {
  K2_CHECK_EQ(src.Dim(), dest->Dim());
  int32_t dim = src.Dim();
  if (std::is_same<S,T>::value) {
    size_t num_bytes = dim * sizeof(S);
    src.Context()->CopyDataTo(num_bytes, src.Data(), dest->Context(),
                              dest->Data());
  } else {
    if (!src.Context()->IsCompatible(*dest->Context())) {
      Array1<S> src_new = src.To(dest->Context());
      Assign(src_new, dest);
    }
    const S *src_data = src.Data();
    T *dest_data = dest->Data();
    K2_EVAL(src.Context(), dim, lambda_copy_data, (int32_t i) -> void {
        dest_data[i] = src_data[i];
      });
  }
}


template <typename T>
Array1<T> MergeWithMap(const Array1<uint32_t> &merge_map, int32_t num_srcs,
                       const Array1<T> **src) {
  NVTX_RANGE(K2_FUNC);
  int32_t dim = merge_map.Dim();
  ContextPtr &c = merge_map.Context();
  std::vector<const T *> src_ptrs_vec(num_srcs);
  int32_t src_tot_dim = 0;
  for (int32_t i = 0; i < num_srcs; ++i) {
    K2_CHECK(c->IsCompatible(*src[i]->Context()));
    src_tot_dim += src[i]->Dim();
    src_ptrs_vec[i] = src[i]->Data();
  }
  K2_CHECK_EQ(src_tot_dim, dim);
  Array1<const T *> src_ptrs(c, src_ptrs_vec);
  Array1<T> ans(c, dim);
  const uint32_t *merge_map_data = merge_map.Data();
  T *ans_data = ans.Data();
  const T **src_ptrs_data = src_ptrs.Data();
  K2_EVAL(
      c, dim, lambda_merge_data, (int32_t i)->void {
        uint32_t m = merge_map_data[i], src_idx = m % (uint32_t)num_srcs,
                 src_pos = m / (uint32_t)num_srcs;
        ans_data[i] = src_ptrs_data[src_idx][src_pos];
      });
  return ans;
}

}  // namespace k2

#endif  // K2_CSRC_ARRAY_OPS_INL_H_
