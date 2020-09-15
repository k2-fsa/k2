/**
 * @brief
 * array_ops_inl
 *
 * @note
 * Don't include this file directly; it is included by array_ops.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Fangjun Kuang (csukuangfj@gmail.com)
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_OPS_INL_H_
#define K2_CSRC_ARRAY_OPS_INL_H_

#ifndef IS_IN_K2_CSRC_ARRAY_OPS_H_
#error "this file is supposed to be included only by array_ops.h"
#endif

#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <type_traits>
#include <vector>

namespace k2 {

// CAUTION: if you fix bugs in this code, please also fix the same bugs in
// Splice() in array_ops.cu, since it was modified from this code.
template <typename T>
Array1<T> Append(int32_t num_arrays, const Array1<T> **src) {
  K2_CHECK_GT(num_arrays, 0);
  ContextPtr c = src[0]->Context();

  std::vector<int32_t> row_splits_vec(num_arrays + 1);
  int32_t sum = 0, max_dim = 0;
  row_splits_vec[0] = sum;
  for (int32_t i = 0; i < num_arrays; i++) {
    int32_t dim = src[i]->Dim();
    if (dim > max_dim) max_dim = dim;
    sum += dim;
    row_splits_vec[i + 1] = sum;
  }
  int32_t ans_size = sum;

  Array1<T> ans(c, ans_size);
  T *ans_data = ans.Data();

  if (c->GetDeviceType() == kCpu) {
    // a simple loop is faster, although the other branchs should still work on
    // CPU.
    for (int32_t i = 0; i < num_arrays; i++) {
      int32_t offset = row_splits_vec[i], this_dim = src[i]->Dim();
      const int32_t *this_src_data = src[i]->Data();
      memcpy(static_cast<void *>(ans_data),
             static_cast<const void *>(this_src_data), sizeof(T) * this_dim);
      ans_data += this_dim;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    Array1<int32_t> row_splits(c, row_splits_vec);
    const int32_t *row_splits_data = row_splits.Data();
    std::vector<T *> src_ptrs_vec(num_arrays);
    for (int32_t i = 0; i < num_arrays; i++) src_ptrs_vec[i] = src[i]->Data();
    Array1<T> src_ptrs(c, src_ptrs_vec);
    auto src_ptrs_data = src_ptrs.Data();
    int32_t avg_input_size = ans_size / num_arrays;
    if (max_dim < 2 * avg_input_size + 512) {
      // here, 2 is a heuristic factor. We're saying, "if the max length of any
      // of the source arrays is not too much larger than the average length of
      // the source arrays."  The `+ 512` is an additional heuristic factor, as
      // we care less about launching too many GPU threads if the number of
      // elements being processed is small. What we're saying is that the
      // arrays' sizes are fairly balanced, so we launch with a simple
      // rectangular kernel.
      auto lambda_set_data = [=] __host__ __device__(int32_t i,
                                                     int32_t j) -> void {
        // TODO(haowen): change to use operator[]
        int32_t row_start = row_splits.Data()[i],
                row_end = row_splits.Data()[i + 1];
        const T *src_ptr = src_ptrs_data[i];
        if (j < row_end - row_start) {
          ans_data[row_start + j] = src_ptr[j];
        }
      };
      Eval2(c, num_arrays, max_dim, lambda_set_data);
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
      for (int32_t i = 0; i < num_arrays; i++) {
        int32_t this_array_size = src[i]->Dim();
        int32_t this_num_blocks = (this_array_size + block_dim - 1) / block_dim;
        for (int32_t j = 0; j < this_num_blocks; j++) {
          index_map.push_back((((uint64_t)j) << 32) + (uint64_t)i);
        }
      }
      Array1<uint64_t> index_map_gpu(c, index_map);
      const uint64_t *index_map_data = index_map_gpu.Data();

      auto lambda_set_data_blocks = [=] __host__ __device__(int32_t i,
                                                            int32_t j) {
        uint64_t index = index_map_data[i];
        uint32_t orig_i = (uint32_t)index,
                 block_index = (uint32_t)(index >> 32);
        int32_t row_start = row_splits.Data()[orig_i],
                row_end = row_splits.Data()[orig_i + 1],
                orig_j = (block_index * block_dim) + j;
        const T *src_ptr = src_ptrs_data[orig_i];
        if (orig_j < row_end - row_start) {
          ans_data[row_start + orig_j] = src_ptr[orig_j];
        }
      };
      Eval2(c, index_map_gpu.Dim(), block_dim, lambda_set_data_blocks);
    }
  }
}

template <typename T>
void ExclusiveSumDeref(Array1<T *> &src, Array1<T> *dest) {
  K2_CHECK(src.Context()->IsCompatible(*dest->Context()));
  struct PtrPtr {
    T **data;
    __host__ __device__ T operator[](int32_t i) { return *(data[i]); }
    explicit PtrPtr(T **data) : data(data) {}
    PtrPtr(const PtrPtr &src) = default;
  };

  int32_t src_dim = src.Dim(), dest_dim = dest->Dim();
  assert(dest_dim == src_dim || dest_dim == src_dim + 1);

  PtrPtr src_data = PtrPtr(src.Data());
  T *dest_data = dest->Data();

  // use the ExclusiveSum() template for pointer-like objects that is declared
  // in utils.h
  ExclusiveSum(src.Context(), dest_dim, src_data, dest_data);
}

template <typename T>
void MaxPerSublist(Ragged<T> &src, T default_value, Array1<T> *max_values) {
  K2_CHECK_EQ(src.NumAxes(), 2);
  K2_CHECK_EQ(src.Dim0(), max_values->Dim());
  K2_CHECK(IsCompatible(src, *max_values));

  ContextPtr c = src.Context();

  const int32_t *row_splits = src.RowSplits(1);
  int32_t num_rows = src.Dim0();
  const T *values_data = src.values.Data();
  T *output_data = max_values->Data();

  if (c->GetDeviceType() == kCpu) {
    int32_t j = row_splits[0];
    for (int32_t i = 0; i < num_rows; i++) {
      T max_val = default_value;
      int32_t row_end = row_splits[i + 1];
      for (; j < row_end; j++) {
        T elem = values_data[j];
        max_val = (elem > max_val ? elem : max_val);
      }
      output_data[i] = max_val;
    }
  } else {
    K2_CHECK(c->GetDeviceType() == kCuda);

    // This code is based on the example here:
    // https://nvlabs.github.io/cub/structcub_1_1_device_segmented_reduce.html

    struct MaxOp {
      __host__ __device__ T operator()(T a, T b) { return (a > b ? a : b); }
      MaxOp(const MaxOp &src) = default;
    } max_op;

    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // The first time it just sets `temp_storage_bytes`.
    // TODO(haowen): uncomment below lines
    /*
    K2_CUDA_API_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes, values_data, output_data, num_rows,
        row_splits, row_splits + 1, max_op, default_value));
    K2_CUDA_API_SAFE_CALL(cudaMalloc(&temp_storage, temp_storage_bytes));
    K2_CUDA_API_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes, values_data, output_data, num_rows,
        row_splits, row_splits + 1, max_op, default_value));
    */
  }
}

template <typename T>
Array1<T> RandUniformArray1(ContextPtr &c, int32_t dim, T min_value,
                            T max_value) {
  Array1<T> temp(GetCpuContext(), dim);
  T *data = temp.Data();
  K2_CHECK(max_value >= min_value);
  if (max_value == min_value) {
    for (int32_t i = 0; i < dim; i++) data[i] = 0;
  } else if (std::is_floating_point<T>::value ||
             std::abs(min_value) > RAND_MAX || std::abs(max_value) > RAND_MAX) {
    for (int32_t i = 0; i < dim; i++)
      data[i] =
          min_value + (rand() * (max_value - min_value) / RAND_MAX);  // NOLINT
  } else {
    for (int32_t i = 0; i < dim; i++)
      data[i] = min_value + (rand() % (max_value + 1 - min_value));  // NOLINT
  }
  return temp.To(c);
}

template <typename T>
Array1<T> Range(ContextPtr &c, int32_t dim, T first_value, T inc /*=1*/) {
  K2_CHECK(dim >= 0);
  DeviceType d = c->GetDeviceType();
  Array1<T> ans = Array1<T>(c, dim);
  T *ans_data = ans.Data();
  if (d == kCpu) {
    for (int32_t i = 0; i < dim; i++) ans_data[i] = first_value + i * inc;
  } else {
    auto lambda_set_values = [=] __host__ __device__(int32_t i) -> void {
      ans_data[i] = first_value + i * inc;
    };
    Eval(c, dim, lambda_set_values);
  }
  return ans;
}

template <typename T>
Array2<T> ToContiguous(const Array2<T> &src) {
  int32_t dim0 = src.Dim0();
  int32_t dim1 = src.Dim1();
  int32_t elem_stride0 = src.ElemStride0();
  if (dim1 == elem_stride0) return src;
  Array2<T> ans(src.Context(), src.Dim0(), src.Dim1());
  T *out = ans.Data();
  const T *in = src.Data();
  auto lambda_copy_elems = [=] __host__ __device__(int32_t i,
                                                   int32_t j) -> void {
    out[i * dim1 + j] = in[i * elem_stride0 + j];
  };
  Eval2(src.Context(), dim0, dim1, lambda_copy_elems);
  return ans;
}

}  // namespace k2

#endif  // K2_CSRC_ARRAY_OPS_INL_H_
