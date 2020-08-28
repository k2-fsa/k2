// k2/csrc/cuda/ops_inl.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
//                      Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

/* Don't include this file directly; it is included by ops.h.
   It contains implementation code. */

#ifndef IS_IN_K2_CSRC_CUDA_OPS_H_
#error "this file is supposed to be included only by ops.h"
#endif

// No header guard for this file since it will only be included
// in ops.h

#include <cassert>
#include <type_traits>

#include "cub/cub.cuh"

namespace k2 {


template <typename T>
Array1<T> Append(int32_t num_arrays, const Array1<T> **src) {
  CHECK_GT(num_arrays, 0);
  ContextPtr c = src[0]->Context();
  Array1<int32_t> row_splits(CpuContext(), num_arrays + 1);
  int32_t *row_splits_data = sizes.Data();
  int32_t sum = 0, max_dim = 0;
  row_splits_data[0] = sum;

  for (int32_t i = 0; i < num_arrays; i++) {
    int32_t dim = src[i]->Dim();
    if (dim > max_dim) max_dim = dim;
    sum += dim;
    row_splits_data[i+1] = sum;
  }
  int32_t ans_size = sum;

  Array1<T> ans(c, ans_size);
  int32 *ans_data = ans.Data();

  if (c->GetDeviceType() == kCpu) {
    // a simple loop is faster, although the other branchs should still work on
    // CPU.
    for (int32_t i = 0; i < num_arrays; i++) {
      int32_t offset = row_splits_data[i],
          this_dim = src[i]->Dim();
      const int32_t *src_data = src[i]->Data();
      for (int32_t j = 0; j < this_dim; j++) {
        ans_data[j] = src_data[j] + offset;
      }
      ans_data += this_dim;
    }
  } else {
    CHECK_EQ(c->GetDeviceType(), kGpu);
    Array1<int32_t*> src_ptrs(CpuContext(), num_arrays);
    const int32_t **src_ptrs_data = src_ptrs.Data();
    for (int32_t i = 0; i < num_arrays; i++)
      src_ptrs_data[i] = src[i]->Data();
    src_ptrs = src_ptrs.To(c);
    src_ptrs_data = src_ptrs.Data();
    row_splits = row_splits.To(c);
    const int32_t *row_splits_data = row_splits.Data();
    itn32_t avg_input_size = ans_size / num_arrays;
    if (max_dim < 2 * avg_input_size + 512) {
      // here, 2 is a heuristic factor. We're saying, "if the max length of any
      // of the source arrays is not too much larger than the average length of
      // the source arrays."  The `+ 512` is an additional safety factor, as we
      // care less about launching too many threads if the number of elements
      // being processed is small.
      // What we're saying is that the arrays' sizes are fairly balanced, so we
      // launch with a simple square kernel.
      auto lambda_set_data = [=] __host__ __device__ (int32_t i, int32_t j) {
          int32_t row_start = row_splits[i],
              row_end = row_splits[i+1];
          const int32_t *src_ptr = src_ptrs_data[i];
          if (j < row_end - row_start) {
            ans_data[row_start + j] = src_ptr[j];
          }
      };
      Eval2(c, num_arrays, max_dim, lambda_set_data);
    } else {
      int32_t block_dim = 256;
      while (block_dim * 4 < avg_input_size && block_dim < 8192)
        block_dim *= 2;

      // `index_map` will map from 'new index' to 'old index', with 0 <=
      // old_index < num_arrays... we handle each source array with multiple
      // blocks.
      //  The elements of `index_map` will be of the form:
      //    old_index + (block_of_this_array << 32).
      // where `old_index` is an index into `src` and `block_of_this_array`
      // tells us which block it is, as in 0, 1, 2, 3...
      // there won't be very many blocks, so it's not a problem to enumerate them
      // on CPU.
      std::vector<int64_t> index_map;
      index_map.reserve((2 * ans_size) / block_dim);
      for (int32_t i = 0; i < num_arrays; i++) {
        int32_t this_array_size = src[i]->Dim();
        int32_t this_num_blocks = (this_array_size + block_dim - 1) / block_dim;
        for (int32 j = 0; j < this_num_blocks; j++) {
          index_map.push_back((((uint64_t)j) << 32) + (uint64_t)i);
        }
      }
      Array1<uint64_t> index_map_gpu(index_map);
      const uint64 *index_map_data = index_map_gpu.Data();

      auto lambda_set_data_blocks = [=] __host__ __device__ (int32_t i, int32_t j) {
          uint64_t index = index_map_data[i];
          uint32_t orig_i = (uint32_t)index,
              block_index = (uint32_t)(index >> 32);
          int32_t row_start = row_splits[orig_i],
              row_end = row_splits[orig_i+1],
              orig_j = (block_index * block_size) + j;
          const int32_t *src_ptr = src_ptrs_data[orig_i];
          if (orig_j < row_end - row_start) {
            ans_data[row_start + orig_j] = src_ptr[orig_j];
          }
      };
      Eval2(c, index_map_gpu.Dim(), block_size, lambda_set_data_blocks);
    }
  }
}



}  // namespace k2
