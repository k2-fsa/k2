/**
 * @brief
 * array_ops
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <vector>

#include "k2/csrc/array_ops.h"

namespace k2 {

// See documentation in header of what this is supposed to do.
// This is similar to the template Append() defined in ops_inl.h,
// but with changes largely about adding `data_offsets`, and
// subtracting one from the dims of all but the last array.
Array1<int32_t> SpliceRowSplits(int32_t num_arrays,
                                const Array1<int32_t> **src) {
  K2_CHECK_GT(num_arrays, 0);
  ContextPtr c = src[0]->Context();

  std::vector<int32_t> row_splits_vec(num_arrays + 1);
  int32_t sum = 0, max_dim = 0;
  row_splits_vec[0] = sum;

  std::vector<const int32_t *> last_elem_ptrs_vec(num_arrays);

  for (int32_t i = 0; i < num_arrays; i++) {
    K2_CHECK_GE(src[i]->Dim(), 1);
    int32_t dim = src[i]->Dim() - (i + 1 < num_arrays ? 1 : 0);
    if (dim > max_dim) max_dim = dim;
    sum += dim;
    row_splits_vec[i + 1] = sum;
    last_elem_ptrs_vec[i] = src[i]->Data() + dim;
  }
  int32_t ans_size = sum;

  Array1<int32_t> ans(c, ans_size);
  int32_t *ans_data = ans.Data();

  Array1<const int32_t *> last_elems_ptrs(c, last_elem_ptrs_vec);
  Array1<int32_t> data_offsets(c, num_arrays);
  // note as data_offsets.Dim() == last_elem_ptrs.Dim(), so the last element of
  // last_elem_ptrs.Dim() will not be summed to data_offsets, it's OK as we
  // don't need that value since we would not drop the last element of the last
  // array.
  ExclusiveSumDeref(last_elems_ptrs, &data_offsets);
  int32_t *data_offsets_data = data_offsets.Data();

  if (c->GetDeviceType() == kCpu) {
    // a simple loop is faster, although the other branchs should still work on
    // CPU.
    for (int32_t i = 0; i < num_arrays; i++) {
      int32_t this_dim = src[i]->Dim();
      const int32_t *this_src_data = src[i]->Data();
      int32_t data_offset = data_offsets_data[i];
      for (int32_t j = 0; j < this_dim; j++) {
        ans_data[j] = this_src_data[j] + data_offset;
      }
      // notice `this_dim - 1` here, it means we will overwrite the copy of last
      // element of src[i] when copying elements in src[i+1] in the next
      // for-loop, it generates the same result with dropping the last element
      // of src[i] as last-elment-of-src[i] == src[i+1]->Data()[0] (equals 0) +
      // data_offsets_data[i+1].
      ans_data += this_dim - 1;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    Array1<int32_t> row_splits(c, row_splits_vec);
    const int32_t *row_splits_data = row_splits.Data();
    std::vector<const int32_t *> src_ptrs_vec(num_arrays);
    for (int32_t i = 0; i < num_arrays; i++) src_ptrs_vec[i] = src[i]->Data();
    Array1<const int32_t *> src_ptrs(c, src_ptrs_vec);
    const int32_t **src_ptrs_data = src_ptrs.Data();

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
        int32_t row_start = row_splits_data[i],
                row_end = row_splits_data[i + 1];
        const int32_t *src_ptr = src_ptrs_data[i];
        // not we have dropped the last element of src[i] in row_splits_data,
        // so here it will not be copied.
        if (j < row_end - row_start) {
          ans_data[row_start + j] = src_ptr[j] + data_offsets_data[i];
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
        int32_t this_num_blocks = NumBlocks(this_array_size, block_dim);
        for (int32_t j = 0; j < this_num_blocks; j++) {
          index_map.push_back((static_cast<uint64_t>(j) << 32) +
                              static_cast<uint64_t>(i));
        }
      }
      Array1<uint64_t> index_map_gpu(c, index_map);
      const uint64_t *index_map_data = index_map_gpu.Data();

      auto lambda_set_data_blocks = [=] __host__ __device__(int32_t i,
                                                            int32_t j) {
        uint64_t index = index_map_data[i];
        uint32_t orig_i = static_cast<uint32_t>(index),
                 block_index = static_cast<uint32_t>(index >> 32);
        int32_t row_start = row_splits_data[orig_i],
                row_end = row_splits_data[orig_i + 1],
                orig_j = (block_index * block_dim) + j;
        const int32_t *src_ptr = src_ptrs_data[orig_i];
        if (orig_j < row_end - row_start) {
          ans_data[row_start + orig_j] =
              src_ptr[orig_j] + data_offsets_data[orig_i];
        }
      };
      Eval2(c, index_map_gpu.Dim(), block_dim, lambda_set_data_blocks);
    }
  }
  return ans;
}

bool ValidateRowIds(const Array1<int32_t> &row_ids,
                    Array1<int32_t> *temp /*=nullptr*/) {
  ContextPtr ctx = row_ids.Context();
  const int32_t *data = row_ids.Data();
  int32_t dim = row_ids.Dim();
  if (dim == 0) return true;  // will treat this as valid
  // note `row_ids[0]` may copy memory from device to host
  if (row_ids[0] < 0) return false;

  Array1<int32_t> temp_array;
  if (temp == nullptr || temp->Dim() == 0) {
    temp_array = Array1<int32_t>(ctx, 1);
  } else {
    K2_CHECK(IsCompatible(row_ids, *temp));
    temp_array = temp->Range(0, 1);
  }
  temp = &temp_array;
  *temp = 0;

  int32_t *temp_data = temp->Data();
  auto lambda_check_row_ids = [=] __host__ __device__(int32_t i) -> void {
    if (data[i] > data[i + 1]) *temp_data = 1;  // means it's bad.
  };
  // Note: we know that dim >= 1 as we would have returned above if dim == 0.
  // This will do nothing if (dim-1) == 0 as we have checked the first element.
  Eval(ctx, dim - 1, lambda_check_row_ids);
  return (*temp)[0] == 0;
}

bool ValidateRowSplits(const Array1<int32_t> &row_splits,
                       Array1<int32_t> *temp /*=nullptr*/) {
  ContextPtr ctx = row_splits.Context();
  const int32_t *data = row_splits.Data();
  int32_t dim = row_splits.Dim();
  // must have at least one element and row_splits[0] == 0
  if (dim == 0 || row_splits[0] != 0) return false;

  Array1<int32_t> temp_array;
  if (temp == nullptr || temp->Dim() == 0) {
    temp_array = Array1<int32_t>(ctx, 1);
  } else {
    K2_CHECK(IsCompatible(row_splits, *temp));
    temp_array = temp->Range(0, 1);
  }
  temp = &temp_array;
  *temp = 0;

  int32_t *temp_data = temp->Data();
  auto lambda_check_row_splits = [=] __host__ __device__(int32_t i) -> void {
    if (data[i] > data[i + 1]) *temp_data = 1;  // means it's bad.
  };
  // Note: we know that dim >= 1 as we would have returned above if dim == 0.
  // This will do nothing if (dim-1) == 0 as we have checked the first element.
  Eval(ctx, dim - 1, lambda_check_row_splits);
  return (*temp)[0] == 0;
}

bool ValidateRowSplitsAndIds(const Array1<int32_t> &row_splits,
                             const Array1<int32_t> &row_ids,
                             Array1<int32_t> *temp /*=nullptr*/) {
  // Check if their context are compatible or not while getting
  ContextPtr ctx = GetContext(row_splits, row_ids);
  int32_t num_rows = row_splits.Dim() - 1, num_elems = row_ids.Dim();
  if (num_rows < 0 || (num_rows == 0 && num_elems > 0)) return false;
  if (row_splits[0] != 0 || row_ids[0] < 0) return false;
  if (num_elems != row_splits[num_rows]) return false;

  const int32_t *row_ids_data = row_ids.Data(),
                *row_splits_data = row_splits.Data();

  Array1<int32_t> temp_array;
  if (temp == nullptr || temp->Dim() == 0) {
    temp_array = Array1<int32_t>(ctx, 1);
  } else {
    K2_CHECK(ctx->IsCompatible(*temp->Context()));
    temp_array = temp->Range(0, 1);
  }
  temp = &temp_array;
  *temp = 0;

  int32_t *temp_data = temp_array.Data();
  auto lambda_check_row_ids = [=] __host__ __device__(int32_t i) -> void {
    // check row_splits
    bool invalid_splits =
        (i < num_rows && row_splits_data[i] > row_splits_data[i + 1]);
    // check row_ids
    bool invalid_ids =
        (i < (num_elems - 1) && row_ids_data[i] > row_ids_data[i + 1]);
    if (invalid_splits || invalid_ids) *temp_data = 1;
    // check if row_splits and row_ids agree with each other
    if (i < num_elems) {
      int32_t this_row = row_ids_data[i];
      if (this_row < 0 || this_row >= num_rows ||
          i < row_splits_data[this_row] || i >= row_splits_data[this_row + 1])
        *temp_data = 1;
    }
  };
  Eval(ctx, std::max(num_elems, num_rows), lambda_check_row_ids);
  return (*temp)[0] == 0;
}

void RowSplitsToRowIds(const Array1<int32_t> &row_splits,
                       Array1<int32_t> &row_ids) {
  ContextPtr c = GetContext(row_splits, row_ids);
  int32_t num_elems = row_ids.Dim(), num_rows = row_splits.Dim() - 1;
  K2_CHECK_GE(num_rows, 0);
  // if there are more than zero elems, there must be at least one row.
  K2_CHECK(num_elems == 0 || num_rows > 0);
  K2_CHECK_EQ(num_elems, row_splits[num_rows]);
  RowSplitsToRowIds(c, num_rows, row_splits.Data(), num_elems, row_ids.Data());
}

void RowIdsToRowSplits(const Array1<int32_t> &row_ids,
                       Array1<int32_t> &row_splits) {
  ContextPtr c = GetContext(row_splits, row_ids);
  int32_t num_elems = row_ids.Dim(), num_rows = row_splits.Dim() - 1;
  K2_CHECK_GE(num_rows, 0);
  // if there are more than zero elems, there must be at least one row.
  K2_CHECK(num_elems == 0 || num_rows > 0);
  if (num_elems > 0) K2_CHECK_GT(num_rows, row_ids[num_elems - 1]);
  RowIdsToRowSplits(c, num_elems, row_ids.Data(), false, num_rows,
                    row_splits.Data());
}

Array1<int32_t> GetCounts(const Array1<int32_t> &src, int32_t n) {
  K2_CHECK_GE(n, 0);
  ContextPtr c = src.Context();
  int32_t dim = src.Dim();
  const int32_t *src_data = src.Data();
  Array1<int32_t> ans(c, n, 0);  // init with 0
  int32_t *ans_data = ans.Data();
  if (n == 0) {
    K2_CHECK_EQ(dim, 0);
    return ans;
  }

  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    for (int32_t i = 0; i < dim; ++i) {
      ++ans_data[src_data[i]];
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    void *d_temp_storage = NULL;
    std::size_t temp_storage_bytes = 0;
    K2_CHECK_CUDA_ERROR(cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes, src_data, ans_data, n + 1, 0, n,
        dim, c->GetCudaStream()));  // The first time is to determine temporary
                                    // device storage requirements.
    void *deleter_context;
    d_temp_storage = c->Allocate(temp_storage_bytes, &deleter_context);
    K2_CHECK_CUDA_ERROR(cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes, src_data, ans_data, n + 1, 0, n,
        dim, c->GetCudaStream()));
    c->Deallocate(d_temp_storage, deleter_context);
  }
  return ans;
}

}  // namespace k2
