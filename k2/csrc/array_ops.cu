/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
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

#include <algorithm>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

// See documentation in header of what this is supposed to do.
// This is similar to the template Cat() defined in ops_inl.h,
// but with changes largely about adding `data_offsets`, and
// subtracting one from the dims of all but the last array.
Array1<int32_t> SpliceRowSplits(int32_t num_arrays,
                                const Array1<int32_t> **src) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_arrays, 0);
  ContextPtr &c = src[0]->Context();

  // row_splits_vec is the exclusive-sum of the modified dimensions of
  // the arrays in `src`.  `Modified` means: is subtracted from the dims
  // of all but the last array.
  std::vector<int32_t> row_splits_vec(num_arrays + 1);
  int32_t sum = 0;
  row_splits_vec[0] = sum;

  // `last_elem_ptrs_vec` contains, for each of the arrays in `num_array`, a
  // pointer to the last element in that array.
  std::vector<const int32_t *> last_elem_ptrs_vec(num_arrays);

  for (int32_t i = 0; i < num_arrays; i++) {
    K2_CHECK_GE(src[i]->Dim(), 1);
    int32_t dim = src[i]->Dim() - (i + 1 < num_arrays ? 1 : 0);
    sum += dim;
    row_splits_vec[i + 1] = sum;
    last_elem_ptrs_vec[i] = src[i]->Data() + src[i]->Dim() - 1;
  }
  int32_t ans_size = sum;

  Array1<int32_t> ans(c, ans_size);
  if (ans_size == 0) return ans;
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
    // a simple loop is faster, although the other branches should still work on
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
    std::vector<const int32_t *> src_ptrs_vec(num_arrays);
    for (int32_t i = 0; i < num_arrays; i++) src_ptrs_vec[i] = src[i]->Data();
    Array1<const int32_t *> src_ptrs(c, src_ptrs_vec);
    const int32_t **src_ptrs_data = src_ptrs.Data();

    mgpu::context_t *mgpu_context = GetModernGpuAllocator(c);
    auto lambda_set_ans = [=] __device__(int32_t index, int32_t seg,
                                         int32_t rank) {
      ans_data[index] = src_ptrs_data[seg][rank] + data_offsets_data[seg];
    };
    K2_CUDA_SAFE_CALL(mgpu::transform_lbs(lambda_set_ans, ans_size,
                                          row_splits.Data(),
                                          row_splits.Dim() - 1, *mgpu_context));
  }
  return ans;
}

Array1<int32_t> CatWithOffsets(const Array1<int32_t> &offsets,
                               const Array1<int32_t> **src) {
  NVTX_RANGE(K2_FUNC);

  int32_t num_arrays = offsets.Dim();
  ContextPtr c = offsets.Context();

  std::vector<int32_t> row_splits_vec(num_arrays + 1);
  int32_t sum = 0;
  row_splits_vec[0] = sum;
  for (int32_t i = 0; i < num_arrays; ++i) {
    int32_t dim = src[i]->Dim();
    sum += dim;
    row_splits_vec[i + 1] = sum;
  }
  int32_t ans_size = sum;

  Array1<int32_t> ans(c, ans_size);
  if (ans_size == 0) return ans;

  int32_t *ans_data = ans.Data();
  const int32_t *offsets_data = offsets.Data();
  if (c->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i != num_arrays; ++i) {
      int32_t this_dim = src[i]->Dim();
      const int32_t *this_src_data = src[i]->Data();
      int32_t offset = offsets_data[i];
      for (int32_t j = 0; j != this_dim; ++j) {
        ans_data[j] = this_src_data[j] + offset;
      }
      ans_data += this_dim;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    Array1<int32_t> row_splits(c, row_splits_vec);
    std::vector<const int32_t *> src_ptrs_vec(num_arrays);
    for (int32_t i = 0; i < num_arrays; ++i) src_ptrs_vec[i] = src[i]->Data();
    Array1<const int32_t *> src_ptrs(c, src_ptrs_vec);
    const int32_t **src_ptrs_data = src_ptrs.Data();

    mgpu::context_t *mgpu_context = GetModernGpuAllocator(c);
    // `index` is idx01, `seg` is idx0, `rank` is idx1, `value_offsets` is just
    // a cache for `offsets_data`.
    auto lambda_set_ans = [=] __device__(int32_t index, int32_t seg,
                                         int32_t rank,
                                         mgpu::tuple<int32_t> value_offsets) {
      ans_data[index] = src_ptrs_data[seg][rank] + mgpu::get<0>(value_offsets);
    };
    K2_CUDA_SAFE_CALL(mgpu::transform_lbs(
        lambda_set_ans, ans_size, row_splits.Data(), row_splits.Dim() - 1,
        mgpu::make_tuple(offsets_data), *mgpu_context));
  }
  return ans;
}

bool ValidateRowIds(const Array1<int32_t> &row_ids,
                    Array1<int32_t> *temp /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &ctx = row_ids.Context();
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
  // Note: we know that dim >= 1 as we would have returned above if dim == 0.
  // This will do nothing if (dim-1) == 0 as we have checked the first element.
  K2_EVAL(
      ctx, dim - 1, lambda_check_row_ids, (int32_t i)->void {
        if (data[i] > data[i + 1]) *temp_data = 1;  // means it's bad.
      });
  return (*temp)[0] == 0;
}

bool ValidateRowSplits(const Array1<int32_t> &row_splits,
                       Array1<int32_t> *temp /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &ctx = row_splits.Context();
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
  // Note: we know that dim >= 1 as we would have returned above if dim == 0.
  // This will do nothing if (dim-1) == 0 as we have checked the first element.
  K2_EVAL(
      ctx, dim - 1, lambda_check_row_splits, (int32_t i)->void {
        if (data[i] > data[i + 1]) *temp_data = 1;  // means it's bad.
      });
  return (*temp)[0] == 0;
}

bool ValidateRowSplitsAndIds(const Array1<int32_t> &row_splits,
                             const Array1<int32_t> &row_ids,
                             Array1<int32_t> *temp /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  // Check if their context are compatible or not while getting
  ContextPtr ctx = GetContext(row_splits, row_ids);
  int32_t num_rows = row_splits.Dim() - 1, num_elems = row_ids.Dim();
  if (num_rows < 0 || (num_rows == 0 && num_elems > 0)) return false;
  if (row_splits[0] != 0 || (num_elems > 0 && row_ids[0] < 0)) return false;
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

  K2_EVAL(
      ctx, std::max(num_elems, num_rows), lambda_check_row_ids,
      (int32_t i)->void {
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
              i < row_splits_data[this_row] ||
              i >= row_splits_data[this_row + 1])
            *temp_data = 1;
        }
      });
  return (*temp)[0] == 0;
}

void RowSplitsToRowIds(const Array1<int32_t> &row_splits,
                       Array1<int32_t> *row_ids) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetContext(row_splits, *row_ids);
  int32_t num_elems = row_ids->Dim(), num_rows = row_splits.Dim() - 1;
  K2_CHECK_GE(num_rows, 0);
  // if there are more than zero elems, there must be at least one row.
  K2_CHECK(num_elems == 0 || num_rows > 0);
  K2_CHECK_EQ(num_elems, row_splits[num_rows]);
  RowSplitsToRowIds(c, num_rows, row_splits.Data(), num_elems, row_ids->Data());
}

void RowIdsToRowSplits(const Array1<int32_t> &row_ids,
                       Array1<int32_t> *row_splits) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetContext(*row_splits, row_ids);
  int32_t num_elems = row_ids.Dim(), num_rows = row_splits->Dim() - 1;
  K2_CHECK_GE(num_rows, 0);
  // if there are more than zero elems, there must be at least one row.
  K2_CHECK(num_elems == 0 || num_rows > 0);
  if (num_elems > 0) K2_CHECK_GT(num_rows, row_ids[num_elems - 1]);
  RowIdsToRowSplits(c, num_elems, row_ids.Data(), false, num_rows,
                    row_splits->Data());
}

Array1<int32_t> GetCounts(ContextPtr c, const int32_t *src_data,
                          int32_t src_dim, int32_t n) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(n, 0);
  Array1<int32_t> ans(c, n, 0);  // init with 0
  int32_t *ans_data = ans.Data();
  if (n == 0) {
    K2_CHECK_EQ(src_dim, 0);
    return ans;
  }

  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    for (int32_t i = 0; i < src_dim; ++i) {
      ++ans_data[src_data[i]];
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    std::size_t temp_storage_bytes = 0;
    K2_CHECK_CUDA_ERROR(cub::DeviceHistogram::HistogramEven(
        nullptr, temp_storage_bytes, src_data, ans_data, n + 1, 0, n, src_dim,
        c->GetCudaStream()));  // The first time is to determine temporary
                               // device storage requirements.

    constexpr std::size_t kThreshold = (static_cast<std::size_t>(1) << 33);

    if (temp_storage_bytes < kThreshold) {
      RegionPtr temp_storage = NewRegion(c, temp_storage_bytes);
      K2_CHECK_CUDA_ERROR(cub::DeviceHistogram::HistogramEven(
          temp_storage->data, temp_storage_bytes, src_data, ans_data, n + 1, 0,
          n, src_dim, c->GetCudaStream()));
    } else {
      // split the array and do a recursive call
      //
      // See https://github.com/NVIDIA/cub/issues/288
      // for why we split it
      int32_t first_start = 0;          // inclusive
      int32_t first_end = src_dim / 2;  // exclusive
      int32_t first_dim = first_end - first_start;

      int32_t second_start = first_end;  // inclusive
      int32_t second_end = src_dim;      // exclusive
      int32_t second_dim = second_end - second_start;

      Array1<int32_t> first_subset =
          GetCounts(c, src_data + first_start, first_dim, n);
      Array1<int32_t> second_subset =
          GetCounts(c, src_data + second_start, second_dim, n);

      const int32_t *first_subset_data = first_subset.Data();
      const int32_t *second_subset_data = second_subset.Data();
      K2_EVAL(
          c, n, set_ans, (int32_t i)->void {
            ans_data[i] = first_subset_data[i] + second_subset_data[i];
          });
    }
  }
  return ans;
}

Array1<int32_t> GetCounts(const Array1<int32_t> &src, int32_t n) {
  return GetCounts(src.Context(), src.Data(), src.Dim(), n);
}

Array1<int32_t> InvertMonotonicDecreasing(const Array1<int32_t> &src) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  int32_t src_dim = src.Dim();
  const int32_t *src_data = src.Data();
  if (src_dim == 0) {
    return Array1<int32_t>(c, 0);
  }

  K2_DCHECK_GT(src.Back(), 0);  // just call Back when debugging
  // note `src[0]` may do a DeviceToHost memory copy
  int32_t ans_dim = src[0];
  Array1<int32_t> ans(c, ans_dim, 0);  // init with 0
  int32_t *ans_data = ans.Data();

  K2_EVAL(
      c, src_dim, lambda_set_values, (int32_t i)->void {
        K2_DCHECK((i + 1 == src_dim || src_data[i + 1] <= src_data[i]));
        if (i + 1 == src_dim || src_data[i + 1] < src_data[i])
          ans_data[src_data[i] - 1] = i + 1;
      });

  MonotonicDecreasingUpperBound(ans, &ans);
#ifndef NDEBUG
  K2_EVAL(
      c, ans_dim, lambda_check_values, (int32_t i)->void {
        int32_t j = ans_data[i];
        K2_CHECK((j == src_dim || src_data[j] <= i) &&
                 (j == 0 || src_data[j - 1] > i));
      });
#endif

  return ans;
}

Array1<int32_t> InvertPermutation(const Array1<int32_t> &src) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  int32_t dim = src.Dim();
  Array1<int32_t> ans(c, dim);
  const int32_t *src_data = src.Data();
  int32_t *ans_data = ans.Data();

  K2_EVAL(
      c, dim, lambda_set_ans, (int32_t i)->void { ans_data[src_data[i]] = i; });

  return ans;
}

Array1<int32_t> RowSplitsToSizes(const Array1<int32_t> &row_splits) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(row_splits.Dim(), 0);
  ContextPtr &c = row_splits.Context();
  int32_t num_rows = row_splits.Dim() - 1;
  Array1<int32_t> sizes(c, num_rows);
  const int32_t *row_splits_data = row_splits.Data();
  int32_t *sizes_data = sizes.Data();

  K2_EVAL(
      c, num_rows, lambda_set_sizes, (int32_t i)->void {
        sizes_data[i] = row_splits_data[i + 1] - row_splits_data[i];
      });

  return sizes;
}

//  This is modified from RowSplitsToRowIdsKernel.
//  When we invoke this we make a big enough grid that there doesn't have to
//  be a loop over rows, i.e. (gridDim.x * blockDim.x) / threads_per_row >=
//  num_rows
__global__ void SizesToMergeMapKernel(int32_t num_rows, int32_t threads_per_row,
                                      const int32_t *row_splits,
                                      int32_t num_elems, uint32_t *merge_map) {
  int32_t thread = blockIdx.x * blockDim.x + threadIdx.x,
          num_threads = gridDim.x * blockDim.x, row = thread / threads_per_row,
          thread_this_row = thread % threads_per_row;

  if (row >= num_rows) return;
  K2_CHECK_GE(num_threads / threads_per_row, num_rows);

  int32_t this_row_split = row_splits[row],
          next_row_split = row_splits[row + 1],
          row_length = next_row_split - this_row_split;

#pragma unroll(4)
  for (; thread_this_row < row_length; thread_this_row += threads_per_row)
    merge_map[this_row_split + thread_this_row] =
        uint32_t(row) + uint32_t(num_rows) * uint32_t(thread_this_row);
}

Array1<uint32_t> SizesToMergeMap(ContextPtr c,
                                 const std::vector<int32_t> &sizes) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_srcs = sizes.size();

  ContextPtr cpu_context = GetCpuContext();
  Array1<int32_t> row_splits_cpu(cpu_context, num_srcs + 1);
  int32_t *row_splits_cpu_data = row_splits_cpu.Data();
  int32_t tot_size = 0;
  row_splits_cpu_data[0] = 0;
  for (int32_t i = 0; i != num_srcs; ++i) {
    tot_size += sizes[i];
    row_splits_cpu_data[i + 1] = tot_size;
  }
  Array1<uint32_t> ans(c, tot_size);
  if (tot_size == 0) return ans;
  uint32_t *ans_data = ans.Data();

  if (c->GetDeviceType() == kCpu) {
    int32_t cur = 0;
    for (int32_t src = 0; src != num_srcs; ++src) {
      int32_t begin = cur,  // i.e. the previous end.
          end = row_splits_cpu_data[src + 1];
      for (; cur != end; ++cur) {
        // the 'src' says which source this item came from, and (cur - begin)
        // is the position within that source.
        ans_data[cur] =
            uint32_t(src) + uint32_t(cur - begin) * uint32_t(num_srcs);
      }
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    Array1<int32_t> row_splits = row_splits_cpu.To(c);

#if 1
    int32_t avg_elems_per_row = (tot_size + num_srcs - 1) / num_srcs,
            threads_per_row = RoundUpToNearestPowerOfTwo(avg_elems_per_row),
            tot_threads = num_srcs * threads_per_row;
    int32_t block_size = 256;
    int32_t grid_size = NumBlocks(tot_threads, block_size);
    K2_CUDA_SAFE_CALL(
        SizesToMergeMapKernel<<<grid_size, block_size, 0, c->GetCudaStream()>>>(
            num_srcs, threads_per_row, row_splits.Data(), tot_size,
            ans.Data()));
#else
    // Below version can be just faster than the above version when
    // num_srcs > 5000 and tot_size > 1,000,000
    mgpu::context_t *mgpu_context = GetModernGpuAllocator(c);
    auto lambda_set_ans = [=] __device__(uint32_t index, uint32_t seg,
                                         uint32_t rank) {
      ans_data[index] = seg + rank * static_cast<uint32_t>(num_srcs);
    };
    K2_CUDA_SAFE_CALL(mgpu::transform_lbs(lambda_set_ans, tot_size,
                                          row_splits.Data(),
                                          row_splits.Dim() - 1, *mgpu_context));
#endif
  }
  return ans;
}

bool IsPermutation(const Array1<int32_t> &a) {
  NVTX_RANGE(K2_FUNC);
  Array1<int32_t> ones(a.Context(), a.Dim(), 1);
  int32_t *ones_data = ones.Data();
  const int32_t *a_data = a.Data();
  int32_t dim = a.Dim();
  K2_EVAL(
      a.Context(), a.Dim(), lambda_set_zero, (int32_t i)->void {
        if (static_cast<uint32_t>(a_data[i]) < static_cast<uint32_t>(dim)) {
          ones_data[a_data[i]] = 0;
        }
      });
  return Equal(ones, 0);
}

void RowSplitsToRowIdsOffset(const Array1<int32_t> &row_splits_part,
                             Array1<int32_t> *row_ids_part) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = row_splits_part.Context();
  Array1<int32_t> row_splits(c, row_splits_part.Dim());
  int32_t *row_splits_data = row_splits.Data();
  const int32_t *row_splits_part_data = row_splits_part.Data();
  K2_EVAL(
      c, row_splits_part.Dim(), lambda_subtract_offset, (int32_t i) {
        row_splits_data[i] = row_splits_part_data[i] - row_splits_part_data[0];
      });
  RowSplitsToRowIds(row_splits, row_ids_part);
}

template <>
Array2<Any> ToContiguous(const Array2<Any> &src) {
  FOR_REAL_AND_INT32_TYPES(src.GetDtype(), T,
                           return ToContiguous(src.Specialize<T>()).Generic());
  return Array2<Any>();  // Silence warning
}

}  // namespace k2
