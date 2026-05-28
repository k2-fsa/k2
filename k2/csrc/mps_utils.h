/**
 * Copyright      2026  k2-fsa Authors
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

// MPS-accelerated implementations of k2 utility operations using ATen ops.
// Only compiled when K2_WITH_MPS is defined.
//
// Priority-2 optimisations (PyTorch >= 2.2 MPS):
//   • All intermediate int64 casts removed — MPS now supports int32 cumsum,
//     diff, repeat_interleave, searchsorted, and bincount natively.
//   • ExclusiveSumMps no longer allocates a zeroed intermediate tensor;
//     it uses constant_pad_nd to prepend the leading 0 in one fused Metal op.
//   • InclusiveSumMps reduced from 3 ATen ops to 1 (direct int32 cumsum).
//   • RowSplitsToRowIdsMps diff cast removed; repeat_interleave uses int32
//     counts.
//   • MaxSizeMps diff + max now operate in int32 throughout.
#pragma once

#ifdef K2_WITH_MPS

#include "torch/torch.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {
namespace mps_ops {

// Returns a Metal-safe 1-D view of `numel` elements of type `dtype`
// starting at `ptr`. Uses the global MPS registry (via MpsRegistryView)
// to find the PyTorch-owned base tensor and create a proper narrow() view.
inline torch::Tensor AsMpsTensor(void *ptr, int64_t numel,
                                  torch::ScalarType dtype = torch::kInt32) {
  return MpsRegistryView(ptr, numel, dtype);
}
inline torch::Tensor AsMpsTensor(const void *ptr, int64_t numel,
                                  torch::ScalarType dtype = torch::kInt32) {
  return MpsRegistryView(ptr, numel, dtype);
}

// ExclusiveSumMps: dest[i] = sum_{j<i} src[j],  dest[0] = 0.
//
// Implementation: prepend a zero to cumsum(src[0..n-2]) via constant_pad_nd.
// This avoids both the intermediate zeroed allocation and the int64 round-trip
// that were present in the previous version.
inline void ExclusiveSumMps(int32_t n, const int32_t *src, int32_t *dest) {
  if (n == 0) return;
  auto src_t = AsMpsTensor(src, (int64_t)n);   // int32 MPS view
  auto dst_t = AsMpsTensor(dest, (int64_t)n);  // int32 MPS view

  if (n == 1) {
    // Only one element — exclusive sum is always 0.
    dst_t.zero_();
    return;
  }
  // cumsum of src[0..n-2] (length n-1), then pad a 0 at the front → length n.
  // constant_pad_nd({left, right}) pads the last dim; for a 1-D tensor this is
  // {left_pad, right_pad}.  We want one zero prepended.
  dst_t.copy_(torch::constant_pad_nd(src_t.slice(0, 0, n - 1).cumsum(0),
                                     {1, 0}));
}

// InclusiveSumMps: dest[i] = sum_{j<=i} src[j].
//
// int32 cumsum is supported natively on MPS (PyTorch >= 2.2).
inline void InclusiveSumMps(int32_t n, const int32_t *src, int32_t *dest) {
  if (n == 0) return;
  auto src_t = AsMpsTensor(src, (int64_t)n);
  auto dst_t = AsMpsTensor(dest, (int64_t)n);
  dst_t.copy_(src_t.cumsum(0));
}

// RowSplitsToRowIdsMps: row_ids[i] = j  where
//   row_splits[j] <= i < row_splits[j+1].
//
// int32 diff and repeat_interleave(int32 counts) are both supported on MPS.
inline void RowSplitsToRowIdsMps(int32_t num_rows, const int32_t *row_splits,
                                  int32_t num_elems, int32_t *row_ids) {
  if (num_rows <= 0 || num_elems <= 0) return;
  auto row_splits_t = AsMpsTensor(row_splits, (int64_t)(num_rows + 1));
  auto row_ids_t    = AsMpsTensor(row_ids,    (int64_t)num_elems);
  auto counts = torch::diff(row_splits_t);  // int32, length num_rows
  auto arange = torch::arange((int64_t)num_rows,
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));
  row_ids_t.copy_(torch::repeat_interleave(arange, counts));
}

// RowIdsToRowSplitsMps: row_splits[j] = number of elements strictly before
// row j.
//
// int32 searchsorted is supported on MPS.
inline void RowIdsToRowSplitsMps(int32_t num_elems, const int32_t *row_ids,
                                   int32_t num_rows, int32_t *row_splits) {
  auto mps_i32 = torch::TensorOptions()
                     .dtype(torch::kInt32).device(torch::kMPS);
  auto row_ids_t    = AsMpsTensor(row_ids,    (int64_t)num_elems);
  auto row_splits_t = AsMpsTensor(row_splits, (int64_t)(num_rows + 1));
  auto boundaries = torch::arange((int64_t)num_rows, mps_i32);
  // searchsorted returns int64; the copy_ into int32 row_splits_t will cast.
  auto result = torch::searchsorted(row_ids_t.contiguous(), boundaries);
  row_splits_t.slice(0, 0, num_rows).copy_(result);
  row_splits_t.slice(0, num_rows).fill_(num_elems);
}

// MaxSizeMps: max over (row_splits[i+1] - row_splits[i])
// for i in [0, num_rows).
//
// int32 diff + max avoids the previous int64 round-trip.
inline int32_t MaxSizeMps(int32_t num_rows, const int32_t *row_splits) {
  if (num_rows == 0) return 0;
  auto row_splits_t = AsMpsTensor(row_splits, (int64_t)(num_rows + 1));
  return torch::diff(row_splits_t).max().item<int32_t>();
}

// GetCountsMps: ans[v] = number of times v appears in src[0..src_dim).
//
// bincount requires int64 input by design (PyTorch API constraint).
inline void GetCountsMps(const int32_t *src_data, int32_t src_dim,
                          int32_t *ans_data, int32_t n) {
  if (n == 0) return;
  auto src_t = AsMpsTensor(src_data, (int64_t)src_dim);
  auto ans_t = AsMpsTensor(ans_data, (int64_t)n);
  ans_t.copy_(torch::bincount(src_t.to(torch::kInt64), {}, (int64_t)n)
                  .to(torch::kInt32));
}

}  // namespace mps_ops
}  // namespace k2

#endif  // K2_WITH_MPS
