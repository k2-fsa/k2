/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Yiming Wang
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
#include <cmath>
#include <memory>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/cub.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/ragged_utils.h"

namespace {

/*
A helper function used in RaggedShape3;
  if both first and second are non-NULL, it will check if the context of them
     is compatible or not and return that context if compatible;
  if one of them is NULL, returns the other one's context.
 */
static k2::ContextPtr GetContext(const k2::Array1<int32_t> *first,
                                 const k2::Array1<int32_t> *second) {
  K2_CHECK(first != nullptr || second != nullptr)
      << "At least one of first and second must be non-NULL";
  if (first == nullptr)
    return second->Context();
  else if (second == nullptr)
    return first->Context();
  else
    return k2::GetContext(*first, *second);
}

}  // namespace

namespace k2 {

RaggedShape RandomRaggedShape(bool set_row_ids, int32_t min_num_axes,
                              int32_t max_num_axes, int32_t min_num_elements,
                              int32_t max_num_elements) {
  ContextPtr c = GetCpuContext();
  K2_CHECK(min_num_axes >= 2 && max_num_axes >= min_num_axes &&
           min_num_elements >= 0 && max_num_elements >= min_num_elements);
  int32_t num_axes = RandInt(min_num_axes, max_num_axes);
  int32_t num_elements = RandIntGeometric(min_num_elements, max_num_elements);

  bool done_repeats = false;
  std::vector<RaggedShapeLayer> axes(num_axes - 1);
  for (int32_t axis = num_axes - 2; axis >= 0; axis--) {
    // this axis will have row_ids of length num_elements and
    // row_splits of length to be determined.
    int32_t cur_row_split = 0;
    std::vector<int32_t> row_splits_vec;
    std::vector<int32_t> row_ids_vec;
    row_splits_vec.push_back(cur_row_split);
    // The reason for "|| RandInt(0, 2) == 0)" is so that even if there
    // are no elements we can still potentially generate empty row-splits.
    while (cur_row_split < num_elements || RandInt(0, 2) == 0) {
      int32_t split_size = RandIntGeometric(0, num_elements - cur_row_split);
      cur_row_split += split_size;
      // sometimes we have a bunch of empty rows in a row (this will test out
      // more of the code), so here we generate a bunch of empty rows, but we
      // just do this only once (that's why we declare `done_repeats` here).
      if (split_size == 0 && RandInt(0, 30) == 0 && !done_repeats) {
        int32_t num_repeats = RandIntGeometric(1, 128);
        row_splits_vec.insert(row_splits_vec.end(), num_repeats, cur_row_split);
        // don't need to set `row_ids_vec` as there's no element.
        done_repeats = true;
      }
      row_splits_vec.push_back(cur_row_split);
      if (set_row_ids) {
        int32_t cur_row = static_cast<int32_t>(row_splits_vec.size()) - 2;
        row_ids_vec.insert(row_ids_vec.end(), split_size, cur_row);
      }
    }
    axes[axis].row_splits = Array1<int32_t>(c, row_splits_vec);
    if (set_row_ids) axes[axis].row_ids = Array1<int32_t>(c, row_ids_vec);
    axes[axis].cached_tot_size = num_elements;
    num_elements = axes[axis].row_splits.Dim() - 1;
  }
  // RaggedShape(axes, true) will check the returned RaggedShape for
  // consistency.
  return RaggedShape(axes, true);
}

RaggedShape RaggedShape2(Array1<int32_t> *row_splits, Array1<int32_t> *row_ids,
                         int32_t cached_tot_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(row_splits != nullptr || row_ids != nullptr)
      << "At least one of row_splits and row_ids must be defined";
  ContextPtr ctx = ::GetContext(row_splits, row_ids);
  if (cached_tot_size != -1) {
    if (row_ids != nullptr) K2_CHECK_EQ(cached_tot_size, row_ids->Dim());
    if (row_splits != nullptr) {
      // may be slow as it may copy memory from device to host
      K2_DCHECK_EQ(cached_tot_size, row_splits->Back())
          << "Bad row splits is: " << *row_splits;
    }
  }
  std::vector<RaggedShapeLayer> axes(1);
  if (row_splits != nullptr) {
    axes[0].row_splits = *row_splits;
  } else {
    // we need to work out row_splits as we always require row_splits is not
    // empty for RaggedShape. Note here we suppose the last element in row_ids
    // is num_rows - 1, i.e. there're no empty rows after row `row_ids[-1]`.
    int32_t num_rows = row_ids->Dim() == 0 ? 0 : row_ids->Back() + 1;
    Array1<int32_t> row_splits_array(ctx, num_rows + 1);
    RowIdsToRowSplits(*row_ids, &row_splits_array);
    axes[0].row_splits = row_splits_array;
  }
  if (row_ids != nullptr) axes[0].row_ids = *row_ids;
  if (cached_tot_size == -1) {
    cached_tot_size =
        row_ids != nullptr ? row_ids->Dim() : axes[0].row_splits.Back();
  }
  axes[0].cached_tot_size = cached_tot_size;
  // note below line will check if row_splits and row_ids are valid and agree
  // with each other.
  return RaggedShape(axes);
}

RaggedShape ComposeRaggedShapes(const RaggedShape &a, const RaggedShape &b) {
  NVTX_RANGE(K2_FUNC);
  if (a.NumElements() != b.Dim0()) {
    K2_LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: " << a.NumElements()
                  << " vs. " << b.Dim0();
  }
  K2_CHECK(IsCompatible(a, b));
  const auto &a_axes = a.Layers();
  const auto &b_axes = b.Layers();
  std::size_t a_size = a_axes.size(), b_size = b_axes.size();
  std::vector<RaggedShapeLayer> axes;
  axes.reserve(a_size + b_size);
  for (std::size_t i = 0; i < a_size; ++i) axes.emplace_back(a_axes[i]);
  for (std::size_t i = 0; i < b_size; ++i) axes.emplace_back(b_axes[i]);
  bool validate = false;
  return RaggedShape(axes, validate);
}

RaggedShape ComposeRaggedShapes3(const RaggedShape &a, const RaggedShape &b,
                                 const RaggedShape &c) {
  NVTX_RANGE(K2_FUNC);
  if (a.NumElements() != b.Dim0()) {
    K2_LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: " << a.NumElements()
                  << " vs. " << b.Dim0();
  }
  if (b.NumElements() != c.Dim0()) {
    K2_LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: " << b.NumElements()
                  << " vs. " << c.Dim0();
  }
  K2_CHECK(IsCompatible(a, b));
  K2_CHECK(IsCompatible(b, c));
  const auto &a_axes = a.Layers();
  const auto &b_axes = b.Layers();
  const auto &c_axes = c.Layers();
  std::size_t a_size = a_axes.size(), b_size = b_axes.size(),
              c_size = c_axes.size();
  std::vector<RaggedShapeLayer> axes;
  axes.reserve(a_size + b_size + c_size);
  for (std::size_t i = 0; i < a_size; ++i) axes.emplace_back(a_axes[i]);
  for (std::size_t i = 0; i < b_size; ++i) axes.emplace_back(b_axes[i]);
  for (std::size_t i = 0; i < c_size; ++i) axes.emplace_back(c_axes[i]);
  bool validate = false;
  return RaggedShape(axes, validate);
}

RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2) {
  NVTX_RANGE(K2_FUNC);

  RaggedShape shape1 = RaggedShape2(row_splits1, row_ids1, cached_tot_size1);

  Array1<int32_t> temp_array;
  if (row_splits2 == nullptr) {
    K2_CHECK_NE(row_ids2, nullptr)
        << "Either row-splits or row-ids must be defined";
    temp_array = Array1<int32_t>(row_ids2->Context(), shape1.NumElements() + 1);
    row_splits2 = &temp_array;
    RowIdsToRowSplits(*row_ids2, row_splits2);
  }

  return ComposeRaggedShapes(
      shape1, RaggedShape2(row_splits2, row_ids2, cached_tot_size2));
}

RaggedShape RaggedShape4(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2,
                         Array1<int32_t> *row_splits3,
                         Array1<int32_t> *row_ids3, int32_t cached_tot_size3) {
  NVTX_RANGE(K2_FUNC);

  RaggedShape shape12 = RaggedShape3(row_splits1, row_ids1, cached_tot_size1,
                                     row_splits2, row_ids2, cached_tot_size2);
  Array1<int32_t> temp_array;
  if (row_splits3 == nullptr) {
    K2_CHECK_NE(row_ids3, nullptr)
        << "Either row-splits or row-ids must be defined";
    temp_array =
        Array1<int32_t>(row_ids3->Context(), shape12.NumElements() + 1);
    row_splits3 = &temp_array;
    RowIdsToRowSplits(*row_ids3, row_splits3);
  }
  return ComposeRaggedShapes(
      shape12, RaggedShape2(row_splits3, row_ids3, cached_tot_size3));
}

RaggedShape RaggedShapeFromTotSizes(ContextPtr c, int32_t num_axes,
                                    const int32_t *tot_sizes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(num_axes, 2);
  std::vector<RaggedShapeLayer> axes(num_axes - 1);
  int32_t tot_size = 0;
  for (int32_t axis = 1; axis < num_axes; ++axis) {
    tot_size += tot_sizes[axis - 1] + 1 + tot_sizes[axis];
  }
  Array1<int32_t> buf(c, tot_size);
  int32_t start = 0;
  for (int32_t axis = 1; axis < num_axes; ++axis) {
    axes[axis - 1].row_splits = buf.Arange(start,
        start + tot_sizes[axis - 1] + 1);
    start += tot_sizes[axis - 1] + 1;
    axes[axis - 1].row_ids = buf.Arange(start, start + tot_sizes[axis]);
    start += tot_sizes[axis];
    axes[axis - 1].cached_tot_size = tot_sizes[axis];
  }
  // Not check here as we did not set the values of row_splits and row_ids
  return RaggedShape(axes, false);
}

// See declaration in ragged.h for documentation of its purpose and interface.
RaggedShape Unsqueeze(const RaggedShape &src, int32_t axis) {
  // If axis == 0, initial row_splits and row_ids will look like the following,
  // if for example src.Dim0() was 5: [ 0 5 ],  [ 0 0 0 0 0 ].  The other axes
  // would be pushed forward.
  //
  // If 0 < axis <= src.NumAxes(), the inserted row_splits and row_ids would
  // look like the following, if for instance the src.TotSize(axis) = 8:
  //   [ 0 1 2 3 4 5 6 7 8 ], [ 0 1 2 3 4 5 6 7 ].
  //
  // The reason why the code is different for axis == 0, is that in that case we
  // are really making visible an "implicit" axis of the input `src`; we could
  // call it axis 0 of the original RaggedShape.  Imagine that "implicit" axis's
  // row_splits and row_ids map respectively from an idx_minus1 -> idx0 and from
  // an idx_0 to idx_minus1, where idx_minus1 is always 0 and 0 <= idx0 <
  // Dim0().

  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  K2_CHECK(axis >= 0 && axis <= src.NumAxes());

  const std::vector<RaggedShapeLayer> &axes_in = src.Layers();
  int32_t num_axes_in = src.NumAxes();

  // Note: in RaggedShape, the vector of RaggedShapeLayer is of length
  // num_axes - 1, so the output will have one more axis than the input.
  std::vector<RaggedShapeLayer> axes_out(num_axes_in);

  int32_t row_splits_dim, row_ids_dim;
  Array1<int32_t> mem;

  if (axis == 0) {
    row_splits_dim = 2;        // e.g. [ 0 5 ]
    row_ids_dim = src.Dim0();  // e.g. [ 0 0 0 0 0 ]
    mem = Array1<int32_t>(c, row_splits_dim + row_ids_dim);
    int32_t *mem_data = mem.Data();
    K2_EVAL(
        c, mem.Dim(), lambda_set_mem, (int32_t i)->void {
          if (i == 1)
            mem_data[i] = row_ids_dim;
          else
            mem_data[i] = 0;
        });
  } else {
    int32_t tot_size = src.TotSize(axis);
    row_splits_dim = tot_size + 1;
    row_ids_dim = tot_size;
    mem = Array1<int32_t>(c, row_splits_dim + row_ids_dim);
    int32_t *mem_data = mem.Data();
    K2_EVAL(
        c, mem.Dim(), lambda_set_mem2,
        (int32_t i)->void { mem_data[i] = i % (tot_size + 1); });
  }
  axes_out[axis].row_splits = mem.Range(0, row_splits_dim);
  axes_out[axis].row_ids = mem.Range(row_splits_dim, row_ids_dim);
  axes_out[axis].cached_tot_size = row_ids_dim;
  for (int32_t i = 0; i < axis; ++i) axes_out[i] = axes_in[i];
  // Note: the returned array has `num_axes_in + 1` axes, so its
  // array of RaggedShapeLayer is of length `num_axes_in`.
  for (int32_t i = axis + 1; i < num_axes_in; ++i) axes_out[i] = axes_in[i - 1];
  return RaggedShape(axes_out);
}

std::vector<RaggedShape> UnsqueezeParallel(int32_t num_srcs, RaggedShape **src,
                                           int32_t axis) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(axis, 0);
  std::vector<RaggedShape> ans;
  if (num_srcs == 0) return ans;
  ans.reserve(num_srcs);
  ContextPtr &c = src[0]->Context();

  std::vector<int32_t> all_row_splits_vec(num_srcs * 2);
  int32_t max_dim = 0;
  // all_row_splits_vec will contain [ 0 d0 0 d1 0 d2 .. ]
  // where d0 == src[0]->Dim0(), d1 == src[1]->Dim0()..
  for (int32_t i = 0; i < num_srcs; ++i) {
    int32_t this_dim0 = src[i]->Dim0();
    if (this_dim0 > max_dim) max_dim = this_dim0;
    all_row_splits_vec[i * 2] = 0;
    all_row_splits_vec[i * 2 + 1] = this_dim0;
  }
  Array1<int32_t> all_row_splits(c, all_row_splits_vec);
  Array1<int32_t> all_row_ids(c, max_dim, 0);

  for (int32_t i = 0; i < num_srcs; ++i) {
    int32_t num_axes = src[i]->NumAxes();
    std::vector<RaggedShapeLayer> axes;
    axes.reserve(num_axes);  //  note, the size of the `layers` of a RaggedShape
                             //  is its NumAxes() - 1.
    axes.resize(1);
    int32_t this_old_dim0 = all_row_splits_vec[i * 2 + 1];
    axes[0].row_splits = all_row_splits.Range(i * 2, 2);
    axes[0].row_ids = all_row_ids.Range(0, this_old_dim0);
    axes[0].cached_tot_size = this_old_dim0;
    axes.insert(axes.end(), src[i]->Layers().begin(), src[i]->Layers().end());
    ans.emplace_back(std::move(axes));
  }
  return ans;
}

/*
  Internal function used in Index(), which gets certain arrays used internally.

     @param [in] src      Source shape to be indexed
     @param [in] new2old  Array of indexes into axis 0 of src; elements
                         equal to -1 will be interpreted as referring to
                         an empty list.
     @param [out] old_offsets   Will be set to new Array2 with dimension
                         (src.NumAxes(), new2old.Dim()), whose (i,j)'th
                         element contains the offset into axis i of `src`
                         where the slice of `src` with index0 (i.e. index
                         into 0'th-axis of `src`) equal to `new2old[j]`
                         begins.
     @param [out] new_offsets   Will be set to new Array2 with dimension
                         (src.NumAxes(), new2old.Dim()+1), whose (i,j)'th
                         element contains the offset into axis i of `ans`
                         where the data in `ans` corresponding to
                         index j (i.e. index j into axis 0 of `ans`) begins.
                         Note: `ans` is the result of Index(), with
                         ans.Dim0() == new2old.Dim().
 */
inline void GetOldAndNewOffsets(RaggedShape &src,
                                const Array1<int32_t> &new2old,
                                Array2<int32_t> *old_offsets,
                                Array2<int32_t> *new_offsets) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(src.NumAxes(), 1);
  ContextPtr &c = src.Context();
  int32_t num_axes = src.NumAxes(), ans_dim0 = new2old.Dim();

  // max 5 layers.
  RowSplitsAccessor<5> row_splits_acc(src);

  const int32_t *new2old_data = new2old.Data();
  *old_offsets = Array2<int32_t>(c, num_axes, ans_dim0);
  *new_offsets = Array2<int32_t>(c, num_axes, ans_dim0 + 1);
  auto old_offsets_acc = old_offsets->Accessor(),
       new_offsets_acc = new_offsets->Accessor();
  // Set old_offsets; and for now, set new_offsets to the corresponding
  // sizes of the output slices.
  K2_EVAL(
      c, ans_dim0, lambda_set_offsets, (int32_t i)->void {
        // 0 <= i < ans_dim0
        int32_t old_offset = new2old_data[i],
            old_offset_next = old_offset + 1,
            offset_diff = 1;
        // The following is a special case that interprets -1 as referring to an
        // empty list.  In this case, old_offset == old_offset_next == 0.
        // The specific value 0 is not necessary; they could be equal
        // and have any value in [0, src.Dim0() - 1] and still refer to
        // the empty list.
        if (old_offset == -1)
          old_offset = 0;
        for (int32_t axis = 0;; axis++) {
          old_offsets_acc(axis, i) = old_offset;
          // Below, 'new_offsets_acc' currently contains the size rather
          // than the offset; we need to do exclusive-sum.
          new_offsets_acc(axis, i) = offset_diff;
          if (axis + 1 == num_axes) return;
          old_offset = row_splits_acc(axis)[old_offset];
          old_offset_next = row_splits_acc(axis)[old_offset_next];
          offset_diff = old_offset_next - old_offset;
        }
      });
  ExclusiveSum(*new_offsets, new_offsets);
}

// Don't make it static to fix the following error on Windows.
// Error : On Windows, the enclosing parent function ("IndexAxis0") for an
// extended __host__ __device__ lambda cannot have internal or no linkage
/*static*/ RaggedShape IndexAxis0(RaggedShape &src,
                                  const Array1<int32_t> &new2old,
                                  Array1<int32_t> *elem_indexes /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  K2_CHECK(IsCompatible(src, new2old));
  int32_t num_axes = src.NumAxes(), src_dim0 = src.Dim0(),
          ans_dim0 = new2old.Dim();
  if (ans_dim0 == 0) {
    if (elem_indexes) *elem_indexes = Array1<int32_t>(c, 0);
    return EmptyRaggedShape(c, num_axes);
  }


  Array2<int32_t> old_offsets,  // num_axes by ans_dim0
      new_offsets;              // num_axes by (ans_dim0 + 1).
  GetOldAndNewOffsets(src, new2old, &old_offsets, &new_offsets);

  // tot_sizes_out is of dimension (num_axes), tot_sizes_out[i] is
  // ans.TotSize(i)
  Array1<int32_t> tot_sizes_out =
      Array1<int32_t>(new_offsets.Col(ans_dim0)).To(GetCpuContext());

  int32_t *tot_sizes_out_cpu_data = tot_sizes_out.Data();
  if (elem_indexes)
    *elem_indexes = Array1<int32_t>(c, tot_sizes_out_cpu_data[num_axes - 1]);

  RaggedShape ans =
      RaggedShapeFromTotSizes(c, num_axes, tot_sizes_out_cpu_data);

  auto old_offsets_acc = old_offsets.Accessor(),
      new_offsets_acc = new_offsets.Accessor();

  for (int32_t axis = 1; axis < num_axes; axis++) {
    // we are not creating the actual row_ids here, except for axis 1; we are
    // creating "composed row_ids" which map to the index on axis 0.
    Array1<int32_t> row_ids = ans.RowIds(axis);
    RowSplitsToRowIds(new_offsets.Row(axis), &row_ids);
  }

  ans.Layers()[0].row_splits = new_offsets.Row(1);

  // Caution: e.g. old_row_splits_acc(i) == src.RowSplits(i+1).
  RowSplitsAccessor<5> old_row_splits_acc(src),
      new_row_splits_acc(ans);
  RowIdsAccessor<5> old_row_ids_acc(src),
      new_row_ids_acc(ans);
  SmallVec<int32_t, 6> tot_sizes;
  K2_CHECK(num_axes <= 6);
  int32_t max_tot_size = 0;
  for (int32_t i = 0; i < num_axes; i++) {
    tot_sizes.data[i] = tot_sizes_out_cpu_data[i];
    max_tot_size = std::max<int32_t>(max_tot_size,
                                     tot_sizes.data[i]);
  }

  int32_t *elem_indexes_data = (elem_indexes != nullptr ?
                                elem_indexes->Data() : nullptr);

  // Note, the first row_splits vector was set above, ans.Layers()[0].row_splits
  // = new_offsets.Row(1).

  auto lambda_set_row_splits_and_ids = [=] __host__ __device__(
                                           int32_t axis, int32_t i) -> void {
    axis++;  // make it one-based.
    int32_t tot_size = tot_sizes(axis);  // == new_offsets_acc(axis, ans_dim0);
    if (i > tot_size)
      return;
    int32_t *composed_row_ids_data = new_row_ids_acc(axis - 1);
    int32_t ans_idx0 = (i == tot_size ? ans_dim0 :
                        composed_row_ids_data[i]),
    job_begin = new_offsets_acc(axis, ans_idx0),
    job_this_idx0 = i - job_begin;
    K2_CHECK_GE(job_this_idx0, 0);
    int32_t row_split_value = 0, new_next_offset = 0;
    if (axis + 1 < num_axes)
      new_next_offset = new_offsets_acc(axis + 1, ans_idx0);
    if (i < tot_size) {
      // "prev" means for axis - 1
      int32_t new_prev_offset = new_offsets_acc(axis - 1, ans_idx0),
          old_prev_offset = old_offsets_acc(axis - 1, ans_idx0),
          old_offset = old_offsets_acc(axis, ans_idx0),
          old_idx = old_offset + job_this_idx0;

      if (axis != 1) {
        // Write row-ids.
        // Actually doing this for axis == 1 is harmless, but unnecessary, as it
        // would write back the same values that were already there.  We avoid
        // the memory access.
        // this_new_row_ids = new_row_ids_acc(axis - 1);
        int32_t *this_new_row_ids = composed_row_ids_data;
        const int32_t *this_old_row_ids = old_row_ids_acc(axis - 1);
        int32_t old_row_id = this_old_row_ids[old_idx],
            new_row_id = old_row_id + new_prev_offset - old_prev_offset;
        this_new_row_ids[i] = new_row_id;
      }

      if (elem_indexes_data != nullptr && axis == num_axes - 1)
        elem_indexes_data[i] = old_idx;

      if (axis + 1 < num_axes) {
        int32_t old_next_offset = old_offsets_acc(axis + 1, ans_idx0),
            next_offset_diff = new_next_offset - old_next_offset;
        const int32_t *old_row_splits_data = old_row_splits_acc(axis);
        row_split_value = next_offset_diff + old_row_splits_data[old_idx];
      }
    } else {
      row_split_value = new_next_offset;
    }
    if (axis + 1 < num_axes) {
      int32_t *new_row_splits_data = new_row_splits_acc(axis);
      new_row_splits_data[i] = row_split_value;
    }
  };

  constexpr int32_t cutoff = 50000;
  if (c->GetDeviceType() == kCpu) {
    for (int32_t axis = 0; axis < num_axes - 1; axis++) {
      int32_t this_size = tot_sizes(axis + 1);
      for (int32_t i = 0; i <= this_size; i++)
        lambda_set_row_splits_and_ids(axis, i);
    }
  } else if (max_tot_size * (num_axes - 1) < cutoff) {
    Eval2Device(c, num_axes - 1, max_tot_size + 1,
                lambda_set_row_splits_and_ids);
  } else {
    // Loop in the kernel rather than submitting an excessive number of threads.
    auto lambda_loop = [=] __device__(int32_t i) {
      for (int32_t axis = 0; axis < num_axes - 1; axis++) {
        lambda_set_row_splits_and_ids(axis, i);
      }
    };
    EvalDevice(c, max_tot_size + 1, lambda_loop);
  }
#if !defined(NDEBUG)
  ans.Check();
#endif
  return ans;
}

RaggedShape Index(RaggedShape &src, int32_t axis,
                  const Array1<int32_t> &indexes,
                  Array1<int32_t> *elem_indexes /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  K2_CHECK_LT(static_cast<uint32_t>(axis), static_cast<uint32_t>(num_axes));
  if (axis == 0) {
    return IndexAxis0(src, indexes, elem_indexes);
  } else if (axis == src.NumAxes() - 1) {
    // This code is related to SubsetRaggedShape(). `indexes` corresponds
    // to `new2old`.
    Array1<int32_t> last_row_ids = src.RowIds(num_axes - 1)[indexes];
#ifndef NDEBUG
    if (!IsMonotonic(last_row_ids)) {
      K2_LOG(FATAL) << "Invalid indexes used when indexing RaggedShape";
    }
#endif
    Array1<int32_t> last_row_splits(last_row_ids.Context(),
                                    src.TotSize(num_axes - 2) + 1);
    RowIdsToRowSplits(last_row_ids, &last_row_splits);

    if (elem_indexes)
      *elem_indexes = indexes;

    std::vector<RaggedShapeLayer> axes = src.Layers();
    axes.back().row_splits = last_row_splits;
    axes.back().row_ids = last_row_ids;
    axes.back().cached_tot_size = last_row_ids.Dim();
    // TODO: disable checking by changing true to false.
    return RaggedShape(axes, true);
  } else {
    RaggedShape top, bottom;
    DecomposeRaggedShape(src, axis, &top, &bottom);

    RaggedShape top_indexed = Index(top, axis, indexes, nullptr),
        bottom_indexed = IndexAxis0(bottom, indexes, elem_indexes);
    return ComposeRaggedShapes(top_indexed, bottom_indexed);
  }
}

// returns array of dim (src[0]->NumAxes() + 1) by (num_srcs + 1),
// see documentation in header.
Array2<int32_t> GetOffsets(int32_t num_srcs, RaggedShape **src) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  int32_t num_axes_in = src[0]->NumAxes();
  ContextPtr &ctx = src[0]->Context();
  Array2<int32_t> src_offsets(GetCpuContext(), num_axes_in + 1, num_srcs + 1);
  int32_t *src_offsets_data = src_offsets.Data();
  int32_t src_offsets_stride0 = src_offsets.ElemStride0();

  // Check if they have same num-axes and compatible context
  for (int32_t i = 1; i < num_srcs; ++i) {
    K2_CHECK_EQ(src[i]->NumAxes(), num_axes_in);
    K2_CHECK(ctx->IsCompatible(*src[i]->Context()));
  }

  for (int32_t axis = 0; axis <= num_axes_in; ++axis) {
    int32_t sum = 0;
    for (int32_t i = 0; i <= num_srcs; ++i) {  // i is the column
      src_offsets_data[axis * src_offsets_stride0 + i] = sum;
      if (i < num_srcs) {
        sum += (axis == 0 ? 1 : src[i]->TotSize(axis - 1));
      }
    }
  }
  return src_offsets;
}

void GetRowInfo(RaggedShape &src, Array1<int32_t *> *row_splits,
                Array1<int32_t *> *row_ids) {
  NVTX_RANGE(K2_FUNC);
  int32_t axes = src.NumAxes();
  K2_CHECK_GE(axes, 2);
  src.Populate();
  std::vector<int32_t *> row_splits_ptrs(axes - 1);
  std::vector<int32_t *> row_ids_ptrs(axes - 1);
  for (int32_t i = 1; i != axes; ++i) {
    row_splits_ptrs[i - 1] = src.RowSplits(i).Data();
    row_ids_ptrs[i - 1] = src.RowIds(i).Data();
  }
  ContextPtr ctx = src.Context();
  *row_splits = Array1<int32_t *>(ctx, row_splits_ptrs);
  *row_ids = Array1<int32_t *>(ctx, row_ids_ptrs);
}

void GetRowInfoMulti(int32_t num_srcs, RaggedShape **src,
                     Array2<int32_t *> *row_splits,
                     Array2<int32_t *> *row_ids) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  int32_t num_axes_in = src[0]->NumAxes();
  K2_CHECK_GE(num_axes_in, 2);
  ContextPtr ctx = src[0]->Context();

  // check if they have same num-axes and compatible context
  for (int32_t i = 1; i < num_srcs; ++i) {
    K2_CHECK_EQ(src[i]->NumAxes(), num_axes_in);
    K2_CHECK(ctx->IsCompatible(*src[i]->Context()));
  }

  Array2<int32_t *> row_splits_ptrs(GetCpuContext(), num_axes_in - 1, num_srcs);
  Array2<int32_t *> row_ids_ptrs(GetCpuContext(), num_axes_in - 1, num_srcs);
  int32_t **splits_ptr_data = row_splits_ptrs.Data();
  int32_t **ids_ptr_data = row_ids_ptrs.Data();

  int32_t stride0 = row_splits_ptrs.ElemStride0();
  K2_CHECK_EQ(stride0, row_ids_ptrs.ElemStride0());

  for (int32_t axis = 0; axis != num_axes_in - 1; ++axis) {
    for (int32_t i = 0; i != num_srcs; ++i) {
      splits_ptr_data[axis * stride0 + i] = src[i]->RowSplits(axis + 1).Data();
      ids_ptr_data[axis * stride0 + i] = src[i]->RowIds(axis + 1).Data();
    }
  }
  *row_splits = row_splits_ptrs.To(ctx);
  *row_ids = row_ids_ptrs.To(ctx);
}

/*static*/ RaggedShape StackAxis0(int32_t num_srcs, RaggedShape **src,
                                  Array1<uint32_t> *merge_map /* == nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  if (num_srcs == 1) {
    if (merge_map)
      *merge_map =
          Arange<uint32_t>(src[0]->Context(), 0, src[0]->NumElements());
    RaggedShape top_layer = TrivialShape(src[0]->Context(), src[0]->Dim0());
    return ComposeRaggedShapes(top_layer, **src);
  }
  // We can't handle num_srcs == 0 because we won't have a context object.
  K2_CHECK_GT(num_srcs, 1);

  int32_t num_axes_in = src[0]->NumAxes(),
      num_axes_out = num_axes_in + 1;
  ContextPtr c = src[0]->Context();

  // Check if they have same num-axes and compatible context
  for (int32_t i = 1; i < num_srcs; ++i) {
    K2_CHECK_EQ(num_axes_in, src[i]->NumAxes());
    K2_CHECK(IsCompatible(*src[0], *src[i]));
  }

  // `offsets` will be on CPU for now.
  // It shape is (num_axes_in + 1 == num_axes_out, num_srcs + 1).
  Array2<int32_t> offsets = GetOffsets(num_srcs, src);
  auto offsets_acc = offsets.Accessor();

  SmallVec<int32_t, 6> tot_sizes_out;
  K2_CHECK(num_axes_out <= 6);
  int32_t max_tot_size = 0;
  for (int32_t axis = 0; axis < num_axes_out; axis++) {
    tot_sizes_out.data[axis] = offsets_acc(axis, num_srcs);
    max_tot_size = std::max<int32_t>(max_tot_size,
                                     tot_sizes_out.data[axis]);
  }

  RaggedShape ans = RaggedShapeFromTotSizes(c, num_axes_out,
                                            tot_sizes_out.data);

  // src_row_splits and src_row_ids are of dim num_axes_in-1 by num_srcs.
  Array2<int32_t *> src_row_splits, src_row_ids;
  GetRowInfoMulti(num_srcs, src, &src_row_splits, &src_row_ids);
  auto src_row_splits_acc = src_row_splits.Accessor(),
       src_row_ids_acc = src_row_ids.Accessor();


  offsets = offsets.To(c);
  offsets_acc = offsets.Accessor();
  for (int32_t axis = 1; axis < num_axes_out; axis++) {
    // we are not creating the actual row_ids here, except for axis 1; we are
    // creating "composed row_ids" which map to the index on axis 0.
    Array1<int32_t> row_ids = ans.RowIds(axis);
    RowSplitsToRowIds(offsets.Row(axis), &row_ids);
  }
  ans.Layers()[0].row_splits = offsets.Row(1);

  // Caution: e.g. old_row_splits_acc(i) == src.RowSplits(i+1).
  RowSplitsAccessor<5> new_row_splits_acc(ans);
  RowIdsAccessor<5> new_row_ids_acc(ans);


  uint32_t *merge_map_data;
  if (merge_map != nullptr) {
    *merge_map = Array1<uint32_t>(c, tot_sizes_out.data[num_axes_out - 1]);
    merge_map_data = merge_map->Data();
  } else {
    merge_map_data = nullptr;
  }

  // Note, the first row_splits vector was set above, ans.Layers()[0].row_splits
  // = new_offsets.Row(1).

  auto lambda_set_row_splits_and_ids = [=] __host__ __device__(
                                           int32_t axis, int32_t i) -> void {
    ++axis;  // We want this to be called starting with axis == 1, but Eval2
             // doesn't suppor that.

    // At this point, 1 < axis < num_axes_out.

    // This kernel will be writing one or both of:
    //    the row-splits for output-layer==`axis`/input-layer==`axis-1`,
    //    the row-ids for output-layer=`axis-1`/input-layer==`axis-2`.

    int32_t tot_size = tot_sizes_out(axis);  // == offsets_acc(axis, num_srcs);
    if (i > tot_size)
      return;
    int32_t *composed_row_ids_data = new_row_ids_acc(axis - 1);
    int32_t ans_idx0 =
                (i == tot_size
                     ? num_srcs
                     : composed_row_ids_data[i]),  // note: ans_idx0 == src_idx.
        job_begin = offsets_acc(axis, ans_idx0), job_this_idx0 = i - job_begin;
    K2_CHECK_GE(job_this_idx0, 0);
    int32_t row_split_value = 0,  new_next_offset = 0;
    uint32_t *merge_map_data_local = nullptr;
    if (axis + 1 < num_axes_out) {
      new_next_offset = offsets_acc(axis + 1, ans_idx0);
    } else {
      merge_map_data_local = merge_map_data;
    }
    if (i < tot_size) {
      // "prev" means for axis - 1
      int32_t new_prev_offset = offsets_acc(axis - 1, ans_idx0);
      if (axis != 1) {
        // Write row-ids.
        // this_new_row_ids = new_row_ids_acc(axis - 1);
        int32_t *this_new_row_ids = composed_row_ids_data;
        const int32_t *this_src_row_ids = src_row_ids_acc(axis - 2, ans_idx0);
        int32_t old_row_id = this_src_row_ids[job_this_idx0],
            new_row_id = old_row_id + new_prev_offset;
        this_new_row_ids[i] = new_row_id;
      }

      if (merge_map_data_local != nullptr) {
        merge_map_data_local[i] = ans_idx0 + num_srcs * job_this_idx0;
      }

      if (axis + 1 < num_axes_out) {
        const int32_t *src_row_splits_data = src_row_splits_acc(axis - 1,
                                                                ans_idx0);
        int32_t old_row_split = src_row_splits_data[job_this_idx0];
        row_split_value = new_next_offset + old_row_split;
      }
    } else {
      row_split_value = new_next_offset;
    }
    if (axis + 1 < num_axes_out) {
      int32_t *new_row_splits_data = new_row_splits_acc(axis);
      new_row_splits_data[i] = row_split_value;
    }
  };

  constexpr int32_t cutoff = 50000;
  if (c->GetDeviceType() == kCpu) {
    for (int32_t axis = 0; axis < num_axes_out - 1; axis++) {
      int32_t this_size = tot_sizes_out(axis + 1);
      for (int32_t i = 0; i <= this_size; i++)
        lambda_set_row_splits_and_ids(axis, i);
    }
  } else if (max_tot_size * (num_axes_out - 1) < cutoff) {
    Eval2Device(c, num_axes_out - 1, max_tot_size + 1,
                lambda_set_row_splits_and_ids);
  } else {
    // Loop in the kernel rather than submitting an excessive number of threads.
    auto lambda_loop = [=] __device__(int32_t i) {
      for (int32_t axis = 0; axis < num_axes_out - 1; axis++) {
        lambda_set_row_splits_and_ids(axis, i);
      }
    };
    EvalDevice(c, max_tot_size + 1, lambda_loop);
  }
#if !defined(NDEBUG)
  ans.Check();
#endif
  return ans;
}

RaggedShape Cat(int32_t axis, int32_t num_srcs, RaggedShape **src,
                Array1<uint32_t> *merge_map /* == nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  if (axis == 0) {
    RaggedShape temp = StackAxis0(num_srcs, src, merge_map);
    std::vector<RaggedShapeLayer> ans_layers(
        temp.Layers().begin() + 1, temp.Layers().end());
    return RaggedShape(ans_layers, false);
  }

  K2_CHECK_LT(static_cast<uint32_t>(axis),
              static_cast<uint32_t>(src[0]->NumAxes()));

  int32_t num_axes = src[0]->NumAxes();
  std::vector<RaggedShapeLayer> ans_layers(num_axes - 1);

  // If axis >= 2, some layers of `src` will pass through unchanged (we should
  // check that they are identical across all sources).
  for (int32_t l = 0; l + 1 < axis; l++) {
    CheckLayerEqual(l, num_srcs, src);
    ans_layers[l] = src[0]->Layers()[l];
  }

  Array1<uint32_t> merge_map_local;
  Array1<uint32_t> *this_m =
      (axis + 1 == num_axes ? merge_map : &merge_map_local);
  RaggedShape s = IntersperseRaggedLayer(axis - 1, num_srcs, src, this_m),
              t = SubsampleRaggedLayer(s, 0, num_srcs);
  ans_layers[axis - 1] = t.Layers()[0];

  for (int32_t l = axis; l + 1 < num_axes; l++) {
    Array1<uint32_t> merge_map_next;
    Array1<uint32_t> *this_m =
        (l + 2 == num_axes ? merge_map : &merge_map_next);
    RaggedShape r = MergeRaggedLayer(l, num_srcs, src, merge_map_local, this_m);
    ans_layers[l] = r.Layers()[0];
    merge_map_local = merge_map_next;
  }
  // TODO(dan) after this is debugged: add ", false".
  return RaggedShape(ans_layers);
}

RaggedShape RemoveAxis(RaggedShape &src, int32_t axis) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(src.NumAxes(), 2);
  K2_CHECK(axis >= 0 && axis < src.NumAxes());

  // note, `axes_in` is of dim src.NumAxes() - 1.
  // Also note: axes_in[i] pertains to the relationship between
  // axes i and i+1 in the source.
  src.Populate();

  const std::vector<RaggedShapeLayer> &axes_in = src.Layers();

  std::vector<RaggedShapeLayer> axes_out(axes_in.size() - 1);
  int32_t axes_out_size = static_cast<int32_t>(axes_out.size());

  for (int32_t i = 0; i < axis - 1; ++i) axes_out[i] = axes_in[i];

  if (axis > 0 && axis + 1 < src.NumAxes()) {
    axes_out[axis - 1].row_ids =
        axes_in[axis - 1].row_ids[axes_in[axis].row_ids];
    axes_out[axis - 1].row_splits =
        axes_in[axis].row_splits[axes_in[axis - 1].row_splits];
    axes_out[axis - 1].cached_tot_size = axes_out[axis - 1].row_ids.Dim();
  }
  for (int32_t i = axis; i < axes_out_size; ++i) axes_out[i] = axes_in[i + 1];
  return RaggedShape(axes_out);
}

RaggedShape MakeTransposable(RaggedShape &src) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  int32_t src_dim0 = src.Dim0(), src_tot_size1 = src.TotSize(1);
  if (src_dim0 <= 1) return src;

  ContextPtr c = src.Context();
  int32_t num_axes = src.NumAxes();
  int32_t max_size = src.MaxSize(1);
  if (max_size <= 0) return src;
  int32_t ans_tot_size1 = max_size * src_dim0;

  src.Populate();

  const std::vector<RaggedShapeLayer> &axes_in = src.Layers();
  std::vector<RaggedShapeLayer> axes_out(num_axes - 1);
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data();
  const int32_t *src_row_ids1_data = src.RowIds(1).Data();

  {
    ParallelRunner pr(c);

    RaggedShapeLayer &axis1_shape = axes_out[0];
    {
      // set ans.RowSplits(1);
      With w(pr.NewStream());
      axis1_shape.row_splits = Range(c, src_dim0 + 1, 0, max_size);
    }
    {
      // set ans.RowIds(1);
      With w(pr.NewStream());
      axis1_shape.row_ids = Array1<int32_t>(c, ans_tot_size1);
      int32_t *row_ids1_data = axis1_shape.row_ids.Data();
      axis1_shape.cached_tot_size = ans_tot_size1;
      K2_EVAL(
          c, ans_tot_size1, lambda_set_row_ids1,
          (int32_t i)->void { row_ids1_data[i] = i / max_size; });
    }
    if (num_axes > 2) {
      RaggedShapeLayer &axis2_shape = axes_out[1];
      const int32_t *src_row_splits2_data = src.RowSplits(2).Data();
      {
        // set ans.RowSplits(2);
        With w(pr.NewStream());
        axis2_shape.cached_tot_size = src.TotSize(2);
        axis2_shape.row_splits = Array1<int32_t>(c, ans_tot_size1 + 1);
        int32_t *ans_row_splits2_data = axis2_shape.row_splits.Data();
        K2_EVAL(
            c, ans_tot_size1 + 1, lambda_set_row_splits2,
            (int32_t idx01)->void {
              if (idx01 == ans_tot_size1) {
                ans_row_splits2_data[idx01] =
                    src_row_splits2_data[src_tot_size1];
                return;
              }
              int32_t idx0 = idx01 / max_size, idx1 = idx01 % max_size;
              int32_t idx0x = src_row_splits1_data[idx0],
                      idx0x_next = src_row_splits1_data[idx0 + 1];
              int32_t num_elems_this_row = idx0x_next - idx0x;
              if (idx1 < num_elems_this_row)
                ans_row_splits2_data[idx01] =
                    src_row_splits2_data[idx0x + idx1];
              else
                ans_row_splits2_data[idx01] =
                    src_row_splits2_data[idx0x_next];  // append empty row
            });
      }
      {
        // set ans.RowIds(2);
        With w(pr.NewStream());
        int32_t tot_size2 = src.TotSize(2);
        axis2_shape.row_ids = Array1<int32_t>(c, tot_size2);
        int32_t *ans_row_ids2_data = axis2_shape.row_ids.Data();
        const int32_t *src_row_ids2_data = src.RowIds(2).Data();
        K2_EVAL(
            c, tot_size2, lambda_set_row_ids2, (int32_t idx012)->void {
              int32_t src_idx01 = src_row_ids2_data[idx012];
              int32_t src_idx0 = src_row_ids1_data[src_idx01];
              int32_t src_idx1 = src_idx01 - src_row_splits1_data[src_idx0];
              ans_row_ids2_data[idx012] = (src_idx0 * max_size) + src_idx1;
            });
      }
    }
  }
  // copy left row_splits and row_ids;
  for (int32_t i = 2; i < num_axes - 1; ++i) axes_out[i] = axes_in[i];
  return RaggedShape(axes_out);
}

// transpose axes 0 and 1.
RaggedShape Transpose(RaggedShape &src, Array1<int32_t> *value_indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(src.NumAxes(), 2);
  ContextPtr c = src.Context();
  int32_t src_dim0 = src.Dim0(), src_tot_size1 = src.TotSize(1);
  if (src_dim0 <= 0) {
    if (value_indexes) *value_indexes = Array1<int32_t>(c, 0);
    return src;
  }
  int32_t src_dim1 = src_tot_size1 / src_dim0;
  K2_CHECK_EQ(src_tot_size1 % src_dim0, 0)
      << "Transpose(): all dims on axis 0 must be the same.\n"
      << "src_tot_size1: " << src_tot_size1 << "\n"
      << "src_dim0: " << src_dim0 << ", array is: " << src;
  K2_DCHECK(
      Equal(src.RowSplits(1), Range(c, src.RowSplits(1).Dim(), 0, src_dim1)))
      << " Expected row-splits to be evenly spaced: " << src.RowSplits(1);
  RaggedShape src_no_axis0 = RemoveAxis(src, 0);
  K2_CHECK_EQ(src_no_axis0.Dim0(), src_tot_size1);
  // `renumbering` is a `new2old` map, that maps from the first index in
  // src_no_axis0_renumbered
  // to the first index into src_no_axis0.
  Array1<int32_t> renumbering(c, src_tot_size1);
  int32_t *renumbering_data = renumbering.Data();
  K2_EVAL(
      c, src_tot_size1, lambda_set_renumbering, (int32_t i)->void {
        int32_t j = i % src_dim0, k = i / src_dim0, i_old = j * src_dim1 + k;
        renumbering_data[i] = i_old;
      });

  RaggedShape src_no_axis0_renumbered =
      Index(src_no_axis0, 0, renumbering, value_indexes);

  int32_t num_rows = src_dim1, row_splits_dim = num_rows + 1,
          row_ids_dim = src_tot_size1;
  std::vector<RaggedShapeLayer> ans_axis0(1);
  Array1<int32_t> mem(c, row_splits_dim + row_ids_dim);
  int32_t *mem_data = mem.Data();
  K2_EVAL(
      c, row_splits_dim + row_ids_dim, lambda_set_row_info, (int32_t i)->void {
        int32_t val;
        if (i >= row_splits_dim) {
          // row_ids
          int32_t elem_idx = i - row_splits_dim;
          val = elem_idx / src_dim0;
        } else {
          // row_splits
          int32_t row_idx = i;
          val = row_idx * src_dim0;
        }
        mem_data[i] = val;
      });
  ans_axis0[0].row_splits = mem.Range(0, row_splits_dim);
  ans_axis0[0].row_ids = mem.Range(row_splits_dim, row_ids_dim);
  ans_axis0[0].cached_tot_size = row_ids_dim;

  RaggedShape temp(ans_axis0);
  return ComposeRaggedShapes(temp, src_no_axis0_renumbered);
}

RaggedShape Stack(int32_t axis, int32_t num_srcs, RaggedShape **src,
                  Array1<uint32_t> *merge_map /* = nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK_LT(static_cast<uint32_t>(axis),
              static_cast<uint32_t>(src[0]->NumAxes()));
  ContextPtr c = src[0]->Context();

  if (axis == 0) {
    return StackAxis0(num_srcs, src, merge_map);
  }

  K2_CHECK_LT(static_cast<uint32_t>(axis),
              static_cast<uint32_t>(src[0]->NumAxes()));

  int32_t num_axes = src[0]->NumAxes();
  std::vector<RaggedShapeLayer> ans_layers(num_axes);

  // If axis >= 2, some layers of `src` will pass through unchanged (we should
  // check that they are identical across all sources).
  for (int32_t l = 0; l + 1 < axis; l++) {
    CheckLayerEqual(l, num_srcs, src);
    ans_layers[l] = src[0]->Layers()[l];
  }

  Array1<uint32_t> merge_map_local;
  Array1<uint32_t> *this_m =
      (axis + 1 == num_axes ? merge_map : &merge_map_local);
  RaggedShape s = IntersperseRaggedLayer(axis - 1, num_srcs, src, this_m);
  // note: s.Dim0() will be a multiple of num_srcs.
  ans_layers[axis - 1] =
      RegularRaggedShape(c, s.Dim0() / num_srcs, num_srcs).Layers()[0];
  ans_layers[axis] = s.Layers()[0];

  for (int32_t l = axis; l + 1 < num_axes; l++) {
    Array1<uint32_t> merge_map_next;
    Array1<uint32_t> *this_m =
        (l + 2 == num_axes ? merge_map : &merge_map_next);
    RaggedShape r = MergeRaggedLayer(l, num_srcs, src, merge_map_local, this_m);
    ans_layers[l + 1] = r.Layers()[0];
    merge_map_local = merge_map_next;
  }
  // TODO(dan) after this is debugged: add ", false".
  return RaggedShape(ans_layers);
}

/*
  Select ragged tensor's shape on axis 0 with a two axes ragged index.

    @param [in] src  Source RaggedShape to select.
    @param [in] indexes  A **TWO** axes ragged tensor containing the indexes
                         into the axis 0 of src. we also support -1 as an index,
                         which will result in the empty list (as if it were the
                         index into a position in `src` that had an empty list)
                         i.e. with `-1 <= indexes[i] < src.TotSize(0)`.
    @param [out] out  The container where the output RaggedShape will write to,
                      MUST NOT be a nullptr. Will be reallocated and the final
                      size of `out` would equal to `indexes.TotSize(0)`.
                      Note, The `NumAxes()` of output RaggedShape is the same
                      as the `NumAxes()` of src.
    @param [out] split_map  If not nullptr will store the element-index within
                            src telling where the elements of split RaggedShape
                            come from. Will be reallocated and the final size of
                            `split_map` would equal to `indexes.TotSize(0)`.

    Suppose indexes is `[ [ 0 3 5 ] [ 1 2 4] [ 6 -1 ] ]`, it means that we will
    select elements 0,3,5 of src's axis 0 to construct the first output
    RaggedShape, 1,2,4 to construct the second output RaggedShape, 6 and a empty
    list to construct the third output RaggedShape.
 */
/*static*/ void SelectAxis0(RaggedShape &src, const Ragged<int32_t> &indexes,
    std::vector<RaggedShape> *out, std::vector<Array1<int32_t>> *split_map) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  K2_CHECK(IsCompatible(src, indexes));
  K2_CHECK_EQ(indexes.NumAxes(), 2);
  K2_CHECK(out != nullptr);
  int32_t num_axes = src.NumAxes(),
          out_size = indexes.Dim0(),
          tot_elems = indexes.NumElements();
  if (out_size == 0) {
    *out = std::vector<RaggedShape>();
    if (split_map) {
      *split_map = std::vector<Array1<int32_t>>();
    }
    return;
  }

  Array2<int32_t> old_offsets,  // num_axes by tot_elems
      new_offsets;              // num_axes by (tot_elems + 1).
  GetOldAndNewOffsets(src, indexes.values, &old_offsets, &new_offsets);

  const int32_t *indexes_row_split1_data = indexes.RowSplits(1).Data(),
                *indexes_row_ids1_data = indexes.RowIds(1).Data();

  // Contains the `TotSize` of each axes of each output RaggedShape
  Array2<int32_t> tot_sizes(c, out_size, num_axes);
  Array2Accessor<int32_t> tot_sizes_acc = tot_sizes.Accessor();
  Array2Accessor<int32_t> new_offsets_acc = new_offsets.Accessor();

  K2_EVAL2(c, out_size, num_axes, lambda_set_tot_sizes,
      (int32_t i, int32_t j) -> void {
      int32_t idx0 = indexes_row_split1_data[i],
              idx0_next = indexes_row_split1_data[i + 1];
      tot_sizes_acc(i, j) =
          new_offsets_acc(j, idx0_next) - new_offsets_acc(j, idx0);
  });

  auto tot_sizes_cpu = tot_sizes.To(GetCpuContext());
  auto tot_sizes_cpu_acc = tot_sizes_cpu.Accessor();
  out->resize(out_size);
  if (split_map != nullptr) split_map->resize(out_size);
  // We can not avoid this for loop on dim0, as we want to allocate memory
  // seperately, may consider using a ThreadPool later.
  for (int32_t i = 0; i < out_size; ++i) {
    out->at(i) = RaggedShapeFromTotSizes(c,
          num_axes, tot_sizes_cpu.Row(i).Data());
    if (split_map != nullptr) {
      split_map->at(i) =
          Array1<int32_t>(c, tot_sizes_cpu_acc(i, num_axes - 1));
    };
  }

  // Caution: e.g. old_row_splits_acc(i) == src.RowSplits(i+1).
  RowSplitsAccessor<5> old_row_splits_acc(src);
  RowIdsAccessor<5> old_row_ids_acc(src);
  auto old_offsets_acc = old_offsets.Accessor();

  // axes_elems contains the elements number of each axes before splitting into
  // different RaggedShape, it should equal to the Col sum of `tot_sizes` above.
  Array1<int32_t> axes_elems =
      Array1<int32_t>(new_offsets.Col(tot_elems)).To(GetCpuContext());

  for (int32_t axis = 0; axis < num_axes; axis++) {
    // Contains the RowSplits & RowIds pointer for current layer,
    // has a dimension of dim0 * 2, the layout is splits_pointer0, ids_pointer0,
    // splits_pointer1, ids_pointer1, ...
    Array1<int32_t *> splits_ids_ptr(GetCpuContext(), out_size * 2);
    int32_t **splits_ids_ptr_data = splits_ids_ptr.Data();

    // Contains the pointers for split_map
    Array1<int32_t *> split_map_ptr;
    int32_t **split_map_ptr_data = nullptr;

    if (axis == num_axes - 1 && split_map != nullptr) {
      split_map_ptr = Array1<int32_t *>(GetCpuContext(), out_size);
      split_map_ptr_data = split_map_ptr.Data();
    }

    for (int32_t i = 0; i < out_size; ++i) {
      splits_ids_ptr_data[2 * i] = axis == num_axes - 1 ? nullptr :
        out->at(i).RowSplits(axis + 1).Data();

      splits_ids_ptr_data[2 * i + 1] =
        axis == 0 ? nullptr : out->at(i).RowIds(axis).Data();

      if (axis == num_axes - 1 && split_map != nullptr) {
        split_map_ptr_data[i] = split_map->at(i).Data();
      }
    }
    // transfer to GPU if we're using a GPU
    splits_ids_ptr = splits_ids_ptr.To(c);
    splits_ids_ptr_data = splits_ids_ptr.Data();

    // set row split1
    if (axis == 0) {
      K2_EVAL(c, tot_elems, lambda_set_row_split1, (int32_t idx01) {
          int32_t index_idx0 = indexes_row_ids1_data[idx01],
                  idx0x = indexes_row_split1_data[index_idx0];
          splits_ids_ptr_data[2 * index_idx0][idx01 - idx0x]
              = new_offsets_acc(axis + 1, idx01) -
                  new_offsets_acc(axis + 1, idx0x);

          // Set the last elements of row_splits1 of each output shape
          if (idx01 == tot_elems - 1 ||
              index_idx0 != indexes_row_ids1_data[idx01 + 1]) {
            splits_ids_ptr_data[2 * index_idx0][idx01 - idx0x + 1]
                = new_offsets_acc(axis + 1, idx01 + 1) -
                    new_offsets_acc(axis + 1, idx0x);
          }
      });
      continue;
    }

    // set last element of each row_splits
    // TODO: Integrate this kernel into the kernel below.
    if (axis < num_axes - 1) {
      K2_EVAL(c, out_size, lambda_set_last_row_splits, (int32_t idx0) {
          int32_t idx0x = indexes_row_split1_data[idx0],
                  idx0x_next = indexes_row_split1_data[idx0 + 1],
                  value = new_offsets_acc(axis + 1, idx0x_next) -
                            new_offsets_acc(axis + 1, idx0x),
                  pos = tot_sizes_acc(idx0, axis);
          splits_ids_ptr_data[2 * idx0][pos] = value;
      });
    }

    if (axis == num_axes - 1 && split_map != nullptr) {
      split_map_ptr = split_map_ptr.To(c);
      split_map_ptr_data = split_map_ptr.Data();
    }

    int32_t num_elems = axes_elems[axis];

    // composed_row_ids maps current idx to idx01 of indexes
    Array1<int32_t> composed_row_ids(c, num_elems);
    RowSplitsToRowIds(new_offsets.Row(axis), &composed_row_ids);

    const int32_t *composed_row_ids_data = composed_row_ids.Data();

    K2_EVAL(c, num_elems, lambda_set_row_splits_and_ids, (int32_t i) {
      // tot_elems = indexes.NumElements(), so tot_idx0 can be interpreted as
      // index_idx01
      int32_t tot_idx0 = composed_row_ids_data[i],
              index_idx0 = indexes_row_ids1_data[tot_idx0],
              index_idx0x = indexes_row_split1_data[index_idx0],

              begin_base = new_offsets_acc(axis, index_idx0x),
              begin = new_offsets_acc(axis, tot_idx0),
              this_idx0 = i - begin,
              this_idx01 = i - begin_base;

      K2_CHECK_GE(this_idx0, 0);
      K2_CHECK_GE(this_idx01, 0);

      // "prev" means for axis - 1
      int32_t new_prev_offset = new_offsets_acc(axis - 1, tot_idx0),
          old_prev_offset = old_offsets_acc(axis - 1, tot_idx0),
          old_offset = old_offsets_acc(axis, tot_idx0),
          old_idx = old_offset + this_idx0;

      if (split_map != nullptr && axis == num_axes - 1)
        split_map_ptr_data[index_idx0][this_idx01] = old_idx;

      // set row ids
      const int32_t *this_old_row_ids = old_row_ids_acc(axis - 1);
      int32_t old_row_id = this_old_row_ids[old_idx],
          new_row_id = old_row_id + new_prev_offset - old_prev_offset,
          new_pre_offset_idx0x = new_offsets_acc(axis - 1, index_idx0x);

      splits_ids_ptr_data[2 * index_idx0 + 1][this_idx01] =
          new_row_id - new_pre_offset_idx0x;

      // set row splits
      if (axis + 1 < num_axes) {
        int32_t new_next_offset = new_offsets_acc(axis + 1, tot_idx0),
                old_next_offset = old_offsets_acc(axis + 1, tot_idx0),
               next_offset_diff = new_next_offset - old_next_offset;
        const int32_t *old_row_splits_data = old_row_splits_acc(axis);
        int32_t row_split_value =
            next_offset_diff + old_row_splits_data[old_idx],
                new_next_offset_idx0x = new_offsets_acc(axis + 1, index_idx0x);
        splits_ids_ptr_data[2 * index_idx0][this_idx01]
            = row_split_value - new_next_offset_idx0x;
      }
    });
  }
}

void Unstack(RaggedShape &src, int32_t axis, bool pad_right,
             std::vector<RaggedShape> *out,
             std::vector<Array1<int32_t>> *split_map) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  if (axis == 0) {
    if (src.NumAxes() == 2) {
        auto new_src = ComposeRaggedShapes(
            TrivialShape(c, src.TotSize(0)), src);
        return Unstack(new_src, 1, pad_right, out, split_map);
    }
    auto indexes = Ragged<int32_t>(RegularRaggedShape(c, src.Dim0(), 1),
        Arange(c, 0, src.Dim0()));

    SelectAxis0(src, indexes, out, split_map);
    for (size_t i = 0; i < out->size(); ++i) {
      out->at(i) = RemoveAxis(out->at(i), 0);
    }
  } else {
    int32_t tot_size_axis_minus1 = src.TotSize(axis - 1),
            tot_size_axis = src.TotSize(axis);
    const int32_t *row_splits_axis = src.RowSplits(axis).Data(),
                  *row_ids_axis = src.RowIds(axis).Data();

    // Each sublist contains the elements of axis `axis`, unstack operation will
    // split all these elements in a sublist to different RaggedShapes, so the
    // number of output RaggedShapes is the size of the sublist with max
    // elements.
    int32_t num_out = src.MaxSize(axis);

    out->resize(num_out);
    if (split_map != nullptr) split_map->resize(num_out);

    // We will select the elements of axis `axis` on each sublist, the number
    // of sublits equals to `src.TotSize(axis - 1)`.
    // Initialize with -1 here, because not all the sublists have the same size,
    // -1s here mean that we don't select anything on those positions
    Array1<int32_t> indexes(c, num_out * tot_size_axis_minus1, -1);
    int32_t *indexes_data = indexes.Data();

    // Decide the elements of axis `axis` will go to which output RaggedShape
    K2_EVAL(c, tot_size_axis, lambda_set_indexes, (int32_t idx01) {
        int32_t idx0 = row_ids_axis[idx01],
                idx0x = row_splits_axis[idx0],
                idx1 = idx01 - idx0x,
                idx_row = idx1;
        if (!pad_right) {
          int32_t idx0x_next = row_splits_axis[idx0 + 1],
                  num_elems = idx0x_next - idx0x;
          idx_row = num_out - num_elems + idx1;
        }
        indexes_data[idx_row * tot_size_axis_minus1 + idx0] = idx01;
    });

    // To make `DecomposeRaggedShape` work, we add a RegularRaggedShape
    // layer after axis `axis` if axis equals to `src.NumAxes() - 1`.
    // Of course, we have to remove the added layer finally.
    bool remove_last_axis = false;
    if (axis == src.NumAxes() - 1) {
      src = ComposeRaggedShapes(src,
         RegularRaggedShape(c, src.NumElements(), 1));
      remove_last_axis = true;
    }

    RaggedShape top, bottom;
    DecomposeRaggedShape(src, axis, &top, &bottom);

    // Unstack will remove current axis (the last axis of top after decomposing
    // on axis), to make `RemoveAxis` work, we add a TrivialShape layer before
    // axix 0, finally we will remove the added layer.
    bool remove_axis0 = false;
    if (top.NumAxes() == 2) {
      top = ComposeRaggedShapes(
          TrivialShape(c, top.TotSize(0)), top);
      remove_axis0 = true;
    }
    top = RemoveAxis(top, top.NumAxes() - 1);

    auto ragged_indexes = Ragged<int32_t>(RegularRaggedShape(c,
          num_out, tot_size_axis_minus1), indexes);

    // Select elements according to indexes into corresponding RaggedShape
    SelectAxis0(bottom, ragged_indexes, out, split_map);

    for (int32_t i = 0; i < num_out; ++i) {
      out->at(i) = ComposeRaggedShapes(top, out->at(i));
      if (remove_axis0 && !remove_last_axis)
        out->at(i) = RemoveAxis(out->at(i), 0);
      if (remove_last_axis) {
        out->at(i) = RemoveEmptyLists(out->at(i), out->at(i).NumAxes() - 2);
        out->at(i) = RemoveAxis(out->at(i), out->at(i).NumAxes() - 1);
      }
    }
  }
}

void Unstack(RaggedShape &src, int32_t axis, std::vector<RaggedShape> *out,
             std::vector<Array1<int32_t>> *split_map /*= nullptr*/) {
  Unstack(src, axis, true/*pad_right*/, out, split_map);
}

RaggedShape Merge(int32_t num_srcs, RaggedShape **src,
                  const Array1<uint32_t> &merge_map,
                  Array1<uint32_t> *merge_map_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(num_srcs > 0);
  int32_t num_layers = src[0]->NumAxes() - 1;

  std::vector<RaggedShapeLayer> ans_layers(num_layers);

  // Note: this is a shallow copy.
  Array1<uint32_t> merge_map_local = merge_map;

  for (int32_t l = 0; l < num_layers; l++) {
    Array1<uint32_t> merge_map_next;
    Array1<uint32_t> *this_m =
        (l + 1 == num_layers ? merge_map_out : &merge_map_next);
    RaggedShape r = MergeRaggedLayer(l, num_srcs, src, merge_map_local, this_m);
    ans_layers[l] = r.Layers()[0];
    merge_map_local = merge_map_next;
  }
  // TODO(dan) after this is debugged: add ", false".
  return RaggedShape(ans_layers);
}

RaggedShape TrivialShape(ContextPtr &c, int32_t num_elems) {
  NVTX_RANGE(K2_FUNC);
  // row_splits= [
  Array1<int32_t> row_splits = Range<int32_t>(c, 2, 0, num_elems);
  Array1<int32_t> row_ids(c, num_elems, 0);
  return RaggedShape2(&row_splits, &row_ids, num_elems);
}

RaggedShape RegularRaggedShape(ContextPtr &c, int32_t dim0, int32_t dim1) {
  NVTX_RANGE(K2_FUNC);
  Array1<int32_t> row_splits = Range<int32_t>(c, dim0 + 1, 0, dim1);
  Array1<int32_t> row_ids(c, dim0 * dim1);
  int32_t *row_ids_data = row_ids.Data();
  K2_EVAL2(
      c, dim0, dim1, lambda_set_row_ids,
      (int32_t i, int32_t j)->void { row_ids_data[i * dim1 + j] = i; });
  return RaggedShape2(&row_splits, &row_ids, dim0 * dim1);
}

Ragged<int32_t> GetCountsPartitioned(Ragged<int32_t> &src,
                                     RaggedShape &ans_ragged_shape) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 2);
  K2_CHECK_EQ(ans_ragged_shape.NumAxes(), 2);
  K2_CHECK(IsCompatible(src, ans_ragged_shape));
  K2_CHECK_EQ(src.Dim0(), ans_ragged_shape.Dim0());
  const Array1<int32_t> &values = src.values;
  const Array1<int32_t> &row_splits = ans_ragged_shape.RowSplits(1);
  int32_t n = ans_ragged_shape.NumElements();
  Array1<int32_t> counts = GetCounts(values, n);
  return Ragged<int32_t>(ans_ragged_shape, counts);
}

/*static*/ Array1<int32_t> GetTransposeReorderingCpu(Ragged<int32_t> &src,
                                                     int32_t num_cols) {
  NVTX_RANGE(K2_FUNC);
  std::vector<std::vector<int32_t>> column_indexes(num_cols);  // [column][row]
  const int32_t *values_data = src.values.Data();
  int32_t n = src.values.Dim();

  for (int32_t i = 0; i != n; ++i) {
    int32_t bucket = values_data[i];
    column_indexes[bucket].push_back(i);
  }

  Array1<int32_t> ans(src.Context(), n);
  int32_t *ans_data = ans.Data();
  for (int32_t i = 0; i != num_cols; ++i) {
    std::copy(column_indexes[i].begin(), column_indexes[i].end(), ans_data);
    ans_data += column_indexes[i].size();
  }
  return ans;
}

#ifndef _MSC_VER
/*static*/ Array1<int32_t> GetTransposeReorderingThreeAxesCuda(
    Ragged<int32_t> &src, int32_t num_cols) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 3);
  ContextPtr &context = src.Context();
  K2_CHECK_EQ(context->GetDeviceType(), kCuda);

  const Array1<int32_t> &row_splits1 = src.RowSplits(1);
  const int32_t *row_ids2_data = src.RowIds(2).Data();
  const int32_t *value_data = src.values.Data();
  Array1<int32_t> segments = src.RowSplits(2)[row_splits1];

  auto lambda_comp = [=] __device__(int32_t a_idx012,
                                    int32_t b_idx012) -> bool {
    int32_t a_col_index = value_data[a_idx012];
    int32_t b_col_index = value_data[b_idx012];

    if (a_col_index < b_col_index) return true;  // sort by column indexes
    if (a_col_index > b_col_index) return false;

    // at this point, a_idx012 and b_idx012 belong to the same column;
    // then we sort by its row indexes

    int32_t a_idx01 = row_ids2_data[a_idx012];
    int32_t b_idx01 = row_ids2_data[b_idx012];

    if (a_idx01 < b_idx01) return true;
    if (a_idx01 > b_idx01) return false;

    // at this point, a_idx012 and b_idx012 are duplicate elements
    return false;  // either true or false is fine
  };

  mgpu::context_t *mgpu_context = GetModernGpuAllocator(context);

  int32_t n = src.values.Dim();
  Array1<int32_t> ans = Range(context, n, 0);
  if (n == 0) return ans;
  K2_CUDA_SAFE_CALL(mgpu::segmented_sort(ans.Data(),          // keys
                                         ans.Dim(),           // count
                                         segments.Data(),     // segments
                                         segments.Dim() - 1,  // num_segments
                                         lambda_comp, *mgpu_context));
  return ans;
}
#endif


/*
// Checks the result of GetTranspoeReordering(), in debug mode and dies if it is wrong.
static void CheckGetTransposeReordering(Ragged<int32_t> &src,
                                        Array1<int32_t> &ans) {
  if (!internal::kDisableDebug && !internal::DisableChecks()) {
    K2_CHECK(IsPermutation(ans));
    K2_CHECK(IsMonotonic(src.values[ans]));
  }
  }*/

Array1<int32_t> GetTransposeReordering(Ragged<int32_t> &src, int32_t num_cols) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &context = src.Context();
  if (src.NumAxes() < 2 || src.values.Dim() == 0) {
    // src is empty
    return Array1<int32_t>(context, 0);
  }

  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCpu) return GetTransposeReorderingCpu(src, num_cols);

  K2_CHECK_EQ(device_type, kCuda);

#ifdef _MSC_VER
  // See https://github.com/k2-fsa/k2/pull/753
  // and
  // https://github.com/k2-fsa/k2/pull/571
  int32_t num_buckets = num_cols;
  int32_t num_elements = src.values.Dim();
  int32_t log_buckets = static_cast<int32_t>(ceilf(log2f(num_buckets)));

  Array1<int32_t> ans = Range(context, num_elements, 0);

  cudaStream_t stream = context->GetCudaStream();

  size_t temp_storage_bytes = 0;
  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, src.values.Data(),
      static_cast<int32_t *>(nullptr), ans.Data(), ans.Data(), num_elements, 0,
      log_buckets, stream));

  Array1<int8_t> d_temp_storage(
      context, temp_storage_bytes + num_elements * sizeof(int32_t));

  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      d_temp_storage.Data() + sizeof(int32_t) * num_elements,
      temp_storage_bytes, src.values.Data(),
      reinterpret_cast<int32_t *>(d_temp_storage.Data()), ans.Data(),
      ans.Data(), num_elements, 0, log_buckets, stream));

  return ans;

#else
  (void)GetTransposeReorderingThreeAxesCuda;  // remove compiler warnings

#if __CUDACC_VER_MAJOR__ > 10 ||   \
    (__CUDACC_VER_MAJOR__ == 10 && \
     (__CUDACC_VER_MINOR__ > 1 ||  \
      (__CUDACC_VER_MINOR__ == 1 && __CUDACC_VER_BUILD__ > 105)))
  // Enable it only for NVCC > 10.1.105
  //
  // Refer to https://github.com/LLNL/axom/issues/88
  // NVCC 10.1.105 has a known issue for cub::DeviceRadixSort
  int32_t num_buckets = num_cols;
  int32_t num_elements = src.values.Dim();
  int32_t log_buckets = static_cast<int32_t>(ceilf(log2f(num_buckets)));

  Array1<int32_t> order = Range(context, num_elements, 0);
  Array1<int32_t> src_tmp_out(context, num_elements);
  Array1<int32_t> ans(context, num_elements);

  cudaStream_t stream = context->GetCudaStream();

  size_t temp_storage_bytes = 0;
  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, src.values.Data(), src_tmp_out.Data(),
      order.Data(), ans.Data(), num_elements, 0, log_buckets, stream));

  Array1<int8_t> d_temp_storage(context, temp_storage_bytes);

  K2_CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
      d_temp_storage.Data(), temp_storage_bytes, src.values.Data(),
      src_tmp_out.Data(), order.Data(), ans.Data(), num_elements, 0,
      log_buckets, stream));

  // CheckGetTransposeReordering(src, ans);
  return ans;
#else  // __CUDACC_VER_MAJOR__
  if (src.NumAxes() == 3) {
    Array1<int32_t> ans = GetTransposeReorderingThreeAxesCuda(src, num_cols);
    // CheckGetTransposeReordering(src, ans);
    return ans;
  }

  const int32_t *row_splits1_data = src.RowSplits(src.NumAxes() - 1).Data();
  const int32_t *row_ids1_data = src.RowIds(src.NumAxes() - 1).Data();
  const int32_t *value_data = src.values.Data();
  int32_t n = src.values.Dim();
  Array1<int32_t> ans = Range(context, n, 0);
  if (n == 0) return ans;

  auto lambda_comp = [=] __device__(int32_t a_idx01, int32_t b_idx01) -> bool {
    int32_t a_idx0 = row_ids1_data[a_idx01];
    int32_t b_idx0 = row_ids1_data[b_idx01];

    int32_t a_col_index = value_data[a_idx01];
    int32_t b_col_index = value_data[b_idx01];

    if (a_col_index < b_col_index) return true;  // sort by column indexes
    if (a_col_index > b_col_index) return false;

    // now we have a_col_index == b_col_index
    if (a_idx0 < b_idx0) return true;  // sort by row indexes
    if (a_idx0 > b_idx0) return false;

    // now we have a_idx0 == b_idx0 && a_col_index == b_col_index
    // this entry is duplicated in the sparse matrix.
    return false;  // we can return either true or false here.
  };

  mgpu::context_t *mgpu_context = GetModernGpuAllocator(context);

  K2_CUDA_SAFE_CALL(mgpu::mergesort(ans.Data(), n, lambda_comp, *mgpu_context));
  // CheckGetTransposeReordering(src, ans);
  return ans;
#endif
#endif  // _MSC_VER
}

RaggedShape ChangeSublistSize(const RaggedShape &src, int32_t size_delta) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  // the result will have the same num-axes as `src` (the NumAxes() of the
  // object is not the same as the number of RaggedShapeLayer axes).
  std::vector<RaggedShapeLayer> ans_axes(src.NumAxes() - 1);
  int32_t last_axis = src.NumAxes() - 1;
  // The following will only do something if src.NumAxes() > 2.
  for (int32_t i = 0; i + 1 < last_axis; ++i) ans_axes[i] = src.Layers()[i];

  ContextPtr &c = src.Context();
  int32_t num_rows = src.TotSize(last_axis - 1),
          src_num_elems = src.TotSize(last_axis),
          num_elems = src_num_elems + size_delta * num_rows;
  ans_axes.back().row_splits = Array1<int32_t>(c, num_rows + 1);
  ans_axes.back().row_ids = Array1<int32_t>(c, num_elems);
  ans_axes.back().cached_tot_size = num_elems;
  const int32_t *src_row_splits_data = src.RowSplits(last_axis).Data(),
                *src_row_ids_data = src.RowIds(last_axis).Data();
  int32_t *row_splits_data = ans_axes.back().row_splits.Data(),
          *row_ids_data = ans_axes.back().row_ids.Data();

  {
    ParallelRunner pr(c);
    {
      With w(pr.NewStream());
      K2_EVAL(
          c, num_rows + 1, lambda_set_row_splits, (int32_t idx0)->void {
            row_splits_data[idx0] =
                src_row_splits_data[idx0] + size_delta * idx0;
          });
    }

    {
      With w(pr.NewStream());
      K2_EVAL(
          c, src_num_elems, lambda_set_row_ids1, (int32_t src_idx01)->void {
            int32_t src_idx0 = src_row_ids_data[src_idx01],
                    src_idx0x = src_row_splits_data[src_idx0],
                    src_idx1 = src_idx01 - src_idx0x,
                    new_idx0x = row_splits_data[src_idx0],
                    new_idx0x_next = row_splits_data[src_idx0 + 1],
                    new_idx01 = new_idx0x + src_idx1;
            // it's only necessary to guard the next statement with in 'if'
            // because size_delta might be negative.
            if (new_idx01 < new_idx0x_next) row_ids_data[new_idx01] = src_idx0;
          });
    }
    if (size_delta > 0) {
      // This sets the row-ids that are not set by lambda_set_row_ids1.
      With w(pr.NewStream());
      K2_EVAL(
          c, num_rows * size_delta, lambda_set_row_ids2, (int32_t i)->void {
            int32_t idx0 = i / size_delta, n = i % size_delta,
                    next_idx0 = idx0 + 1;
            // The following formula is the same as the one in
            // lambda_set_row_splits; we want to compute the new value of
            // row_splits_data[next_idx0] without waiting for that kernel to
            // terminate.
            int32_t next_idx0x =
                src_row_splits_data[next_idx0] + size_delta * next_idx0;
            row_ids_data[next_idx0x - 1 - n] = idx0;
          });
    }
    // make the ParallelRunner go out of scope (should do this before any
    // validation code that gets invoked by the constructor of RaggedShape
    // below).
  }
  return RaggedShape(ans_axes);
}

// TODO(dan): this could definitely be made more efficient.
RaggedShape ChangeSublistSizePinned(RaggedShape &src, int32_t size_delta) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);

  // the result will have the same num-axes as `src` (the NumAxes() of the
  // object is not the same as the number of RaggedShapeLayer axes).
  std::vector<RaggedShapeLayer> ans_axes(src.NumAxes() - 1);
  int32_t last_axis = src.NumAxes() - 1;
  // The following will only do something if src.NumAxes() > 2.
  for (int32_t i = 0; i + 1 < last_axis; ++i) ans_axes[i] = src.Layers()[i];

  ContextPtr &c = src.Context();

  int32_t num_rows = src.TotSize(last_axis - 1);
  ans_axes.back().row_splits = Array1<int32_t>(c, num_rows + 1);

  const int32_t *src_row_splits_data = src.RowSplits(last_axis).Data();
  int32_t *row_splits_data = ans_axes.back().row_splits.Data();

  K2_EVAL(
      c, num_rows, lambda_set_row_sizes, (int32_t idx0)->void {
        int32_t orig_size =
                    src_row_splits_data[idx0 + 1] - src_row_splits_data[idx0],
                size;
        if (orig_size == 0 || orig_size + size_delta <= 0)
          size = 0;
        else
          size = orig_size + size_delta;
        row_splits_data[idx0] = size;
      });
  ExclusiveSum(ans_axes.back().row_splits, &ans_axes.back().row_splits);
  ans_axes.back().row_ids =
      Array1<int32_t>(c, ans_axes.back().row_splits.Back());
  RowSplitsToRowIds(ans_axes.back().row_splits, &ans_axes.back().row_ids);
  ans_axes.back().cached_tot_size = ans_axes.back().row_ids.Dim();
  return RaggedShape(ans_axes);
}

RaggedShape Prefix(RaggedShape &src, int32_t n) {
  NVTX_RANGE(K2_FUNC);
  int32_t dim0 = src.Dim0();
  K2_CHECK(n >= 0 && n <= dim0);

  src.Populate();
  int32_t num_axes = src.NumAxes();
  K2_CHECK_GE(num_axes, 2);
  const std::vector<RaggedShapeLayer> &axes_in = src.Layers();
  std::vector<RaggedShapeLayer> axes_out(axes_in.size());

  int32_t row_end = n;
  for (int32_t axis = 0; axis < num_axes - 1; ++axis) {
    axes_out[axis].row_splits = axes_in[axis].row_splits.Arange(0, row_end + 1);
    // notice here we may do a memory copy from GPU to CPU.
    row_end = axes_in[axis].row_splits[row_end];
    axes_out[axis].row_ids = axes_in[axis].row_ids.Arange(0, row_end);
    axes_out[axis].cached_tot_size = row_end;
  }
  return RaggedShape(axes_out);
}

std::vector<RaggedShape> GetPrefixes(RaggedShape &src,
                                     const std::vector<int32_t> &sizes) {
  NVTX_RANGE(K2_FUNC);
  src.Populate();
  int32_t dim0 = src.Dim0();
  int32_t num_axes = src.NumAxes();
  K2_CHECK_GE(num_axes, 2);
  ContextPtr &c = src.Context();
  const std::vector<RaggedShapeLayer> &axes_in = src.Layers();

  // get those row_end elements at each axis.
  int32_t ans_size = static_cast<int32_t>(sizes.size());
  Array1<int32_t> row_ends(c, num_axes * ans_size);
  Array1<int32_t> sizes_array(GetCpuContext(), sizes);
  Array1<int32_t> indexes = row_ends.Arange(0, ans_size);
  indexes.CopyFrom(sizes_array);
  for (int32_t axis = 1; axis < num_axes; ++axis) {
    Array1<int32_t> curr_axis_row_ends =
        row_ends.Arange(axis * ans_size, (axis + 1) * ans_size);
    axes_in[axis - 1].row_splits.Index(indexes, &curr_axis_row_ends);
    indexes = curr_axis_row_ends;
  }

  row_ends = row_ends.To(GetCpuContext());
  std::vector<RaggedShape> ans(ans_size);
  for (int32_t i = 0; i != ans_size; ++i) {
    std::vector<RaggedShapeLayer> axes_out(axes_in.size());
    int32_t row_end = row_ends[i];
    K2_CHECK(row_end >= 0 && row_end <= dim0);
    for (int32_t axis = 0; axis < num_axes - 1; ++axis) {
      axes_out[axis].row_splits =
          axes_in[axis].row_splits.Arange(0, row_end + 1);
      row_end = row_ends[i + (axis + 1) * ans_size];
      axes_out[axis].row_ids = axes_in[axis].row_ids.Arange(0, row_end);
      axes_out[axis].cached_tot_size = row_end;
    }
    ans[i] = RaggedShape(axes_out, false);
  }
  return ans;
}

RaggedShape Arange(RaggedShape &src, int32_t axis, int32_t begin, int32_t end,
                   std::pair<int32_t, int32_t> *value_range /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  K2_CHECK_GE(num_axes, 2);
  K2_CHECK(axis >= 0 && axis < num_axes - 1);
  K2_CHECK(begin >= 0 && begin <= end && end <= src.TotSize(axis));

  if (begin == end) {
    RaggedShape ans = EmptyRaggedShape(src.Context(), num_axes - axis);
    // as begin == end, user always get empty values when doing
    // `src.values.Arange(begin, end)`
    if (value_range != nullptr) *value_range = std::make_pair(begin, end);
    return ans;
  }

  src.Populate();
  ContextPtr &c = src.Context();
  const std::vector<RaggedShapeLayer> &axes_in = src.Layers();
  int32_t ans_num_axes = num_axes - axis;
  // `-1` as Layers().size is NumAxes() - 1
  std::vector<RaggedShapeLayer> axes_out(ans_num_axes - 1);

  // get those `row_begin` and `row_end` indexes for all axes in a kernel so we
  // can do just one GPU to CPU memory transfer.
  // the format of `indexes` is: row_begin_axis0, row_end_axis0,
  // row_begin_axis1, row_end_axis2, etc. axis0, axis1 here are the axis of ans.
  Array1<int32_t> indexes(c, ans_num_axes * 2);
  int32_t *indexes_data = indexes.Data();
  RowSplitsAccessor<5> src_row_splits_acc(src);

  K2_EVAL(
      c, 1, lambda_set_indexes, (int32_t i)->void {
        // we just start a kernel with only one element here.
        K2_CHECK_EQ(i, 0);
        int32_t row_begin = begin, row_end = end;
        indexes_data[0] = row_begin, indexes_data[1] = row_end;
        for (int32_t cur_axis = axis; cur_axis < num_axes - 1; ++cur_axis) {
          row_begin = src_row_splits_acc(cur_axis)[row_begin];
          row_end = src_row_splits_acc(cur_axis)[row_end];
          int32_t indexes_pos = ((cur_axis - axis) + 1) * 2;
          indexes_data[indexes_pos] = row_begin;
          indexes_data[indexes_pos + 1] = row_end;
        }
      });
  indexes = indexes.To(GetCpuContext());

  int32_t row_begin = indexes[0], row_end = indexes[1];
  for (int32_t cur_axis = axis; cur_axis < num_axes - 1; ++cur_axis) {
    axes_out[cur_axis - axis].row_splits =
        axes_in[cur_axis].row_splits.Arange(row_begin, row_end + 1);
    int32_t row_id = row_begin;
    int32_t indexes_pos = ((cur_axis - axis) + 1) * 2;
    row_begin = indexes[indexes_pos];
    row_end = indexes[indexes_pos + 1];
    axes_out[cur_axis - axis].row_splits =
        Minus(axes_out[cur_axis - axis].row_splits, row_begin);
    axes_out[cur_axis - axis].row_ids =
        axes_in[cur_axis].row_ids.Arange(row_begin, row_end);
    axes_out[cur_axis - axis].row_ids =
        Minus(axes_out[cur_axis - axis].row_ids, row_id);
    axes_out[cur_axis - axis].cached_tot_size = row_end - row_begin;
  }
  if (value_range != nullptr) *value_range = std::make_pair(row_begin, row_end);
  return RaggedShape(axes_out);
}

Ragged<int32_t> AddSuffixToRagged(const Ragged<int32_t> &src,
                                  const Array1<int32_t> &suffix) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  K2_CHECK_GE(num_axes, 2);
  K2_CHECK_EQ(suffix.Dim(), src.TotSize(num_axes - 2));
  ContextPtr &c = src.Context();
  Array1<int32_t> dst_values(c, src.NumElements() + suffix.Dim());
  RaggedShape dst_shape = ChangeSublistSize(src.shape, 1);
  // "row_splits1" and "row_ids1" below are actually on the last axis. We name
  // them with "1" so that we can use "idx01" and "idx0" for those indexes in
  // lambda, following the naming convention explained in k2/csrc/utils.h
  const int32_t *dst_row_splits1_data =
                    dst_shape.RowSplits(num_axes - 1).Data(),
                *dst_row_ids1_data = dst_shape.RowIds(num_axes - 1).Data(),
                *src_values_data = src.values.Data(),
                *suffix_data = suffix.Data();
  int32_t *dst_values_data = dst_values.Data();

  K2_EVAL(
      c, dst_shape.NumElements(), lambda_copy_values, (int32_t idx01)->void {
        int32_t idx0 = dst_row_ids1_data[idx01];
        if (idx01 == dst_row_splits1_data[idx0 + 1] - 1) {
          // idx01 points to the last element of this row; copy from suffix
          dst_values_data[idx01] = suffix_data[idx0];
        } else {
          // copy from src
          int32_t src_idx01 = idx01 - dst_row_ids1_data[idx01];
          dst_values_data[idx01] = src_values_data[src_idx01];
        }
      });

  return Ragged<int32_t>(dst_shape, dst_values);
}

Ragged<int32_t> AddPrefixToRagged(const Ragged<int32_t> &src,
                                  const Array1<int32_t> &prefix) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  K2_CHECK_GE(num_axes, 2);
  K2_CHECK_EQ(prefix.Dim(), src.TotSize(num_axes - 2));
  ContextPtr &c = src.Context();
  Array1<int32_t> dst_values(c, src.NumElements() + prefix.Dim());
  RaggedShape dst_shape = ChangeSublistSize(src.shape, 1);
  // "row_splits1" and "row_ids1" below are actually on the last axis. We name
  // them with "1" so that we can use "idx01" and "idx0" for those indexes in
  // lambda, following the naming convention explained in k2/csrc/utils.h
  const int32_t *dst_row_splits1_data =
                    dst_shape.RowSplits(num_axes - 1).Data(),
                *dst_row_ids1_data = dst_shape.RowIds(num_axes - 1).Data(),
                *src_values_data = src.values.Data(),
                *prefix_data = prefix.Data();
  int32_t *dst_values_data = dst_values.Data();

  K2_EVAL(
      c, dst_shape.NumElements(), lambda_copy_values, (int32_t idx01)->void {
        int32_t idx0 = dst_row_ids1_data[idx01];
        if (idx01 == dst_row_splits1_data[idx0]) {
          // idx01 points to the first element of this row; copy from prefix
          dst_values_data[idx01] = prefix_data[idx0];
        } else {
          // copy from src
          int32_t src_idx01 = idx01 - dst_row_ids1_data[idx01] - 1;
          dst_values_data[idx01] = src_values_data[src_idx01];
        }
      });

  return Ragged<int32_t>(dst_shape, dst_values);
}

RaggedShape SubsetRaggedShape(RaggedShape &src, Renumbering &renumbering,
                              int32_t axis, Array1<int32_t> *elems_new2old) {
  NVTX_RANGE(K2_FUNC);
  axis = axis < 0 ? src.NumAxes() + axis : axis;
  K2_CHECK_EQ(renumbering.NumOldElems(), src.TotSize(axis));
  return Index(src, axis, renumbering.New2Old(), elems_new2old);
}

RaggedShape SubsetRaggedShape(RaggedShape &src, Renumbering &r_before_last,
                              Renumbering &r_last) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(r_before_last.NumOldElems(), src.TotSize(src.NumAxes() - 2));
  K2_CHECK_EQ(r_last.NumOldElems(), src.NumElements());

  // Make sure final and before-final row-ids are populated.
  src.RowIds(src.NumAxes() - 2);
  src.RowIds(src.NumAxes() - 1);
  std::vector<RaggedShapeLayer> axes = src.Layers();

  // Suppose this shape has 3 axes (0,1,2).  Its NumAxes()==3;
  // axes.size()==2.
  // r_before_last deals with the numbering on axis 1.
  // r_last deals with the numbering on axis 2.

  RaggedShapeLayer &before_last = axes[axes.size() - 2],
                   &last = axes[axes.size() - 1];

  int32_t new_tot_size1 = r_before_last.NumNewElems(),
          new_tot_size2 = r_last.NumNewElems();

  ContextPtr c = src.Context();
  Array1<int32_t> before_last_row_ids(c, new_tot_size1),
      last_row_splits(c, new_tot_size1 + 1), last_row_ids(c, new_tot_size2);

  // The variable names below use this 3-axis assumption but the
  // code will work for greater number of axes.
  int32_t *new_row_ids1_data = before_last_row_ids.Data(),
          *new_row_splits2_data = last_row_splits.Data(),
          *new_row_ids2_data = last_row_ids.Data();

  const int32_t *old_row_ids1_data = before_last.row_ids.Data(),
                *old_row_splits2_data = last.row_splits.Data(),
                *old_row_ids2_data = last.row_ids.Data();

  const int32_t *idx01_new2old_data = r_before_last.New2Old().Data(),
                *idx01_old2new_data = r_before_last.Old2New().Data(),
                *idx012_new2old_data = r_last.New2Old().Data(),
                *idx012_old2new_data = r_last.Old2New().Data();

  ParallelRunner pr(c);
  {
    With w(pr.NewStream());
    // before_last.row_splits maps from idx0 -> idx01 (contains idx01's).  Map
    // the idx01's; the idx0s stay the same.
    before_last.row_splits = r_before_last.Old2New()[before_last.row_splits];
  }
  {
    With w(pr.NewStream());
    K2_EVAL(
        c, new_tot_size1 + 1, lambda_set_row_ids1_and_row_splits2,
        (int32_t new_idx01)->void {
          // row_ids1 maps from idx01 -> idx0.  Select subset of
          // idx01's; the idx0 stays the same.
          int32_t old_idx01 = idx01_new2old_data[new_idx01];
          if (new_idx01 < new_tot_size1)
            new_row_ids1_data[new_idx01] = old_row_ids1_data[old_idx01];
          // row_splits2 maps from idx01 -> idx012.  Map both indexes.
          // idx01's; the idx0 stays the same.
          new_row_splits2_data[new_idx01] =
              idx012_old2new_data[old_row_splits2_data[old_idx01]];
        });
  }

  {
    With w(pr.NewStream());
    K2_EVAL(
        c, new_tot_size2, lambda_set_row_ids2, (int32_t new_idx012)->void {
          // row_ids2 maps from idx012 -> idx01.  Both must be mapped.

          int32_t old_idx012 = idx012_new2old_data[new_idx012];
          int32_t old_idx01 = old_row_ids2_data[old_idx012],
                  new_idx01 = idx01_old2new_data[old_idx01];
          new_row_ids2_data[new_idx012] = new_idx01;
        });
  }

  before_last.row_ids = before_last_row_ids;
  before_last.cached_tot_size = new_tot_size1;
  last.row_splits = last_row_splits;
  last.row_ids = last_row_ids;
  last.cached_tot_size = new_tot_size2;
  return RaggedShape(axes);
}

RaggedShape EmptyRaggedShape(ContextPtr &c, int32_t num_axes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(num_axes, 2);
  std::vector<RaggedShapeLayer> axes(num_axes - 1);
  axes[0].row_splits = Array1<int32_t>(c, 1, 0);
  // row_ids will be the empty vector, with context `c`.
  axes[0].row_ids = axes[0].row_splits.Range(0, 0);
  axes[0].cached_tot_size = 0;
  for (int32_t a = 1; a + 1 < num_axes; ++a) axes[a] = axes[0];
  return RaggedShape(axes);
}

Array1<int32_t> GetDecreasingSizeOrder(RaggedShape &shape) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = shape.Context();

  Array1<int32_t> sizes = RowSplitsToSizes(shape.RowSplits(1));
  Array1<int32_t> index_map;
  Sort<int32_t, GreaterThan<int32_t>>(&sizes, &index_map);
  return index_map;
}

RaggedShape GetLayer(const RaggedShape &src, int32_t layer) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(layer, 0);
  K2_CHECK_LT(layer, src.NumAxes() - 1);
  std::vector<RaggedShapeLayer> layers;
  layers.push_back(src.Layers()[layer]);
  bool check = false;
  return RaggedShape(layers, check);
}

void DecomposeRaggedShape(const RaggedShape &src, int32_t axis,
                          RaggedShape *top, RaggedShape *bottom) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(axis, 0);
  K2_CHECK_LT(axis, src.NumAxes() - 1);
  const std::vector<RaggedShapeLayer> &src_layers = src.Layers();
  std::vector<RaggedShapeLayer> top_layers(axis),
      bottom_layers(src_layers.size() - axis);
  int32_t src_size = static_cast<int32_t>(src_layers.size());
  for (int32_t i = 0; i < axis; ++i) top_layers[i] = src_layers[i];
  for (int32_t i = axis; i < src_size; ++i)
    bottom_layers[i - axis] = src_layers[i];
  *top = RaggedShape(top_layers);
  *bottom = RaggedShape(bottom_layers);
}

RaggedShape RemoveEmptyLists(RaggedShape &src_shape, int32_t axis,
                             Renumbering *renumbering_out) {
  NVTX_RANGE(K2_FUNC);
  if (axis == 0) {
    return RemoveEmptyListsAxis0(src_shape, renumbering_out);
  }
  RaggedShape top_shape, bottom_shape;
  DecomposeRaggedShape(src_shape, axis, &top_shape, &bottom_shape);

  Renumbering r_temp;
  if (!renumbering_out) renumbering_out = &r_temp;
  bottom_shape = RemoveEmptyListsAxis0(bottom_shape, renumbering_out);
  top_shape = SubsetRaggedShape(top_shape, *renumbering_out);
  return ComposeRaggedShapes(top_shape, bottom_shape);
}

RaggedShape RemoveSomeEmptyLists(RaggedShape &src_shape, int32_t axis,
                                 Renumbering &renumbering) {
  NVTX_RANGE(K2_FUNC);
  if (axis == 0) {
    return RenumberAxis0Simple(src_shape, renumbering);
  }
  RaggedShape top_shape, bottom_shape;
  DecomposeRaggedShape(src_shape, axis, &top_shape, &bottom_shape);

  bottom_shape = RenumberAxis0Simple(bottom_shape, renumbering);
  top_shape = SubsetRaggedShape(top_shape, renumbering);
  return ComposeRaggedShapes(top_shape, bottom_shape);
}

RaggedShape RemoveEmptyListsAxis0(RaggedShape &src_shape,
                                  Renumbering *renumbering_out) {
  NVTX_RANGE(K2_FUNC);
  Renumbering r_temp;
  if (!renumbering_out) renumbering_out = &r_temp;

  ContextPtr &c = src_shape.Context();
  int32_t num_lists = src_shape.Dim0();
  *renumbering_out = Renumbering(c, num_lists);
  const int32_t *row_splits_data = src_shape.RowSplits(1).Data();
  char *keep_data = renumbering_out->Keep().Data();
  K2_EVAL(
      c, num_lists, lambda_set_keep, (int32_t i)->void {
        keep_data[i] = (row_splits_data[i + 1] != row_splits_data[i]);
      });
  return RenumberAxis0Simple(src_shape, *renumbering_out);
}

RaggedShape RenumberAxis0Simple(RaggedShape &src_shape,
                                Renumbering &renumbering) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(renumbering.NumOldElems(), src_shape.Dim0());
  ContextPtr c = src_shape.Context();
  src_shape.RowIds(1);  // make sure RowIds(1) is populated.
  std::vector<RaggedShapeLayer> layers = src_shape.Layers();
  int32_t num_layers = layers.size();
  int32_t new_num_lists = renumbering.NumNewElems(),
          num_elems = src_shape.TotSize(1);  // unchanged old vs. new.
  Array1<int32_t> new_row_splits(c, new_num_lists + 1),
      new_row_ids = renumbering.Old2New()[src_shape.RowIds(1)];
  int32_t *new_row_splits_data = new_row_splits.Data();
  const int32_t *old_row_splits_data = src_shape.RowSplits(1).Data(),
                *new2old_data = renumbering.New2Old().Data();
  // set `new_row_splits_data`.

#ifndef NDEBUG
  {
    Array1<int32_t> is_ok(c, 1, 1);
    int32_t *is_ok_data = is_ok.Data();
    int32_t old_num_lists = src_shape.Dim0();
    const int32_t *old2new_data = renumbering.Old2New().Data();
    K2_EVAL(
        c, old_num_lists, lambda_check_preconditions, (int32_t i)->void {
          if (old2new_data[i + 1] == old2new_data[i]) {  // This list not kept
            if (old_row_splits_data[i + 1] != old_row_splits_data[i]) {
              // this list was nonempty...
              is_ok_data[0] = 0;
            }
          }
        });
    K2_CHECK_NE(is_ok[0], 0) << "RenumberAxis0Simple(): preconditions not met; "
                                "renumbering removes nonempty lists.";
  }
#endif

  K2_EVAL(
      c, new_num_lists + 1, lambda_set_new_row_splits, (int32_t new_i)->void {
        int32_t j;
        if (new_i == new_num_lists) {
          j = num_elems;
        } else {
          int32_t old_i = new2old_data[new_i];
          j = old_row_splits_data[old_i];
        }
        new_row_splits_data[new_i] = j;
      });
  layers[0].row_splits = new_row_splits;
  layers[0].row_ids = new_row_ids;
  // no need to set its cached_tot_size; that didn't change.
  return RaggedShape(layers);
}

RaggedShape CoveringShape(int32_t num_srcs, RaggedShape **srcs) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  if (num_srcs == 1) return *srcs[0];

  K2_CHECK_EQ(srcs[0]->NumAxes(), 2);
  int32_t dim0 = srcs[0]->Dim0();
  ContextPtr &c = srcs[0]->Context();
  for (int32_t i = 1; i != num_srcs; ++i) {
    K2_CHECK_EQ(srcs[i]->NumAxes(), 2);
    K2_CHECK_EQ(srcs[i]->Dim0(), dim0);
    K2_CHECK(c->IsCompatible(*srcs[i]->Context()));
  }

  // get row splits of srcs
  Array1<int32_t *> row_splits_ptrs(GetCpuContext(), num_srcs);
  int32_t **splits_ptr_data = row_splits_ptrs.Data();
  for (int32_t i = 0; i != num_srcs; ++i) {
    splits_ptr_data[i] = srcs[i]->RowSplits(1).Data();
  }
  row_splits_ptrs = row_splits_ptrs.To(c);
  int32_t **src_row_splits_ptr_data = row_splits_ptrs.Data();

  RaggedShape shape = RegularRaggedShape(c, dim0, num_srcs);
  Array1<int32_t> values(c, dim0 * num_srcs);
  // elements in row i of `sublist_sizes` are the sizes of row i
  // of src[0], src[1]...
  Ragged<int32_t> sublist_sizes(shape, values);
  int32_t *values_data = sublist_sizes.values.Data();
  K2_EVAL2(
      c, dim0, num_srcs, lambda_set_sublist_sizes,
      (int32_t i, int32_t j)->void {
        values_data[i * num_srcs + j] =
            src_row_splits_ptr_data[j][i + 1] - src_row_splits_ptr_data[j][i];
      });

  Array1<int32_t> ans_row_splits(c, dim0 + 1);
  Array1<int32_t> ans_row_sizes = ans_row_splits.Arange(0, dim0);
  MaxPerSublist(sublist_sizes, 0, &ans_row_sizes);
  ExclusiveSum(ans_row_sizes, &ans_row_splits);
  return RaggedShape2(&ans_row_splits, nullptr, -1);
}

Array1<int32_t> CoveringShapeForwardMap(RaggedShape &src,
                                        RaggedShape &covering) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 2);
  K2_CHECK_EQ(covering.NumAxes(), 2);
  K2_CHECK_EQ(src.Dim0(), covering.Dim0());
  int32_t num_elems = covering.NumElements();
  K2_CHECK_GE(num_elems, src.NumElements());
  ContextPtr c = GetContext(src, covering);

  Array1<int32_t> ans(c, num_elems);
  int32_t *ans_data = ans.Data();
  const int32_t *covering_row_splits_data = covering.RowSplits(1).Data(),
                *covering_row_ids_data = covering.RowIds(1).Data(),
                *src_row_splits_data = src.RowSplits(1).Data();
  K2_EVAL(
      c, num_elems, lambda_set_value, (int32_t covering_idx01)->void {
        int32_t covering_idx0 = covering_row_ids_data[covering_idx01],
                covering_idx0x = covering_row_splits_data[covering_idx0],
                covering_idx1 = covering_idx01 - covering_idx0x;
        // src and covering has the same dim0
        int32_t src_idx0x = src_row_splits_data[covering_idx0],
                src_cur_row_size =
                    src_row_splits_data[covering_idx0 + 1] - src_idx0x;
        K2_DCHECK_GE(
            covering_row_splits_data[covering_idx0 + 1] - covering_idx0x,
            src_cur_row_size);
        if (covering_idx1 >= src_cur_row_size)
          ans_data[covering_idx01] = -1;
        else
          ans_data[covering_idx01] = src_idx0x + covering_idx1;  // src_idx01
      });
  return ans;
}

void RaggedShapeAxis0Splitter::Init(RaggedShape &src) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_layers = src.NumLayers(), num_layers_out = num_layers - 1,
          dim0 = src.Dim0();
  K2_CHECK_LE(num_layers_out, 4);  // If this fails, add something to the 4s and
                                   // 5s here and in the header.
  K2_CHECK_GT(num_layers, 1);

  ContextPtr c = src.Context();
  composite_row_splits_ = Array2<int32_t>(c, num_layers + 1, dim0 + 1);
  Array2Accessor<int32_t> composite_row_splits_acc =
      composite_row_splits_.Accessor();

  RowSplitsAccessor<5> src_row_splits_acc(src);

  SmallVec<int32_t *, 5> row_splits_out_acc;
  K2_CHECK(num_layers_out <= 5);
  Array1<int32_t> garbage1(c, dim0 + dim0 + 1);  // won't be read.
  row_splits_out_acc.data[0] = garbage1.Data();
  for (int32_t l = 0; l < num_layers_out; l++) {
    row_splits_out_[l] = Array1<int32_t>(c, src.TotSize(l + 1) + dim0 + 1);
    row_splits_out_acc.data[l + 1] = row_splits_out_[l].Data();
  }

  // set composite_row_splits_ and also those elements of
  // the output row_splits which are bound to be zero.
  K2_EVAL(
      c, dim0 + 1, lambda_set_composite_row_splits, (int32_t i)->void {
        int32_t cur_pos = i;
        composite_row_splits_acc(0, i) = cur_pos;
        for (int32_t l = 0; l < num_layers; l++) {
          // The following statement sets the zero at the beginning of each
          // row_splits, plus a final zero that we write to avoid an
          // if-statement.
          row_splits_out_acc.data[l][cur_pos + i] = 0;
          cur_pos = src_row_splits_acc.ptrs[l][cur_pos];
          composite_row_splits_acc(l + 1, i) = cur_pos;
        }
      });

  composite_row_splits_cpu_ = composite_row_splits_.To(GetCpuContext());

  // Right now to_idx0 maps from an idx0 to an idx0 (identity map); next time it
  // will map from an idx01 to to an idx0, then idx012 to idx0 (all w.r.t. src).
  // It doesn't include the extra last element like a row_splits would; it's
  // like a composite row_ids vector: row_ids1, row_ids12 and so on.
  Array1<int32_t> to_idx0 = composite_row_splits_.Row(0).Arange(0, dim0);

  for (int32_t layer = 0; layer < num_layers_out; layer++)
    row_ids_out_[layer] = Array1<int32_t>(c, src.TotSize(layer + 2));

  Array1<int32_t> garbage2(c,
                           src.TotSize(1));  // corresponds to row_ids_out_[-1].

  for (int32_t layer = 0; layer <= num_layers_out; layer++) {
    // num_elems is the number of elements we process in this kernel.
    int32_t num_elems = src.TotSize(layer + 1);

    // The names here are valid for layer == 1; this just happens to be useful
    // for exposition.
    const int32_t *src_row_ids2_data = src.RowIds(layer + 1).Data(),
                  *idx01_to_idx0_data = to_idx0.Data();

    int32_t *row_ids1_out_data =
        (layer == 0 ? garbage2.Data() : row_ids_out_[layer - 1].Data());

    if (layer < num_layers_out) {
      Array1<int32_t> to_idx0_next(c, num_elems);
      int32_t *row_splits2_out_data = row_splits_out_[layer].Data(),
              *idx012_to_idx0_data = to_idx0_next.Data();
      const int32_t *src_row_splits3_data = src.RowSplits(layer + 2).Data();
      // row_splits3 maps from idx012 -> idx012x.

      // remember: the names are valid for layer == 1, just as an example.
      K2_EVAL(
          c, num_elems, lambda_set_row_splits_and_ids,
          (int32_t src_idx012)->void {
            int32_t src_idx01 = src_row_ids2_data[src_idx012],
                    src_idx012x_next = src_row_splits3_data[src_idx012 + 1],
                    src_idx0 = idx01_to_idx0_data[src_idx01];
            idx012_to_idx0_data[src_idx012] = src_idx0;  // <-- output here.
            int32_t src_idx0x = composite_row_splits_acc(layer, src_idx0),
                    src_idx0xxx = composite_row_splits_acc(layer + 2, src_idx0),
                    src_idx1 = src_idx01 - src_idx0x,
                    src_idx12x_next = src_idx012x_next - src_idx0xxx,
                    out_idx0 = src_idx1, out_idx01x_next = src_idx12x_next;
            row_ids1_out_data[src_idx012] = out_idx0;
            // below, the "+1" is because each element handles the next one
            // within this output row_splits array, with the zeros (1st elem of
            // each output row_splits array) handled by
            // lambda_set_composite_row_splits.  The "+ idx0" is to make room
            // for the extra final element of all the previous row_splits
            // arrays.
            row_splits2_out_data[src_idx012 + 1 + src_idx0] = out_idx01x_next;
          });
      to_idx0 = to_idx0_next;
    } else {
      // The next code is a subset of the other branch.
      K2_EVAL(
          c, num_elems, lambda_set_row_ids, (int32_t src_idx012)->void {
            int32_t src_idx01 = src_row_ids2_data[src_idx012],
                    idx0 = idx01_to_idx0_data[src_idx01],
                    src_idx0x = composite_row_splits_acc(layer, idx0),
                    src_idx1 = src_idx01 - src_idx0x, out_idx0 = src_idx1;
            row_ids1_out_data[src_idx012] = out_idx0;
          });
    }
  }
}

RaggedShape RaggedShapeAxis0Splitter::GetElement(int32_t i,
                                                 int32_t *elem_offset) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_layers_out = composite_row_splits_.Dim0() - 2;
  std::vector<RaggedShapeLayer> out;
  out.reserve(num_layers_out);

  auto composite_row_splits_cpu_acc = composite_row_splits_cpu_.Accessor();

  for (int32_t layer = 0; layer < num_layers_out; layer++) {
    int32_t row_begin = composite_row_splits_cpu_acc(layer + 1, i),
            row_end = composite_row_splits_cpu_acc(layer + 1, i + 1),
            elem_begin = composite_row_splits_cpu_acc(layer + 2, i),
            elem_end = composite_row_splits_cpu_acc(layer + 2, i + 1),
            num_elems = elem_end - elem_begin;
    if (layer + 1 == num_layers_out && elem_offset != nullptr)
      *elem_offset = elem_begin;

    // the "+ i" is to account for the extra final elements of preceding

    // row_splits vectors; the + 1 is for the final element of this one.
    Array1<int32_t> splits = row_splits_out_[layer].Arange(row_begin + i,
                                                           row_end + i + 1),
                    ids = row_ids_out_[layer].Arange(elem_begin, elem_end);
    out.emplace_back(RaggedShapeLayer{splits, ids, num_elems});
  }
  // TODO: when thoroughly debugged, maybe turn off validation?
  return RaggedShape(out);
}



namespace hash_internal {
// Utilities for hashing strings (actually: sequences of int32_t).

/*
  T can be int32_t or int64_t.
  The following code shows what we are computing:

    std::vector<int32_t> input;
    T hash1 = 13, hash2 = 787;
    for (size_t i = 0; i < input.size(); i++) {
      hash1 = 31 * hash1 + input[i];
      hash2 = 167 * hash2 + input[i];
    }
    hash = hash1 + 104729 * hash2;

  I'm not sure that these constants are very optimal, but they are primes.

  The actual calculation is a little different from the above because
  of the need to do it via a reduction.
*/
template <typename T>
struct Hash {
  T hash1;
  T hash2;
  T product1;
  T product2;

  // Would like this to be a POD type so not adding the following constructor:
  // Hash(int32_t i): hash1(i), hash2(i), product1(31), product2(167) { }
  // .. but implementing it in HashInputIterator.
};

template <typename T>
struct HashInputIterator {
  explicit __host__ __device__ __forceinline__ HashInputIterator(const int32_t *i)  // NOLINT
      : i_(i) {}
  __device__ __forceinline__ Hash<T> operator[](int32_t idx) const {
    return Hash<T>{i_[idx], i_[idx], 31, 167};
  }
  __device__ __forceinline__ HashInputIterator operator+(int32_t offset) const {
    return HashInputIterator(i_ + offset);
  }
  const int32_t *i_;
};

template <typename T>
struct HashOutputIteratorDeref {  // this is what you get when you dereference
                                  // HashOutputIterator, it pretends to be a
                                  // Hash<T> but really only stores the `idx`
                                  // member.
  explicit __device__ __forceinline__ HashOutputIteratorDeref(T *t)
      : t_(t) {}
  __device__ __forceinline__ HashOutputIteratorDeref &operator=(
      const Hash<T> &h) {
    *t_ = h.hash1 + 13 * h.product1 + 104729 * h.hash2 +
          (104729 * 787) * h.product2;
    return *this;
  }
  T *t_;
};

template <typename T>
struct HashOutputIterator {  // outputs just the index of the pair.
  explicit HashOutputIterator(T *t) : t_(t) {}
  __device__ __forceinline__ HashOutputIteratorDeref<T> operator[](
      int32_t idx) const {
    return HashOutputIteratorDeref<T>(t_ + idx);
  }
  __device__ __forceinline__ HashOutputIterator operator+(size_t offset) {
    return HashOutputIterator{t_ + offset};
  }
  T *t_;
};

template <typename T>
struct HashCombineOp {
  __device__ __forceinline__ Hash<T> operator()(const Hash<T> &a,
                                                const Hash<T> &b) const {
    return Hash<T>{a.hash1 * b.product1 + b.hash1,
                   a.hash2 * b.product2 + b.hash2,
                   a.product1 * b.product1,
                   a.product2 * b.product2};
  }
};

}  // namespace hash_internal
}  // namespace k2

namespace std {
// those below typedefs are required by cub::DeviceSegmentedReduce:Reduce
template <typename T>
struct iterator_traits<k2::hash_internal::HashInputIterator<T>> {
  typedef k2::hash_internal::Hash<T> value_type;
};
template <typename T>
struct iterator_traits<k2::hash_internal::HashOutputIterator<T>> {
  typedef k2::hash_internal::Hash<T> value_type;
  typedef k2::hash_internal::HashOutputIteratorDeref<T> reference;
};
}  // namespace std

namespace k2 {
template <typename T>
Array1<T> ComputeHash(Ragged<int32_t> &src) {
  NVTX_RANGE(K2_FUNC);

  int32_t last_axis = src.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = src.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;
  ContextPtr &c = src.Context();
  Array1<T> ans(c, num_rows);

  const int32_t *row_splits = row_splits_array.Data();
  const int32_t *values_data = src.values.Data();
  T *output_data = ans.Data();

  if (c->GetDeviceType() == kCpu) {
    int32_t j = row_splits[0];
    for (int32_t i = 0; i < num_rows; ++i) {
      T hash1 = 13, hash2 = 787;
      int32_t row_end = row_splits[i + 1];
      for (; j < row_end; ++j) {
        T elem = values_data[j];
        hash1 = 31 * hash1 + elem;
        hash2 = 167 * hash2 + elem;
      }
      T hash = hash1 + 104729 * hash2;
      output_data[i] = hash;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    hash_internal::HashInputIterator<T> input_iter(values_data);
    hash_internal::HashOutputIterator<T> output_iter(output_data);
    hash_internal::HashCombineOp<T> op;
    hash_internal::Hash<T> initial_hash{ 0, 0, 1, 1 };

    // This code is based on the example here:
    // https://nvlabs.github.io/cub/structcub_1_1_device_segmented_reduce.html
    std::size_t temp_storage_bytes = 0;

    // the first time is to determine temporary device storage requirements
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_bytes, input_iter, output_iter, num_rows,
        row_splits, row_splits + 1, op, initial_hash, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage.Data(), temp_storage_bytes, input_iter, output_iter,
        num_rows, row_splits, row_splits + 1, op, initial_hash,
        c->GetCudaStream()));
  }
  return ans;
}

Ragged<int32_t> UniqueSequences(Ragged<int32_t> &src,
                                Ragged<int32_t> *num_repeats /*=nullptr*/,
                                Array1<int32_t> *new2old_indexes /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  if (src.NumAxes() == 2) {
    // Put 'fake' layer at front, process, then remove.
    Ragged<int32_t> temp = Unsqueeze(src, 0);
    return UniqueSequences(temp, num_repeats, new2old_indexes).RemoveAxis(0);
  }
  Array1<int64_t> hashes = ComputeHash<int64_t>(src);
  int32_t hashes_dim = hashes.Dim();
  Array1<int32_t> order(c, hashes_dim);

  // Using the layer before the last layer of `src` for the shape of
  // `ragged_hashes`
  Ragged<int64_t> ragged_hashes(GetLayer(src.shape, src.shape.NumLayers() - 2),
                                hashes);

  SortSublists<int64_t, LessThan<int64_t> >(&ragged_hashes, &order);

  Renumbering renumber_lists(c, hashes.Dim());
  const int32_t *ragged_hashes_row_ids_data = ragged_hashes.RowIds(1).Data(),
      *ragged_hashes_row_splits_data = ragged_hashes.RowSplits(1).Data();
  const int64_t *ragged_hashes_data = ragged_hashes.values.Data();
  char *keep_list_data = renumber_lists.Keep().Data();
  K2_EVAL(
      c, hashes_dim, lambda_set_keep, (int32_t i)->void {
        char keep;
        if (i == ragged_hashes_row_splits_data[ragged_hashes_row_ids_data[i]]) {
          // this is the first element of its sub-list in `ragged_hashes`.
          keep = 1;
        } else {
          keep = (ragged_hashes_data[i] != ragged_hashes_data[i - 1]);
        }
        keep_list_data[i] = keep;
      });
  Array1<int32_t> new2old = renumber_lists.New2Old(),
      new2unsorted = order[new2old];
  Ragged<int32_t> ans = Index(src, src.NumAxes() - 2, new2unsorted);
  if (num_repeats != nullptr) {
    int32_t new2old_dim = new2old.Dim();
    Array1<int32_t> num_repeats_array(c, new2old_dim);
    const int32_t *new2old_data = new2old.Data();
    int32_t *num_repeats_data = num_repeats_array.Data();
    K2_EVAL(
        c, new2old_dim, set_num_repeats, (int32_t i)->void {
          if (i < new2old_dim - 1) {
            num_repeats_data[i] = new2old_data[i + 1] - new2old_data[i];
          } else {
            num_repeats_data[i] = hashes_dim - new2old_data[i];
          }
        });
    *num_repeats = Ragged<int32_t>(GetLayer(ans.shape, ans.NumAxes() - 3),
                                   num_repeats_array);
  }
  if (new2old_indexes != nullptr) {
    *new2old_indexes = std::move(new2unsorted);
  }
  return ans;
}

// Instantiate template for int64 and int32.
template
Array1<int64_t> ComputeHash(Ragged<int32_t> &src);
template
Array1<int32_t> ComputeHash(Ragged<int32_t> &src);


}  // namespace k2
