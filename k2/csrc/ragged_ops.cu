/**
 * @brief
 * ragged_ops
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Yiming Wang
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "cub/cub.cuh"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/ragged_utils.h"
#include "moderngpu/kernel_mergesort.hxx"

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
  NVTX_RANGE(K2_FUNC);
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
  std::size_t a_size = a_axes.size(),
              b_size = b_axes.size(),
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
  // In future we might choose to allocate everything in one big array, to avoid
  // multiple allocations, but for now just do it the simple way.
  for (int32_t axis = 1; axis < num_axes; ++axis) {
    axes[axis - 1].row_splits = Array1<int32_t>(c, tot_sizes[axis - 1] + 1);
    axes[axis - 1].row_ids = Array1<int32_t>(c, tot_sizes[axis]);
    axes[axis - 1].cached_tot_size = tot_sizes[axis];
  }
  // Not check here as we did not set the values of row_splits and row_ids
  return RaggedShape(axes, false);
}

Array1<int32_t *> GetRowSplitsPtr(RaggedShape &src) {
  NVTX_RANGE(K2_FUNC);
  int32_t axes = src.NumAxes();
  K2_CHECK_GE(axes, 2);
  std::vector<int32_t *> row_splits_start(axes - 1);
  for (int32_t i = 1; i != axes; ++i) {
    Array1<int32_t> &cur_splits = src.RowSplits(i);
    row_splits_start[i - 1] = cur_splits.Data();
  }
  return Array1<int32_t *>(src.Context(), row_splits_start);
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
  ContextPtr c = src.Context();
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
  ContextPtr c = src[0]->Context();

  std::vector<int32_t> all_row_splits_vec(num_srcs * 2);
  int32_t max_dim = 0;
  // all_row_splits_vec will contain [ 0 d0 0 d1 0 d2 .. ]
  // where d0 == src[0]->Dim0(), d1 == src[1]->Dim0()..
  for (int32_t i = 0; i < num_srcs; i++) {
    int32_t this_dim0 = src[i]->Dim0();
    if (this_dim0 > max_dim) max_dim = this_dim0;
    all_row_splits_vec[i * 2] = 0;
    all_row_splits_vec[i * 2 + 1] = this_dim0;
  }
  Array1<int32_t> all_row_splits(c, all_row_splits_vec);
  Array1<int32_t> all_row_ids(c, max_dim, 0);

  for (int32_t i = 0; i < num_srcs; i++) {
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
    ans.emplace_back(axes);
  }
  return ans;
}

/*
  Internal function used in Index(), which gets certain arrays used internally.

     @param [in] src      Source shape to be indexed
     @param [in] src_row_splits_ptrs  Result of calling GetRowSplitsPtr(src)
     @param [in] new2old  Array of indexes into axis 0 of src
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
                                const Array1<int32_t *> &src_row_splits_ptrs,
                                const Array1<int32_t> &new2old,
                                Array2<int32_t> *old_offsets,
                                Array2<int32_t> *new_offsets) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(src.NumAxes(), 1);
  ContextPtr &c = src.Context();
  int32_t num_axes = src.NumAxes(), ans_dim0 = new2old.Dim();
  int32_t *const *src_row_splits_ptrs_data = src_row_splits_ptrs.Data();
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
        int32_t old_offset = new2old_data[i], old_offset_next = old_offset + 1;
        for (int32_t axis = 0;; axis++) {
          old_offsets_acc(axis, i) = old_offset;
          // Below, 'new_offsets_acc' currently contains the size rather
          // than the offset; we need to do exclusive-sum.
          new_offsets_acc(axis, i) = old_offset_next - old_offset;
          if (axis + 1 == num_axes) return;
          old_offset = src_row_splits_ptrs_data[axis][old_offset];
          old_offset_next = src_row_splits_ptrs_data[axis][old_offset_next];
        }
      });
  ExclusiveSum(*new_offsets, new_offsets);
}

RaggedShape Index(RaggedShape &src, const Array1<int32_t> &new2old,
                  Array1<int32_t> *elem_indexes /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  bool is_cpu = (c->GetDeviceType() == kCpu);
  K2_CHECK(IsCompatible(src, new2old));
  int32_t num_axes = src.NumAxes(), src_dim0 = src.Dim0(),
          ans_dim0 = new2old.Dim();
  if (ans_dim0 == 0) {
    if (elem_indexes) *elem_indexes = Array1<int32_t>(c, 0);
    return EmptyRaggedShape(c, num_axes);
  }

  Array1<int32_t *> src_row_splits_ptrs = GetRowSplitsPtr(src);
  Array2<int32_t> old_offsets,  // num_axes by ans_dim0
      new_offsets;              // num_axes by (ans_dim0 + 1).
  GetOldAndNewOffsets(src, src_row_splits_ptrs, new2old, &old_offsets,
                      &new_offsets);

  Array1<int32_t> tot_sizes_out =
      Array1<int32_t>(new_offsets.Col(ans_dim0)).To(GetCpuContext());

  if (elem_indexes) *elem_indexes = Array1<int32_t>(c, tot_sizes_out.Back());

  RaggedShape ans = RaggedShapeFromTotSizes(c, num_axes, tot_sizes_out.Data());

  auto old_offsets_acc = old_offsets.Accessor(),
       new_offsets_acc = new_offsets.Accessor();

  ParallelRunner pr(c);
  std::vector<cudaStream_t> streams(num_axes);
  int32_t num_jobs = ans_dim0 * 2;  // note: this formula is not a heuristic;
                                    // it's how TaskRedirect works..
  Array2<TaskRedirect> task_redirects(c, num_axes, num_jobs);
  auto task_redirects_acc = task_redirects.Accessor();
  for (int32_t axis = 0; axis < num_axes; ++axis) {
    streams[axis] = pr.NewStream();
    With w(streams[axis]);
    const int32_t *new_offsets_ptr = new_offsets_acc.Row(axis);
    TaskRedirect *task_redirect_ptr = task_redirects_acc.Row(axis);
    GetTaskRedirect(c, ans_dim0, new_offsets_ptr, task_redirect_ptr);
  }

  for (int32_t axis = 0; axis < num_axes - 1; ++axis) {
    {
      int32_t *this_new_row_splits = ans.RowSplits(axis + 1).Data();
      const int32_t *this_old_row_splits = src.RowSplits(axis + 1).Data();

      auto lambda_set_row_splits = [=] __host__ __device__(
                                       int32_t ans_idx0, int32_t num_threads,
                                       int32_t thread_idx) -> void {
        //  0 <= ans_idx0 < ans_dim0; and 0 <= thread_idx < num_threads,
        //  num_threads may have any value > 0 as far as this code is concerned.
        //
        // Reminder of how row_splits work dimensionally: they are a map
        // from, e.g. an idx0 to an idx0x.   An offsets_acc(0,n) is
        // dimensionally an idx0; an offsets_acc(1,n) an idx01, and so on.
        // The locations in the row_splits array are as given by
        // the `axis`'th row of `offsets`; the values in the array
        // are related to those in the `axis+1`'th row.
        int32_t this_new_offset = new_offsets_acc(axis, ans_idx0),
                next_new_offset = new_offsets_acc(axis, ans_idx0 + 1),
                num_rows = next_new_offset - this_new_offset,
                this_old_offset = old_offsets_acc(axis, ans_idx0),
                value_offset = new_offsets_acc(axis + 1, ans_idx0) -
                               old_offsets_acc(axis + 1, ans_idx0);

        // Using <= instead of < below causes threads for different ans_idx0 to
        // write a single overlapping value, but also ensures that the
        // terminating value is written.  This only works because row_splits
        // vectors always start with 0, which is not necessarily the case
        // for row-ids.
        for (; thread_idx <= num_rows; thread_idx += num_threads) {
          this_new_row_splits[this_new_offset + thread_idx] =
              value_offset + this_old_row_splits[this_old_offset + thread_idx];
        }
      };
      int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis],
              target_num_loops = (is_cpu || tot_work > 1000000 ? 8 : 2);
      EvalWithRedirect(streams[axis], num_jobs, task_redirects_acc.Row(axis),
                       min_threads_per_job, tot_work, target_num_loops,
                       lambda_set_row_splits);
    }

    {
      int32_t *this_new_row_ids = ans.RowIds(axis + 1).Data();
      const int32_t *this_old_row_ids = src.RowIds(axis + 1).Data();
      int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis],
              target_num_loops = (is_cpu || tot_work > 1000000 ? 8 : 2);

      if (elem_indexes == nullptr || axis != num_axes - 2) {
        // If we don't need to write to `elem_indexes`...  [caution: the next
        // code block differs from this only by a statement that sets
        // `elem_indexes` and they should be kept in sync.]

        auto lambda_set_row_ids = [=] __host__ __device__(
                                      int32_t ans_idx0, int32_t num_threads,
                                      int32_t thread_idx) -> void {
          // Reminder of how row_ids work dimensionally: they are a map
          // from, e.g. an idx01 to an idx0.   An offsets_acc(0,n) is
          // dimensionally an idx0; an offsets_acc(1,n) an idx01, and so on.
          // The locations in the row_ids array are as given by
          // the `axis+1`'th row of `offsets`; the values in the array
          // are related to those in the `axis`'th row.
          int32_t this_new_offset = new_offsets_acc(axis + 1, ans_idx0),
                  next_new_offset = new_offsets_acc(axis + 1, ans_idx0 + 1),
                  num_elems = next_new_offset - this_new_offset,
                  this_old_offset = old_offsets_acc(axis + 1, ans_idx0),
                  value_offset = new_offsets_acc(axis, ans_idx0) -
                                 old_offsets_acc(axis, ans_idx0);
          for (; thread_idx < num_elems; thread_idx += num_threads) {
            this_new_row_ids[this_new_offset + thread_idx] =
                value_offset + this_old_row_ids[this_old_offset + thread_idx];
          }
        };
        EvalWithRedirect(streams[axis + 1], num_jobs,
                         task_redirects_acc.Row(axis + 1), min_threads_per_job,
                         tot_work, target_num_loops, lambda_set_row_ids);
      } else {
        int32_t *elem_indexes_data = elem_indexes->Data();
        // We need to write to `elem_indexes`.  Note: this code block only
        // differs from the above by an extra statement regarding
        // `elem_indexes`. Comments have been removed.
        auto lambda_set_row_ids_and_elem_indexes =
            [=] __host__ __device__(int32_t ans_idx0, int32_t num_threads,
                                    int32_t thread_idx) -> void {
          int32_t this_new_offset = new_offsets_acc(axis + 1, ans_idx0),
                  next_new_offset = new_offsets_acc(axis + 1, ans_idx0 + 1),
                  num_elems = next_new_offset - this_new_offset,
                  this_old_offset = old_offsets_acc(axis + 1, ans_idx0),
                  value_offset = new_offsets_acc(axis, ans_idx0) -
                                 old_offsets_acc(axis, ans_idx0);
          for (; thread_idx < num_elems; thread_idx += num_threads) {
            this_new_row_ids[this_new_offset + thread_idx] =
                value_offset + this_old_row_ids[this_old_offset + thread_idx];
            elem_indexes_data[this_new_offset + thread_idx] =
                this_old_offset + thread_idx;
          }
        };
        EvalWithRedirect(streams[axis + 1], num_jobs,
                         task_redirects_acc.Row(axis + 1), min_threads_per_job,
                         tot_work, target_num_loops,
                         lambda_set_row_ids_and_elem_indexes);
      }
    }
  }
#if !defined(NDEBUG)
  ans.Check();
#endif
  return ans;
}

Array2<int32_t> GetOffsets(int32_t num_srcs, RaggedShape **src) {
  K2_CHECK_GT(num_srcs, 0);
  int32_t num_axes_in = src[0]->NumAxes();
  ContextPtr ctx = src[0]->Context();
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

static RaggedShape AppendAxis0(int32_t num_srcs, RaggedShape **src,
                               Array1<uint32_t> *merge_map /* == nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  if (num_srcs == 1) {
    if (merge_map)
      *merge_map =
          Arange<uint32_t>(src[0]->Context(), 0, src[0]->NumElements());
    return **src;
  }
  K2_CHECK_GT(num_srcs, 1);

  int32_t num_axes = src[0]->NumAxes();
  ContextPtr c = src[0]->Context();
  bool is_cpu = (c->GetDeviceType() == kCpu);

  // Check if they have same num-axes and compatible context
  for (int32_t i = 1; i < num_srcs; ++i) {
    K2_CHECK_EQ(num_axes, src[i]->NumAxes());
    K2_CHECK(IsCompatible(*src[0], *src[i]));
  }

  // `offsets` will be on CPU for now.
  Array2<int32_t> offsets = GetOffsets(num_srcs, src);
  auto offsets_acc = offsets.Accessor();

  std::vector<int32_t> tot_sizes_out(num_axes);
  for (int32_t axis = 0; axis < num_axes; ++axis)
    tot_sizes_out[axis] = offsets_acc(axis + 1, num_srcs);

  RaggedShape ans = RaggedShapeFromTotSizes(c, num_axes, tot_sizes_out.data());

  Array2<int32_t *> src_row_splits, src_row_ids;
  GetRowInfoMulti(num_srcs, src, &src_row_splits, &src_row_ids);
  auto src_row_splits_acc = src_row_splits.Accessor(),
       src_row_ids_acc = src_row_ids.Accessor();
  offsets = offsets.To(c);
  offsets_acc = offsets.Accessor();  // on GPU now (if we're using one)

  ParallelRunner pr(c);
  std::vector<cudaStream_t> streams(num_axes);
  int32_t num_jobs = num_srcs * 2;

  // task_redirects is a device array (if using GPU).
  // We have `num_axes - 1` different sets of row_splits/row_ids to
  // populate but they have different sizes; the total number of distinct
  // sizes is `num_axes`.
  Array2<TaskRedirect> task_redirects(c, num_axes, num_jobs);
  auto task_redirects_acc = task_redirects.Accessor();
  // populate task_redirects (these allocate blocks of threads roughly
  // proportionally to the amount of data to process from this source.
  for (int32_t axis = 0; axis < num_axes; ++axis) {
    streams[axis] = pr.NewStream();
    With w(streams[axis]);
    const int32_t *offsets = offsets_acc.Row(axis + 1);
    // c->GetCudaStream() == stream[axis] as it has been overridden by With
    GetTaskRedirect(c, num_srcs, offsets, task_redirects_acc.Row(axis));
  }

  for (int32_t axis = 0; axis < num_axes - 1; axis++) {
    // first set the row-splits.
    int32_t **this_src_row_splits = src_row_splits_acc.Row(axis),
            **this_src_row_ids = src_row_ids_acc.Row(axis);
    int32_t *this_dest_row_splits = ans.RowSplits(axis + 1).Data(),
            *this_dest_row_ids = ans.RowIds(axis + 1).Data();
    const int32_t *offsets_this_axis = offsets_acc.Row(axis + 1),
                  *offsets_next_axis = offsets_acc.Row(axis + 2);
    auto lambda_set_row_splits = [=] __host__ __device__(
                                     int32_t src_idx, int32_t num_threads,
                                     int32_t thread_idx) -> void {
      // Reminder of how row_splits work dimensionally: they are a map
      // from, e.g. an idx0 to an idx0x.   An offsets_acc(0,n) is
      // dimensionally an idx0; an offsets_acc(1,n) an idx01, and so on.
      int32_t this_offset = offsets_this_axis[src_idx],
              next_offset = offsets_this_axis[src_idx + 1],
              this_value_offset = offsets_next_axis[src_idx],
              num_rows = next_offset - this_offset;
      int32_t *src_row_splits_ptr = this_src_row_splits[src_idx];
      // Using <= instead of < below causes threads for different src_idx to
      // write a single overlapping value, but also ensures that the
      // terminating value is written.  This only works because row_splits
      // vectors always start with 0, which is not necessarily the case
      // for row-ids.
      for (; thread_idx <= num_rows; thread_idx += num_threads) {
        this_dest_row_splits[this_offset + thread_idx] =
            this_value_offset + src_row_splits_ptr[thread_idx];
      }
    };

    int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis],
            target_num_loops = (is_cpu || tot_work > 1000000 ? 8 : 2);
    EvalWithRedirect(streams[axis], num_jobs, task_redirects_acc.Row(axis),
                     min_threads_per_job, tot_work, target_num_loops,
                     lambda_set_row_splits);

    {  // set the row-ids
      auto lambda_set_row_ids = [=] __host__ __device__(
                                    int32_t src_idx, int32_t num_threads,
                                    int32_t thread_idx) -> void {
        // Reminder of how row_ids work dimensionally: they are a map
        // from, e.g. an idx01 to an idx0.   An offsets_acc(0,n) is
        // dimensionally an idx0; an offsets_acc(1,n) an idx01, and so on.
        int32_t this_offset = offsets_next_axis[src_idx],
                next_offset = offsets_next_axis[src_idx + 1],
                this_value_offset = offsets_this_axis[src_idx],
                num_elems = next_offset - this_offset;
        int32_t *src_row_ids_ptr = this_src_row_ids[src_idx];
        for (; thread_idx < num_elems; thread_idx += num_threads) {
          this_dest_row_ids[this_offset + thread_idx] =
              this_value_offset + src_row_ids_ptr[thread_idx];
        }
      };
      int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis + 1],
              target_num_loops = (tot_work > 1000000 ? 4 : 2);
      // TODO(haowen): maybe we should launch kernels for row_splits and row_ids
      // in different streams
      EvalWithRedirect(streams[axis + 1], num_jobs,
                       task_redirects_acc.Row(axis + 1), min_threads_per_job,
                       tot_work, target_num_loops, lambda_set_row_ids);
    }
  }
  if (merge_map) {
    std::vector<int32_t> num_elems_out(num_srcs);
    for (int32_t i = 0; i < num_srcs; ++i)
      num_elems_out[i] = src[i]->NumElements();
    *merge_map = SizesToMergeMap(c, num_elems_out);
  }
  return ans;
}

RaggedShape Append(int32_t axis, int32_t num_srcs, RaggedShape **src,
                   Array1<uint32_t> *merge_map /* == nullptr*/) {
  K2_CHECK(num_srcs > 0);
  if (axis == 0) return AppendAxis0(num_srcs, src, merge_map);

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
  if (src_dim0 <= 0) return src;
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
      Index(src_no_axis0, renumbering, value_indexes);

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
    RaggedShape ans_appended = AppendAxis0(num_srcs, src, merge_map);
    ContextPtr cpu = GetCpuContext();
    Array1<int32_t> row_splits(cpu, num_srcs + 1);
    int32_t *row_splits_data = row_splits.Data();
    for (int32_t i = 0; i < num_srcs; i++) row_splits_data[i] = src[i]->Dim0();
    int32_t cutoff = 32;
    if (num_srcs < cutoff) row_splits = row_splits.To(c);
    ExclusiveSum(row_splits, &row_splits);
    if (num_srcs >= cutoff) row_splits = row_splits.To(c);
    int32_t num_elems = ans_appended.Dim0();
    Array1<int32_t> row_ids(c, num_elems);
    RowSplitsToRowIds(row_splits, &row_ids);
    RaggedShape ans_layer0 = RaggedShape2(&row_splits, &row_ids, num_elems);
    return ComposeRaggedShapes(ans_layer0, ans_appended);
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

RaggedShape Merge(int32_t num_srcs, RaggedShape **src,
                  const Array1<uint32_t> &merge_map,
                  Array1<uint32_t> *merge_map_out) {
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

static Array1<int32_t> GetTransposeReorderingCpu(Ragged<int32_t> &src,
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

static Array1<int32_t> GetTransposeReorderingThreeAxesCuda(Ragged<int32_t> &src,
                                                           int32_t num_cols) {
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
  if (src.NumAxes() == 3)
    return GetTransposeReorderingThreeAxesCuda(src, num_cols);

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
  return ans;
#endif
}

RaggedShape ChangeSublistSize(RaggedShape &src, int32_t size_delta) {
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
  Array1<int32_t *> src_row_splits_ptr = GetRowSplitsPtr(src);
  int32_t **src_row_splits_ptr_data = src_row_splits_ptr.Data();
  K2_EVAL(
      c, 1, lambda_set_indexes, (int32_t i)->void {
        // we just start a kernel with only one element here.
        K2_CHECK_EQ(i, 0);
        int32_t row_begin = begin, row_end = end;
        indexes_data[0] = row_begin, indexes_data[1] = row_end;
        for (int32_t cur_axis = axis; cur_axis < num_axes - 1; ++cur_axis) {
          row_begin = src_row_splits_ptr_data[cur_axis][row_begin];
          row_end = src_row_splits_ptr_data[cur_axis][row_end];
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

Ragged<int32_t> AddSuffixToRagged(Ragged<int32_t> &src,
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

Ragged<int32_t> AddPrefixToRagged(Ragged<int32_t> &src,
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

RaggedShape SubsampleRaggedShape(RaggedShape &src, Renumbering &renumbering) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(renumbering.NumOldElems(), src.NumElements());

  // Make sure final row-ids are populated.
  src.RowIds(src.NumAxes() - 1);
  std::vector<RaggedShapeLayer> axes = src.Layers();
  axes.back().row_ids = axes.back().row_ids[renumbering.New2Old()];
  axes.back().row_splits = renumbering.Old2New()[axes.back().row_splits];
  axes.back().cached_tot_size = axes.back().row_ids.Dim();
  return RaggedShape(axes);
}

RaggedShape SubsampleRaggedShape(RaggedShape &src, Renumbering &r_before_last,
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
  ContextPtr &c = shape.Context();

  Array1<int32_t> sizes = RowSplitsToSizes(shape.RowSplits(1));
  Array1<int32_t> index_map;
  Sort<int32_t, GreaterThan<int32_t>>(&sizes, &index_map);
  return index_map;
}

RaggedShape GetLayer(const RaggedShape &src, int32_t layer) {
  K2_CHECK_GE(layer, 0);
  K2_CHECK_LT(layer, src.NumAxes() - 1);
  std::vector<RaggedShapeLayer> layers;
  layers.push_back(src.Layers()[layer]);
  bool check = false;
  return RaggedShape(layers, check);
}

void DecomposeRaggedShape(const RaggedShape &src, int32_t axis,
                          RaggedShape *top, RaggedShape *bottom) {
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
  if (axis == 0) {
    return RemoveEmptyListsAxis0(src_shape, renumbering_out);
  }
  RaggedShape top_shape, bottom_shape;
  DecomposeRaggedShape(src_shape, axis, &top_shape, &bottom_shape);

  Renumbering r_temp;
  if (!renumbering_out) renumbering_out = &r_temp;
  bottom_shape = RemoveEmptyListsAxis0(bottom_shape, renumbering_out);
  top_shape = SubsampleRaggedShape(top_shape, *renumbering_out);
  return ComposeRaggedShapes(top_shape, bottom_shape);
}

RaggedShape RemoveSomeEmptyLists(RaggedShape &src_shape, int32_t axis,
                                 Renumbering &renumbering) {
  if (axis == 0) {
    return RenumberAxis0Simple(src_shape, renumbering);
  }
  RaggedShape top_shape, bottom_shape;
  DecomposeRaggedShape(src_shape, axis, &top_shape, &bottom_shape);

  bottom_shape = RenumberAxis0Simple(bottom_shape, renumbering);
  top_shape = SubsampleRaggedShape(top_shape, renumbering);
  return ComposeRaggedShapes(top_shape, bottom_shape);
}

RaggedShape RemoveEmptyListsAxis0(RaggedShape &src_shape,
                                  Renumbering *renumbering_out) {
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

}  // namespace k2
