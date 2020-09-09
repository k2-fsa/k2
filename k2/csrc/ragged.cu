/**
 * @brief
 * ragged
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/log.cuh"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"

namespace k2 {

int32_t RaggedShape::MaxSize(int32_t axis) {
  K2_CHECK_GT(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  const auto &row_splits = axes_[axis - 1].row_splits;
  const int32_t num_rows = row_splits.Dim() - 1;
  if (num_rows == 0) return 0;
  // TODO(haowen): write a new function to allocate just one element?
  Array1<int32_t> max_array(row_splits.Context(), 1, 0);
  int32_t *max_value = max_array.Data();
  const int32_t *row_splits_data = row_splits.Data();
  auto lambda_get_max = [=] __host__ __device__(int32_t i) -> void {
    int32_t value = row_splits_data[i + 1] - row_splits_data[i];
    if (value > max_value[0]) max_value[0] = value;
  };
  Eval(row_splits.Context(), num_rows, lambda_get_max);
  // this will convert to memory on CPU
  return max_array[0];
}

void RaggedShape::Check() {
  ContextPtr c = Context();
  int32_t num_axes = axes_.size();
  for (int32_t axis = 0; axis < axes_.size(); axis++) {
    RaggedShapeDim &rsd = axes_[axis];
    K2_CHECK_GE(rsd.row_splits.Dim(), 0);
    if (rsd.cached_tot_size >= 0) {
      K2_CHECK(rsd.row_splits.Dim() == 0 ||
               rsd.cached_tot_size == rsd.row_splits[rsd.row_splits.Dim() - 1]);
      K2_CHECK(rsd.row_ids.Dim() == 0 ||
               rsd.cached_tot_size == rsd.row_ids.Dim());
    } else {
      K2_CHECK_EQ(rsd.cached_tot_size, 1);
      K2_CHECK_EQ(rsd.row_ids.Dim(), 0);
    }
    int32_t num_elems;
    {  // Check row_splits.

      // meta[0] is a bool, ok == 1, not-ok == 0.
      // meta[1] will contain the number of row_splits.
      Array1<int32_t> meta(c, 2, 1);
      int32_t *ok_data = meta.Data(), *num_elems_data = ok_data + 1;
      const int32_t *row_splits_data = rsd.row_splits.Data();
      int32_t num_rows = rsd.row_splits.Dim() - 1;

      auto lambda_check_row_splits =
          [=] __host__ __device__(int32_t i) -> void {
        int32_t this_idx = row_splits_data[i];
        if (i == 0 && this_idx != 0) *ok_data = 0;
        if (i < num_rows) {
          int32_t next_idx = row_splits_data[i + 1];
          if (next_idx <= this_idx) *ok_data = 0;
        } else {
          K2_CHECK(i == num_rows);
          *num_elems_data = this_idx;
        }
      };
      Eval(c, num_rows + 1, lambda_check_row_splits);
      meta = meta.To(GetCpuContext());
      num_elems = meta[1];
      int32_t ok = meta[0];
      if (!ok) {
        K2_LOG(FATAL) << "Problem validating row-splits: for axes_[" << axis
                      << "], row_splits = " << rsd.row_splits.Dim();
      }
      if (rsd.cached_tot_size > 0 && rsd.cached_tot_size != num_elems) {
        K2_LOG(FATAL) << "Problem validating row-splits: for axes_[" << axis
                      << "], row_splits[-1] = " << num_elems
                      << " but cached_tot_size == " << rsd.cached_tot_size;
      }
    }
    if (axis + 1 < axes_.size()) {
      int32_t next_num_rows = axes_[axis + 1].row_splits.Dim();
      if (num_elems != next_num_rows) {
        K2_LOG(FATAL) << "Ragged shape has num_elems for axes_[" << axis
                      << "] == " << num_elems << " and num-rows for axes_["
                      << (axis + 1) << "] == " << next_num_rows;
      }
    }

    if (rsd.row_ids.Dim() != 0) {  // check row_ids.
      K2_CHECK(IsCompatible(rsd.row_ids, rsd.row_splits));
      // 1st elem is `ok` (1 or 0); 2nd elem is location of bad index
      // into row_splits
      Array1<int32_t> meta(c, 1, 2);
      int32_t *ok_data = meta.Data(), *bad_index_data = ok_data + 1;

      const int32_t *row_splits_data = rsd.row_splits.Data(),
                    *row_ids_data = rsd.row_ids.Data();
      int32_t num_elems = rsd.row_ids.Dim(),
              num_rows = rsd.row_splits.Dim() - 1;

      auto lambda_check_row_ids = [=] __host__ __device__(int32_t i) -> void {
        int32_t this_row = row_ids_data[i];
        if (this_row < 0 || this_row >= num_rows ||
            i < row_splits_data[this_row] ||
            i >= row_splits_data[this_row + 1]) {
          *ok_data = 0;
          *bad_index_data = i;
        }
      };
      // TODO: could do this and the other one in separate streams.
      Eval(c, num_elems, lambda_check_row_ids);
      meta = meta.To(GetCpuContext());  // since we have 2 accesses, this should
                                        // be faster.
      int32_t ok = meta[0];
      if (!ok) {
        K2_LOG(FATAL) << "Problem validating row-ids: for axes_[" << axis
                      << "], row_splits = " << rsd.row_splits
                      << ", row_ids = " << rsd.row_ids << ", see index "
                      << *bad_index_data << " of row_ids, whose dim is "
                      << rsd.row_ids.Dim();
      }
    }
    if (axis + 1 < axes_.size()) {
      K2_CHECK(IsCompatible(rsd.row_splits, axes_[axis + 1].row_splits));
    }
  }
}

// See declaration in ragged.h for documentation of its purpose and interface.
RaggedShape Unsqueeze(RaggedShape &src, int32_t axis) {
  // If axis == 0, initial row_splits and row_ids will look like the following,
  // if for example src.Dim0() was 5: [ 0 5 ],  [ 0 0 0 0 0 ].  The other axes
  // would be pushed forward.
  //
  // If 0 < axis <= src.NumAxes(), the inserted row_splits and row_ids would
  // look like the following, if for instance the src.TotSize(axis-1) = 8:
  //   [ 0 1 2 3 4 5 6 7 8 ], [ 0 1 2 3 4 5 6 7 ].
  //
  // The reason why the code is different for axis == 0, is that in that case we
  // are really making visible an "implicit" axis of the input `src`; we could
  // call it axis 0 of the original RaggedShape.  Imagine that "implicit" axis's
  // row_splits and row_ids map respectively from an idx_minus1 -> idx0 and from
  // an idx_0 to idx_minus1, where idx_minus1 is always 0 and 0 <= idx0 <
  // Dim0().

  ContextPtr c = src.Context();
  K2_CHECK(axis >= 0 && axis <= src.NumAxes());

  const std::vector<RaggedShapeDim> &axes_in = src.Axes();

  int32_t num_axes_in = src.NumAxes();

  // Note: in RaggedShape, the vector of RaggedShapeDim is of length num_axes -
  // 1,
  // so the output will have one more axis than the input.
  std::vector<RaggedShapeDim> axes_out(num_axes_in);

  int32_t row_splits_dim, row_ids_dim;
  Array1<int32_t> mem;

  if (axis == 0) {
    row_splits_dim = 2;        // e.g. [ 0 5 ]
    row_ids_dim = src.Dim0();  // e.g. [ 0 0 0 0 0 ]
    mem = Array1<int32_t>(c, row_splits_dim + row_ids_dim);
    int32_t *mem_data = mem.Data();
    auto lambda_set_mem = [=] __host__ __device__(int32_t i) -> void {
      if (i == 1)
        mem_data[i] = row_ids_dim;
      else
        mem_data[i] = 0;
    };
    Eval(c, mem.Dim(), lambda_set_mem);
    axes_out[0].row_splits = mem.Range(0, 2);
  } else {
    int32_t tot_size = src.TotSize(axis - 1);
    row_splits_dim = tot_size + 1;
    row_ids_dim = tot_size;
    mem = Array1<int32_t>(c, row_splits_dim + row_ids_dim);
    int32_t *mem_data = mem.Data();
    auto lambda_set_mem2 = [=] __host__ __device__(int32_t i) -> void {
      mem_data[i] = i % (tot_size + 1);
    };
    Eval(c, mem.Dim(), lambda_set_mem2);
  }
  axes_out[axis].row_splits = mem.Range(0, row_splits_dim);
  axes_out[axis].row_ids = mem.Range(row_splits_dim, row_ids_dim);
  for (int32_t i = 0; i < axis; i++) axes_out[axis] = axes_in[axis];
  // Note: the returned array has `num_axes_in + 1` axes, so its
  // array of RaggedShapeDim is of length `num_axes_in`.
  for (int32_t i = axis + 1; i < num_axes_in; i++)
    axes_out[axis] = axes_in[axis - 1];
  return RaggedShape(axes_out);
}

RaggedShape Renumber(RaggedShape &src, const Array1<int32_t> &new2old) {
  ContextPtr c = src.Context();
  K2_CHECK(IsCompatible(src, new2old));
  int32_t num_axes = src.NumAxes(), dim0 = src.Dim0();
  K2_CHECK_EQ(new2old.Dim(), dim0);
  std::vector<int32_t> tot_sizes_out(num_axes);
  for (int32_t axis = 0; axis < num_axes; axis++)
    tot_sizes_out[axis] = src.TotSize(axis);
  // the arrays in `ans` will be the same sizes as those in `src`.
  RaggedShape ans = RaggedShapeFromTotSizes(num_axes, tot_sizes_out.data());

  src.Populate();
  Array2<int32_t> old_offsets(c, num_axes, dim0 + 1),
      new_offsets(c, num_axes, dim0 + 1);
  auto old_offsets_acc = old_offsets.Accessor(),
       new_offsets_acc = new_offsets.Accessor();

  Array1<int32_t *> row_splits_ptrs = GetRowSplitsPtr(src);
  int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();

  // Set old_offsets
  auto lambda_get_old_offsets = [=] __host__ __device__(int32_t i) {
    // 0 <= i <= dim0
    int32_t cur_offset = i;
    for (int32_t axis = 0; axis < num_axes; axis++) {
      old_offsets_acc(0, i) = cur_offset;
      if (axis + 1 == num_axes) return;
      cur_offset = row_splits_ptrs_data[axis][cur_offset];
    }
  };
  Eval(c, dim0 + 1, lambda_get_old_offsets);
  const int32_t *new2old_data = new2old.Data();
  auto lambda_get_new_offsets = [=] __host__ __device__(int32_t axis,
                                                        int32_t new_i) {
    // 0 <= axis < num_axes;  0 <= new_i < dim0
    int32_t old_i = new2old_data[new_i],
            this_old_offset = old_offsets_acc(axis, old_i),
            next_old_offset = old_offsets_acc(axis, old_i + 1),
            size = next_old_offset - this_old_offset;
    new_offsets_acc(axis, new_i) = size;
  };
  Eval2(c, num_axes, dim0, lambda_get_new_offsets);
  ExclusiveSum(new_offsets, &new_offsets);
  // Now new_offsets contains the offsets, not the sizes.

  ParallelRunner pr(c);
  std::vector<cudaStream_t> streams(num_axes);
  int32_t num_jobs = dim0 * 2;  // note: this formula is not a heuristic; it's
                                // how TaskRedirect works..
  Array2<TaskRedirect> task_redirects(c, num_axes, num_jobs);
  auto task_redirects_acc = task_redirects.Accessor();
  for (int32_t axis = 0; axis < num_axes; axis++) {
    streams[axis] = pr.NewStream();
    With w(streams[axis]);
    const int32_t *new_offsets_ptr = new_offsets_acc.Row(axis);
    TaskRedirect *task_redirect_ptr = task_redirects_acc.Row(axis);
    GetTaskRedirect(c, dim0, new_offsets_ptr, task_redirect_ptr);
  }

  for (int32_t axis = 0; axis < num_axes - 1; axis++) {
    {
      int32_t *this_new_row_splits = ans.RowSplits(axis).Data();
      const int32_t *this_old_row_splits = src.RowSplits(axis).Data();

      auto lambda_set_row_splits = [=] __host__ __device__(
                                       int32_t new_idx, int32_t num_threads,
                                       int32_t thread_idx) -> void {
        //  0 <= new_idx < dim0; and 0 <= thread_idx < num_threads,
        //  num_threads may have any value > 0 as far as this code is concerned.
        //
        // Reminder of how row_splits work dimensionally: they are a map
        // from, e.g. an idx0 to an idx01.   An offsets_acc(0,n) is
        // dimensionally an idx0; an offsets_acc(1,n) an idx01, and so on.
        // The locations in the row_splits array are as given by
        // the `axis`'th row of `offsets`; the values in the array
        // are related to those in the `axis+1`'th row.
        int32_t old_idx = new2old_data[new_idx],
                this_old_offset = old_offsets_acc(axis, old_idx),
                next_old_offset = old_offsets_acc(axis, old_idx + 1),
                this_new_offset = new_offsets_acc(axis, old_idx),
                num_rows = next_old_offset - this_old_offset,
                value_offset = new_offsets_acc(axis + 1, new_idx) -
                               old_offsets_acc(axis + 1, old_idx);

        // Using <= instead of < below causes threads for different src_idx to
        // write a single overlapping value, but also ensures that the
        // terminating value is written.  This only works because row_splits
        // vectors always start with 0, which is not necessarily the case
        // for row-ids.
        for (; thread_idx <= num_rows; thread_idx += num_threads) {
          this_new_row_splits[this_new_offset + thread_idx] =
              value_offset + this_old_row_splits[thread_idx];
        }
      };
      int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis],
              target_num_loops = (tot_work > 1000000 ? 4 : 2);
      // bool include_final_task = false;
      EvalWithRedirect(streams[axis], num_jobs, task_redirects_acc.Row(axis),
                       min_threads_per_job, tot_work, target_num_loops,
                       lambda_set_row_splits);
    }

    {
      int32_t *this_new_row_ids = ans.RowIds(axis).Data();
      const int32_t *this_old_row_ids = src.RowIds(axis).Data();

      auto lambda_set_row_ids = [=] __host__ __device__(
                                    int32_t new_idx, int32_t num_threads,
                                    int32_t thread_idx) -> void {
        //  0 <= new_idx < dim0; and 0 <= thread_idx < num_threads,
        //  num_threads may have any value > 0 as far as this code is concerned.
        //
        // Reminder of how row_ids work dimensionally: they are a map
        // from, e.g. an idx01 to an idx0.   An offsets_acc(0,n) is
        // dimensionally an idx0; an offsets_acc(1,n) an idx01, and so on.
        // The locations in the row_ids array are as given by
        // the `axis+1`'th row of `offsets`; the values in the array
        // are related to those in the `axis`'th row.
        int32_t old_idx = new2old_data[new_idx],
                this_old_offset = old_offsets_acc(axis + 1, old_idx),
                next_old_offset = old_offsets_acc(axis + 1, old_idx + 1),
                this_new_offset = new_offsets_acc(axis + 1, old_idx),
                num_rows = next_old_offset - this_old_offset,
                value_offset = new_offsets_acc(axis, new_idx) -
                               old_offsets_acc(axis, old_idx);

        // Using <= instead of < below causes threads for different src_idx to
        // write a single overlapping value, but also ensures that the
        // terminating value is written.  This only works because row_splits
        // vectors always start with 0, which is not necessarily the case
        // for row-ids.
        for (; thread_idx < num_rows; thread_idx += num_threads) {
          this_new_row_ids[this_new_offset + thread_idx] =
              value_offset + this_old_row_ids[thread_idx];
        }
        // TODO: maybe remove this if I decide last value is not needed.
        if (new_idx == dim0 - 1 && thread_idx == num_rows) {
          int32_t next_value_offset = new_offsets_acc(axis, new_idx + 1) -
                                      old_offsets_acc(axis, old_idx + 1);
          this_new_row_ids[this_new_offset + thread_idx] = next_value_offset;
        }
      };
      int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis],
              target_num_loops = (tot_work > 1000000 ? 4 : 2);
      EvalWithRedirect(streams[axis], num_jobs, task_redirects_acc.Row(axis),
                       min_threads_per_job, tot_work, target_num_loops,
                       lambda_set_row_ids);
    }
  }
#ifndef NDEBUG
  ans.Check();
#endif
  return ans;
}

RaggedShape Append(int32_t num_srcs, RaggedShape **src, int32_t axis) {
  K2_CHECK_EQ(axis, 0) << "Append() with axis > 0 not yet supported";
  K2_CHECK_GT(num_srcs, 0);

  int32_t num_axes = src[0]->NumAxes();
  ContextPtr c = src[0]->Context();
  for (int32_t i = 1; i < num_srcs; i++) {
    // Check they have the same num-axes.
    K2_CHECK_EQ(num_axes, src[i]->NumAxes());
    K2_CHECK(IsCompatible(*src[0], *src[i]));
  }

  // `offsets` will be on CPU for now.
  Array2<int32_t> offsets = GetOffsets(num_srcs, src);
  auto offsets_acc = offsets.Accessor();

  std::vector<int32_t> tot_sizes_out(num_axes);
  for (int32_t axis = 0; axis < num_axes; axis++)
    tot_sizes_out[axis] = offsets_acc(axis, num_srcs);

  RaggedShape ans = RaggedShapeFromTotSizes(num_axes, tot_sizes_out.data());
  Array1<int32_t *> dest_row_splits, dest_row_ids;
  GetRowInfo(ans, &dest_row_splits, &dest_row_ids);

  Array2<int32_t *> src_row_splits, src_row_ids;
  GetRowInfoMulti(num_srcs, src, &src_row_splits, &src_row_ids);

  if (c->GetDeviceType() != kCpu) offsets = offsets.To(c);

  int32_t **dest_row_splits_data = dest_row_splits.Data(),
          **dest_row_ids_data = dest_row_ids.Data();
  auto src_row_splits_acc = src_row_splits.Accessor(),
       src_row_ids_acc = src_row_ids.Accessor();
  offsets_acc = offsets.Accessor();  // on GPU now (if we're using one)

  ParallelRunner pr(c);
  std::vector<cudaStream_t> streams(num_axes + 1);
  int32_t num_jobs = num_srcs * 2;
  // task_redirects is a device array (if using GPU).
  // We have `num_axes - 1` different sets of row_splits/row_ids to
  // populate but they have different sizes; the total number of distinct
  // sizes is `num_axes`.
  Array2<TaskRedirect> task_redirects(c, num_axes, num_jobs);
  auto task_redirects_acc = task_redirects.Accessor();
  // populate task_redirects (these allocate blocks of threads roughly
  // proportionally to the amount of data to process from this source.
  for (int32_t axis = 0; axis < num_axes; axis++) {
    streams[axis] = pr.NewStream();
    const int32_t *offsets = &(offsets_acc(axis, 0));
    GetTaskRedirect(c, num_srcs, offsets, task_redirects_acc.Row(axis));
  }

  for (int32_t axis = 0; axis < num_axes - 1; axis++) {
    // first set the row-splits.
    TaskRedirect *tr = &(task_redirects_acc(axis, 0));

    int32_t **this_src_row_splits = &(src_row_splits_acc(axis, 0)),
            **this_src_row_ids = &(src_row_ids_acc(axis, 0));
    int32_t *this_dest_row_splits = ans.RowSplits(axis + 1).Data(),
            *this_dest_row_ids = ans.RowIds(axis + 1).Data();
    const int32_t *offsets_this_axis = &(offsets_acc(axis, 0)),
                  *offsets_next_axis = &(offsets_acc(axis + 1, 0));
    auto lambda_set_row_splits = [=] __host__ __device__(
                                     int32_t src_idx, int32_t num_threads,
                                     int32_t thread_idx) -> void {
      // Reminder of how row_splits work dimensionally: they are a map
      // from, e.g. an idx0 to an idx01.   An offsets_acc(0,n) is
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
            target_num_loops = (tot_work > 1000000 ? 4 : 2);
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
        // We need to write the very last value at the end of all the
        // arrays; the last job (for src_idx == num_srcs - 1) does this
        // by adding 1 to num_srcs.  We can't let them all write an
        // extra value, because unlike row_splits, row_ids vectors may not
        // start with 0 in general; so having 2 threads write that
        // value (the 1st of each; one past the last of each) would cause
        // indeterminacy.
        if (src_idx == num_srcs - 1) num_elems++;
        for (; thread_idx <= num_elems; thread_idx += num_threads) {
          this_dest_row_ids[this_offset + thread_idx] =
              this_value_offset + src_row_ids_ptr[thread_idx];
        }
      };
      int32_t min_threads_per_job = 2, tot_work = tot_sizes_out[axis + 1],
              target_num_loops = (tot_work > 1000000 ? 4 : 2);
      // bool include_final_task = false;
      EvalWithRedirect(streams[axis + 1], num_jobs,
                       task_redirects_acc.Row(axis + 1), min_threads_per_job,
                       tot_work, target_num_loops, lambda_set_row_ids);
    }
  }
  return ans;
}

// transpose axes 0 and 1.
RaggedShape Transpose(RaggedShape &src) {
  K2_CHECK(src.NumAxes() > 2);
  int32_t src_dim0 = src.Dim0(), src_tot_size1 = src.TotSize(1);
  int32_t src_dim1 = src_tot_size1 % src_dim0;
  if (src_tot_size1 % src_dim0 != 0) {
    K2_LOG(FATAL) << "Transpose(): all dims on axis 0 must be the same.";
  }
  RaggedShape src_no_axis0 = RemoveAxis(src, 0);
  assert(src_no_axis0.Dim0() == src_tot_size1);
  ContextPtr c = src.Context();
  // `renumbering` is a `new2old` map, that maps from the first index in
  // src_no_axis0_renumbered
  // to the first index into src_no_axis0.
  Array1<int32_t> renumbering(c, src_tot_size1);
  int32_t *renumbering_data = renumbering.Data();
  auto lambda_set_renumbering = [=] __host__ __device__(int32_t i) {
    int32_t j = i % src_dim1, k = i / src_dim1, i_old = j * src_dim0 + k;
    renumbering_data[i] = i_old;
  };
  Eval(c, src_tot_size1, lambda_set_renumbering);

  RaggedShape src_no_axis0_renumbered = Renumber(src_no_axis0, renumbering);

  int32_t num_rows = src_dim1, row_splits_dim = num_rows + 1,
          row_ids_dim = src_tot_size1;
  std::vector<RaggedShapeDim> ans_axis0(1);
  Array1<int32_t> mem(c, row_splits_dim + row_ids_dim);
  ans_axis0[0].row_splits = mem.Range(0, row_splits_dim);
  ans_axis0[0].row_ids = mem.Range(0, row_ids_dim);
  ans_axis0[0].cached_tot_size = row_ids_dim;

  int32_t *mem_data = mem.Data();
  auto lambda_set_row_info = [=] __host__ __device__(int32_t i) {
    int32_t val;
    if (i >= row_splits_dim) {
      int32_t elem_idx = i - row_splits_dim;
      val = elem_idx / src_dim0;
    } else {
      int32_t row_idx = i;
      val = row_idx * src_dim0;
    }
    mem_data[i] = val;
  };
  Eval(c, row_splits_dim + row_ids_dim, lambda_set_row_info);
  RaggedShape temp(ans_axis0);
  return ComposeRaggedShapes(temp, src_no_axis0_renumbered);
}

}  // namespace k2
