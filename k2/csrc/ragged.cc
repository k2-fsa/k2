/**
 * @brief
 * ragged
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/ragged.h"
#include "k2/csrc/math.h"

namespace k2 {

RaggedShape RandomRaggedShape(int32_t min_num_axes,
                              int32_t max_num_axes,
                              int32_t min_num_elements,
                              int32_t max_num_elements) {
  ContextPtr c = GetCpuContext();
  K2_CHECK(min_num_axes >= 2 && max_num_axes >= min_num_axes &&
           min_num_elements >= 0 && max_num_elements >= min_num_elements);
  int32_t num_axes = RandInt(min_num_axes, max_num_axes);

  int32_t done_repeats = 0;

  std::vector<RaggedShapeDim> axes(static_cast<unsigned long>(num_axes - 1));
  int32_t num_elements = RandIntGeometric(min_num_elements, max_num_elements);
  for (int32_t axis = num_axes - 2; axis >= 0; axis--) {
    // this axis will have row_ids of length num_elements and row_splits of length
    // to be determined.

    int32_t cur_row_split = 0;
    std::vector<int32_t> row_splits_vec;
    row_splits_vec.push_back(cur_row_split);
    // The reason for "|| RandInt(0, 2) == 0)" is so that even if there
    // are no elements we can still potentially generate empty row-splits.
    while (cur_row_split < num_elements || RandInt(0, 2) == 0) {
      int32_t split_size = RandIntGeometric(0, num_elements - cur_row_split);
      cur_row_split += split_size;
      // sometimes we have a bunch of empty rows in a row (this will test out
      // more of the code).
      int32_t num_repeats = 1;
      if (split_size == 0 && RandInt(0, 30) == 0 &&
          done_repeats == 0) {
        num_repeats = RandIntGeometric(1, 128);
        done_repeats = 0;
      }
      row_splits_vec.push_back(cur_row_split);
    }
    axes[axis].row_splits = Array1<int32_t>(c, row_splits_vec);
    axes[axis].cached_tot_size = static_cast<size_t>(num_elements);
//    num_elements = axes[axis].row_splits;
  }
  return RaggedShape(axes);
}


// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos < shape.TotSize(axis).


void PrintRaggedShapePart(std::ostream &stream,
                          RaggedShape &shape,
                          int32_t axis,
                          int32_t begin_pos, int32_t end_pos) {
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() &&
           begin_pos >= 0 && begin_pos <= end_pos &&
           end_pos <= shape.TotSize(axis));
  for (int32_t d = begin_pos; d < end_pos; d++) {
    if (axis == shape.NumAxes() - 1) {
      stream << d << " ";
    } else {
      stream << "[ ";
      const int32_t *row_splits = shape.RowSplits(axis + 1).Data();
      K2_DCHECK(d < shape.RowSplits(axis + 1).Dim());
      int32_t row_start = row_splits[d],
          row_end = row_splits[d+1];
      PrintRaggedShapePart(stream, shape, axis + 1,
                           row_start, row_end);
      stream << "] ";
    }
  }
}

// prints a RaggedShape as e.g. [ [ 0 1 ] [ 2 ] [] ].  Note, the 'values'
// are just the positions in the array, this is for readability.
std::ostream &operator<<(std::ostream &stream, const RaggedShape &shape) {
  if (shape.Context()->GetDeviceType() != kCpu) {
    return stream << shape.To(GetCpuContext());
  } else {
    stream << "[ ";
    PrintRaggedShapePart(stream, shape, 0, 0, shape.Dim0());
    stream << "]";
    return stream;
  }
}

RaggedShapeFromTotSizes(ContextPtr &c, int32_t num_axes, int32_t *tot_sizes) {
  std::vector<RaggedShapeDim> axes(num_axes - 1);
  // In future we might choose to allocate everything in one big array, to avoid
  // multiple allocations, but for now just do it the simple way.
  for (int32_t axis = 1; axis < num_axes; axis++) {
    axes[axis-1].row_splits = Array1<int32_t>(c, tot_sizes[axis - 1] + 1);
    axes[axis-1].row_ids = Array1<int32_t>(c, tot_sizes[axis] + 1);
    axes[axis-1].cached_tot_size = tot_sizes[axis];
  }
  return RaggedShape(axes);
}

const std::vector<int32_t> &RaggedShapeIndexIterator::Value() {
  return idx_;
}

int32_t k2::RaggedShape::MaxSize(int32_t axis) {
  return 0;
}

k2::RaggedShape k2::RaggedShape::Index(int32_t axis, int32_t value) {
  return k2::RaggedShape();
}

int32_t k2::RaggedShape::operator[](const std::vector<int32_t> &indexes) {
  return nullptr;
}

k2::RaggedShapeIndexIterator k2::RaggedShape::Iterator() {
  return k2::RaggedShapeIndexIterator(*this);
}
void k2::RaggedShape::Populate() {}

void k2::RaggedShape::Check() {}

RaggedShape RaggedShape::To(ContextPtr ctx) {
  return k2::RaggedShape();
}

void RaggedShape::Check() {}

RaggedShape RaggedShape::Index(int32_t axis, int32_t value) {
  return RaggedShape();
}

void RaggedShape::Check() {
  Context c = Context();
  int32_t num_axes = axes_.size();
  for (int32_t axis = 0; axis < axes_.size(); axis++) {
    RaggedShapeDim &rsd = axes_[axis];
    CHECK_GE(rsd.row_splits.Dim(), 0);
    if (rsd.cached_tot_size >= 0) {
      K2_CHECK(row_splits.Dim() == 0 ||
               rsd.cached_tot_size == row_splits[row_splits.Dim() - 1]);
      K2_CHECK(row_ids.Dim() == 0 ||
               rsd.cached_tot_size == row_ids.Dim());
    } else {
      K2_ASSERT(rsd.cached_tot_size == -1 && row_ids.Dim() == 0);
    }

    int32_t num_elems;
    { // Check row_splits.

      // meta[0] is a bool, ok == 1, not-ok == 0.
      // meta[1] will contain the number of row_splits.
      Array1<int32_t> meta(c, 2, 1);
      int32_t *ok_data = meta.Data(),
          *num_elems_data = ok_data + 1;
          *row_splits_data = rsd.row_splits.Data();
      int32_t num_rows = rsd.row_splits.Dim() - 1;

      auto lambda_check_row_splits = __host__ __device__ [=] (int32_t i) -> void {
         int32_t this_idx = row_splits_data[i];
         if (i == 0 && this_idx != 0) *ok_data = 0;
         if (i < num_rows) {
           int32_t next_idx = row_splits_data[i + 1];
           if (next_idx <= this_idx)
             *ok_data = 0;
         } else {
           K2_CHECK(i == num_rows);
           *num_elems_data = this_idx;
         }
      };
      Eval(c, num_rows + 1, lambda_check_row_splits);
      meta = meta.To(CpuContext());
      num_elems = meta[1];
      int32_t ok = meta[0];
      if (!ok) {
        K2_LOG(FATAL) << "Problem validating row-splits: for axes_["
                      << axis << "], row_splits = "
                      << rsd.row_splits.Dim();
      }
      if (rsd.cached_tot_size > 0 && rsd.cached_tot_size != num_elems) {
        K2_LOG(FATAL) << "Problem validating row-splits: for axes_["
                      << axis << "], row_splits[-1] = "
                      << num_elems << " but cached_tot_size == "
                      << rsd.cached_tot_size;
      }
    }
    if (axis + 1 < axes_.size()) {
      int32_t next_num_rows = axes_[axis + 1].row_splits.Dim();
      if (num_elems != next_num_rows) {
        K2_LOG(FATAL) << "Ragged shape has num_elems for axes_["
                      << axis << "] == " << num_elems
                      << " and num-rows for axes_[" << (axis+1)
                      << "] == " << next_num_rows;
      }
    }

    if (rsd.row_ids.Dim() != 0) { // check row_ids.
      K2_CHECK(IsCompatible(rsd.row_ids, rsd.row_splits));
      // 1st elem is `ok` (1 or 0); 2nd elem is location of bad index
      // into row_splits
      Array1<int32_t> meta(c, 1, 2);
      int32_t *ok_data = meta.Data(),
          *bad_index_data = ok_data + 1;

      const int32_t *row_splits_data = rsd.row_splits.Data(),
          *row_ids_data = rsd.row_ids.Data();
      int32_t num_elems = rsd.row_ids.Dim(),
          num_rows = rsd.row_splits.Dim() - 1;

      auto lambda_check_row_ids = __host__ __device__ [=] (int32_t i) -> void {
         int32_t this_row = rsd.row_ids_data[i];
         if (this_row < 0 ||
             this_row >= num_rows ||
             i < row_splits_data[this_row] ||
             i >= row_splits_data[this_row + 1]) {
           *ok_data = 0;
           *bad_index_data = i;
         }
      };
      // TODO: could do this and the other one in separate streams.
      Eval(c, num_elems, lambda_check_row_ids);
      meta = meta.To(CpuContext());  // since we have 2 accesses, this should be
                                     // faster.
      int32_t ok = meta[0];
      if (!ok) {
        K2_LOG(FATAL) << "Problem validating row-ids: for axes_[" << axis
                      << "], row_splits = " << rsd.row_splits
                      << ", row_ids = " << rsd.row_ids
                      << ", see index " << meta[1]
                      << " of row_ids, whose dim is " << rsd.row_ids.Dim();
      }
    }
    if (axis + 1 < axes_.size()) {
      K2_CHECK(IsCompatible(rsd.row_splits, axes_[axis + 1].row_splits));
    }
  }
}


RaggedShape RaggedShape2(Array1<int32_t> *row_splits,
                         Array1<int32_t> *row_ids,
                         int32_t cached_tot_size) {
  if (!row_splits && !row_ids) {
    LOG(FATAL) << "At least one of row_splits and row_ids must be defined";
  }
  if (cached_tot_size != -1) {
    if (row_ids != nullptr)
      CHECK(cached_tot_size == row_ids->Size()-1);
    if (row_splits != nullptr)  // caution: next check may be slow...
      CHECK(cached_tot_size == row_splits[row_splits->Size()-1]);
  }
  axes_.resize(1);
  if (row_splits)
    axes_[0].row_splits = *row_splits;
  if (row_ids)
    axes_[0].row_ids = *row_ids;
  axes_[0].cached_tot_size = cached_tot_size;
}

RaggedShape ComposeRaggedShapes(RaggedShape &a,
                                RaggedShape &b) {
  if (a.NumElements() != b.Dim0()) {
    LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: "
               << a.NumElements() << " vs. " << b.Dim0();
  }
  std::vector<RaggedShapeDim> axes(a.axes_.size() + b.axes_.size());
  size_t a_size = a.axes_.size(), b_size = b.axes_.size();
  for (size_t i = 0; i < a_size; i++)
    axes[i] = a.axes_[i];
  for (size_t i = 0; i < b_size; i++)
    axes[i + a_size] = b.axes_[i];
  return RaggedShape(axes);
}


RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2) {
  // This is a slightly lazy implementation, could save a couple copies of metadata by
  // implementing it directly.
  return ComposeRaggedShapes(RaggedShape2(row_splits1, row_ids1, cached_tot_size1),
                             RaggedShape2(row_splits2, row_ids2, cached_tot_size2));
}

RaggedShapeIndexIterator RaggedShape::Iterator() {
  return RaggedShapeIndexIterator(*this);
}

int32_t RaggedShape::operator[](const std::vector<int32_t> &indexes) {
  K2_CHECK(indexes.size() == NumAxes());
  K2_CHECK(Context().GetDeviceType() == kCpu);
  int32_t cur_idx = indexes[0];
  for (int32_t i = 1; i < NumAxes(); i++) {
    Array<int32_t> &row_splits = axes_[i-1].row_splits;
    K2_CHECK(cur_idx >= 0 && cur_idx + 1 < row_splits.Dim());
    cur_idx = row_splits[cur_idx];
    cur_idx += indexes[i];
  }
  return cur_idx;
}


// See declaration in ragged.h for documentation of its purpose and interface.
RaggedShape Unsqueeze(const RaggedShape &src, int32_t axis) {
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

  ContextPtr c = src.GetContext();
  K2_CHECK(axis >= 0 && axis <= src.NumAxes());

  const std::vector<RaggedShapeDim> &axes = src.Axes();

  int32_t num_axes_in = src.NumAxes();

  // Note: in RaggedShape, the vector of RaggedShapeDim is of length num_axes - 1,
  // so the output will have one more axis than the input.
  std::vector<RaggedShapeDim> axes_out(num_axes_in);


  int32_t row_splits_dim, row_ids_dim;
  Array1<int32_t> mem;

  if (axis == 0) {
    row_splits_dim = 2;  // e.g. [ 0 5 ]
    row_ids_dim = src.Dim0();  // e.g. [ 0 0 0 0 0 ]
    mem = Array1<int32_t>(c, row_splits_dim + row_ids_dim);
    int32_t *mem_data = mem.Data();
    auto lambda_set_mem [=] __host__ __device__ (int32_t i) -> void {
      if (i == 1) mem_data[i] = row_ids_dim;
      else mem_data[i] = 0;
    };
    Eval(c, mem.Dim(), lambda_set_mem);
         axes_out[0].row_splits = mem.Ra
             nge(0, 2);
  } else {
    int32_t tot_size = src.TotSize(axis - 1);
    row_splits_dim = tot_size + 1;
    row_ids_dim = tot_size;
    mem = Array1<int32_t>(c, row_splits_dim + row_ids_dim);
    int32_t *mem_data = mem.Data();
    auto lambda_set_mem2 [=] __host__ __device__ (int32_t i) -> void {
        mem_data[i] = i % (tot_size + 1);
    };
    Eval(c, mem.Dim(), lambda_set_mem2);
  }
  axes_out[axis].row_splits = mem.Range(0, row_splits_dim);
  axes_out[axis].row_ids = mem.Range(row_splits_dim, row_ids_dim);
  for (int32_t i = 0; i < axis; i++)
    axes_out[axis] = axes_in[axis];
  // Note: the returned array has `num_axes_in + 1` axes, so its
  // array of RaggedShapeDim is of length `num_axes_in`.
  for (int32_t i = axis + 1; i < num_axes_in; i++)
    axes_out[axis] = axes_in[axis - 1];
  return RaggedShape(axes_out);
}

RaggedShape Renumber(const RaggedShape &src, const Array1<int32_t> &new2old) {
  ContextPtr c = src.Context();
  K2_ASSERT(IsCompatible(src, new2old));
  int32 num_axes = src.NumAxes(),
      dim0 = src.Dim0();
  K2_ASSERT(new2old.Dim() == dim0);
  std::vector<int32_t> tot_sizes_out(num_axes);
  for (int32_t axis = 0; axis < num_axes; axis++)
    tot_sizes_out[axis] = src.TotSize(axis);
  // the arrays in `ans` will be the same sizes as those in `src`.
  RaggedShape ans = RaggedShapeFromTotSizes(c, tot_sizes_out);

  src.Populate();
  Array2<int32_t> old_offsets(c, num_axes, dim0 + 1),
      new_offsets(c, num_axes, dim0 + 1);
  auto old_offsets_acc = old_offsets.Accessor(),
      new_offsets_acc = new_offsets.Accessor();
  Array<int32_t*> row_splits_ptrs = GetRowSplitsPtrs(src);
  int32_t *row_splits_ptrs_data = row_splits_ptrs.Data();

  // Set old_offsets
  auto lambda_get_old_offsets = [=] __host__ __device__ (int32_t i) {
     // 0 <= i <= dim0
     int32_t cur_offset = i;
     for (int32_t axis = 0; axis < num_axes; axis++) {
       old_offsets_acc(0, i) = cur_offset;
       if (axis + 1 == num_axes)
         return;
       cur_offset = row_splits_ptrs_data[axis][cur_offset];
     }
  };
  Eval(c, dim0 + 1, lambda_get_old_offsets);
  const int32_t *new2old_data = new2old.Data();
  auto lambda_get_new_sizes = [=] __host__ __device__ (int32_t axis, int32_t new_i) {
     // 0 <= axis < num_axes;  0 <= new_i < dim0
     int32_t old_i = new2old_data[new_i],
        this_old_offset = old_offsets_acc(axis, old_i),
        next_old_offset = old_offsets_acc(axis, old_i + 1),
        size = next_old_offset - this_old_offset;
     new_offsets_acc(axis, i) = size;
  };
  Eval(c, num_axes, dim0, lambda_get_new_offsets);
  ExclusiveSum(new_offsets, &new_offsets);
  // Now new_offsets contains the offsets, not the sizes.

  ParallelRunner pr(c);
  std::vector<cudaStream_t> streams(num_axes);
  int32_t num_jobs = dim0 * 2;  // note: this formula is not a heuristic; it's
                                // how TaskRedirect works..
  Array2<TaskRedirect> task_redirects(c, num_axes, num_jobs);
  auto task_redirects_acc = task_redirects.Accessor();
  for (int32_t axis = 0; axis < num_axes; axis++) {
    cudaStream_t stream = streams[axis] = pr.NewStream();
    With(streams[axis] = pr.NewStream()) _w;
    const int32_t *new_offsets_ptr = new_offsets_acc.Row(axis);
    TaskRedirect *task_redirect_ptr = task_redirects_acc.Row(axis);
    GetTaskRedirect(c, dim0, new_offsets_ptr, task_redirect_ptr);
  }


  for (int32_t axis = 0; axis < num_axes - 1; axis++) {

    {
      int32_t *this_new_row_splits = ans.RowSplits(axis).Data();
      const int32_t *this_old_row_splits = src.RowSplits(axis).Data();

      auto lambda_set_row_splits = [=] __host__ __device__ (
          int32_t new_idx, int32_t num_threads, int32_t thread_idx) -> void {
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
      int32_t min_threads_per_job = 2,
          tot_work = tot_sizes_out[axis],
          target_num_loops = (tot_work > 1000000 ? 4 : 2);
      bool include_final_task = false;
      EvalWithRedirect(streams[axis], num_jobs,
                       task_redirects_acc.Row(axis), min_threads_per_job,
                       tot_work, target_num_loops, include_final_task,
                       lambda_set_row_splits);
    }

    {
      int32_t *this_new_row_ids = ans.RowIds(axis).Data();
      const int32_t *this_old_row_ids = src.RowIds(axis).Data();

      auto lambda_set_row_ids = [=] __host__ __device__ (
          int32_t new_idx, int32_t num_threads, int32_t thread_idx) -> void {
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
           this_new_row_ids[this_new_offset + thread_idx] =
               next_value_offset;
         }
      };
      int32_t min_threads_per_job = 2,
          tot_work = tot_sizes_out[axis],
          target_num_loops = (tot_work > 1000000 ? 4 : 2);
      EvalWithRedirect(streams[axis], num_jobs,
                       task_redirects_acc.Row(axis), min_threads_per_job,
                       tot_work, target_num_loops, lambda_set_row_splits);
    }
  }
#ifndef NDEBUG
  ans.Check();
#endif
  return ans;
}



/*
  Returns a CPU array of shape (src[0]->NumAxes()+1) by (num_srcs + 1), where
  each row is the exclusive-sum of the TotSize() of the respective sources,
  on the previous axis (or 1 for axis 0).  Specifically: it's the same
  as setting ans(i,j) to (i == 0 ? 1 : src[j]->TotSize(i-1)), and then
  doing an exclusive-sum on each row of i.

     @param [in] num_srcs  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes whose sizes we want.  Must all have the
                      same NumAxes().
     @return   Returns a freshly allocated CPU Array2<int32_t> of dimension
               src[0]->NumAxes() by (num_srcs + 1), where each
               row is the exclusive-sum of the TotSize() of the respective
               sources, on that axis.  Its last column contains the totals.

 */
inline Array2<int32_t> GetOffsets(int32_t num_srcs, RaggedShape **src) {
  //  src_offsets[i,j]  == src_offsets.Data()[i*num_axes_in + j] contains:
  //          sum(0 <= k < i) src[k]->TotSize(j).
  int23_t num_axes_in = src[0]->NumAxes()
  Array2<int32_t> src_offsets(CpuContext(), num_srcs + 1, num_axes_in);
  int32_t *src_offsets_data = src_offsets.Data();
  int32_t src_offsets_stride0 = num_srcs + 1;
  DCHECK_EQ(src_offsets.Stride0(), src_offsets_stride0);

  for (int32_t axis = 0; axis  < num_axes_in; axis++) {
    int32_t sum = 0;
    for (int32_t i = 0; i <= num_srcs; i++) {
      src_offsets_data[i*src_offsets_stride0 + axis] = sum;
      if (i < num_srcs) {
        sum += (axis == 0 ? 1 : src->TotSize(axis - 1));
      }
    }
  }
  return src_offsets;
}



/*
  TODO: fix this documentation...

  Extract meta-info from the shape (this will include populating any row_ids and
  row_splits that were not already populated).  This is used inside algorithms
  when we need to transfer meta-info to GPU.

     @param [in]   src   Ragged shape that we're extracting meta-info from
     @param [out] row_splits  This will be set to an array of size src.NumAxes()-1,
                         containing pointers to the row_splits' Data() vectors.
                         The array will be on the same device as `src`.
     @param [out] row_splits  This will be set to an array of size src.NumAxes()-1,
                         containing pointers to the row_splits' Data() vectors.
                         The array will be on the same device as `src`.


     Host (i.e. CPU memory) array of RowInfo that we're
                         writing to (but the pointers inside them are to the
                         memory of the device used in shape.Context());
                         must be the start of an array of length
                         shape.NumAxes().
                         Note: element 0 of the output `ptrs` contains a
                         row_splits with [ 0, shape.Dim0() ]
                         and the row_ids are [ 0 0 0 (repeats shape.Dim0() times) 1 ]
*/
// outputs have dims src.NumAxes() - 1.
void GetRowInfo(RaggedShape &src,
                Array1<int32_t*> *row_splits,
                Array1<int32_t*> *row_ids) {
  // TODO
}


RaggedShape Append(int32_t num_srcs, RaggedShape **src, int32_t axis) {
  K2_ASSERT(axis == 0 &&  "Append() with axis > 0 not yet supported");
  K2_CHECK_GT(num_srcs, 0);

  int32_t num_axes = src->NumAxes();
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

  RaggedShape ans = RaggedShapeFromTotSizes(c, tot_sizes_out);
  Array1<int32_t*> dest_row_splits, dest_row_ids;
  GetRowInfo(ans, &dest_row_splits, &dest_row_ids);

  Array2<int32_t*> src_row_splits, src_row_ids;
  GetRowInfoMulti(num_srcs, src, &src_row_splits, &src_row_ids);

  if (c.DeviceType() != kCpu)
    offsets = offsets.To(c);

  int32_t **dest_row_splits_data = dest_row_splits.Data(),
      **dest_row_ids_data = dest_row_ids.Data();
  auto src_row_splits_acc = src_row_splits.Accessor(),
      src_row_ids_acc = src_row_ids.Accessor();
  offsets_acc = offsets.Accessor();  // on GPU now (if we're using one)

  ParallelRunner pr(c);
  std::vector<cudaStream_t> streams(num_axes_in + 1);
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
    GetTaskRedirect(stream[axis], num_srcs, offsets,
                    task_redirects_acc.Row(axis));
  }

  for (int32_t axis = 0; axis < num_axes - 1; axis++) {
    RowInfo *dest_info = dest_row_ptrs_data + axis,
        *src_info = src_row_ptrs_data + axis;

    {
      // first set the row-splits.
      TaskRedirect *tr = &(task_redirects_acc(axis, 0));

      const int32_t **this_src_row_splits = &(src_row_splits_acc(axis, 0)),
          **this_src_row_ids = &(src_row_ids_acc(axis, 0));
      int32_t *this_dest_row_splits = ans.RowSplits(axis + 1),
          *this_dest_row_ids = ans.RowIds(axis + 1);
      const int32_t *offsets_this_axis = &(offsets_acc(axis, 0)),
          *offsets_next_axis = &(offsets_acc(axis + 1, 0)),

      {
        auto lambda_set_row_splits = [=] __host__ __device__ (
            int32_t src_idx, int32_t num_threads, int32_t thread_idx) -> void {
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

        int32_t min_threads_per_job = 2,
            tot_work = tot_sizes_out[axis],
            target_num_loops = (tot_work > 1000000 ? 4 : 2);
        EvalWithRedirect(stream[axis], num_jobs,
                         task_redirects_acc.Row(axis), min_threads_per_job,
                         tot_work, target_num_loops, lambda_set_row_splits);
      }

      { // set the row-ids
        auto lambda_set_row_ids = [=] __host__ __device__ (
          int32_t src_idx, int32_t num_threads, int32_t thread_idx) -> void {
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
          if (src_idx == num_srcs - 1)
             num_elems++;
          for (; thread_idx <= num_elems; thread_idx += num_threads) {
            this_dest_row_ids[this_offset + thread_idx] =
                this_value_offset + src_row_ids_ptr[thread_idx];
          }
        };
        int32_t min_threads_per_job = 2,
            tot_work = tot_sizes_out[axis+1],
            target_num_loops = (tot_work > 1000000 ? 4 : 2);
        bool include_final_task = false;
        EvalWithRedirect(stream[axis+1], num_jobs,
                         task_redirects_acc.Row(axis+1), min_threads_per_job,
                         tot_work, target_num_loops, include_final_task,
                         lambda_set_row_ids);
      }
    }
  }
  return ans;
}


/*
  Get some meta-info for an array of RaggedShape, and transfer them
  to the
  device that `src` is located on

     @param [in] num_src  Number of source arrays to process.
     @param [in] src      Source arrays.  Let num_axes be src[0]->NumAxes().
     @param [in] row_splits  Output array of row_splits pointers,
                          will be of dimension num_axes-1 by num_src
     @param [in] row_splits  Output array of row_splits pointers,
                          will be of dimension num_axes-1 by num_src
     @param [out] offsets   Output array of `offsets` pointers,
                          will be of dimension num_axes by num_src+1;
                          these are the exclusive-sum of the TotSize(axis)
                          of the respective sources.
     @param [out] tot_sizes  The last column of `offsets`, as a std::vector
*/
void GetInfoMulti(int32_t num_src,
                  RaggedShape **src,
                  Array2<int32_t*> *row_splits,
                  Array2<int32_t*> *row_ids,
                  Array2<int32_t*> *offsets,
                  std::vector<int32_t> *tot_sizes);



struct RowInfoWithOffsets {
  int32_t *row_splits;
  int32_t *row_ids;
  int32_t num_rows;
  int32_t num_elems;
  int32_t row_splits_offset;
  int32_t row_ids_offset;
};


RaggedShape RemoveAxis(RaggedShape &src, int32_t axis) {
  CHECK_GT(src.NumAxes(), 2);
  CHECK(axis >= 0 && axis < src.NumAxes());

  // note, `axes` is of dim src.NumAxes() - 1.
  // Also note: axes_in[i] pertains to the relationship between
  // axes i and i+1 in the source.
  src.Populate();
  std::vector<RaggedShapeDim> &axes_in = src.Axes();

  std::vector<RaggedShapeDim> axes_out(axes_in.size() - 1);

  for (int32_t i = 0; i < axis - 1; i++)
    axes_out[i] = axes_in[i];

  if (axis > 0 && axis + 1 < src.NumAxes()) {
    axes_out[axis - 1].row_ids = axes_in[axis - 1].row_ids[axes_in[axis].row_ids];
    axes_out[axis - 1].row_splits = axes_in[axis].row_splits[axes_in[axis - 1].row_splits];
  }
  for (int32_t i = axis; i < axes_out.size(); i++)
    axes_out[i] = axes_in[i + 1];
  return RaggedShape(axes_out);
}


// transpose axes 0 and 1.
RaggedShape Transpose(RaggedShape &src) {
  K2_CHECK(src.NumAxes() > 2);
  int32_t src_dim0 = src.Dim0(),
      src_tot_size1 = src.TotSize(1),
      src_dim1 = src_dim0 % src_dim1;
  if (src_dim0 % src_tot_size1 != 0) {
    K2_LOG(FATAL) << "Transpose(): all dims on axis 0 must be the same.";
  }
  RaggedShape src_no_axis0 = RemoveAxis(src, 0);
  assert(src_no_axis0.Dim0() == src_tot_size1);
  Context c = src.Context();
  // `renumbering` is a `new2old` map, that maps from the first index in src_no_axis0_renumbered
  // to the first index into src_no_axis0.
  Array1<int32_t> renumbering(c, src_tot_size1);
  int32_t *renumbering_data = renumbering.Data();
  auto lambda_set_renumbering = [=] __host__ __device__ (int32_t i) {
      int32_t j = i % src_dim1,
          k = i / src_dim1,
          i_old = j * src_dim0 + k;
      renumbering_data[i] = i_old;
  };
  Eval(c, src_tot_size1, lambda_set_renumbering);

  RaggedShape src_no_axis0_renumbered = Renumber(src_no_axis0, renumbering);

  int32_t num_rows = src_dim1,
      row_splits_dim = num_rows + 1,
      row_ids_dim = src_tot_size1;
  std::vector<RaggedShapeDim> ans_axis0(1);
  Array1<int32_t> mem(c, row_splits_dim + row_ids_dim);
  ans_axis0[0].row_splits = mem.Range(0, row_splits_dim);
  ans_axis0[0].row_ids = mem.Range(0, row_ids_dim);
  ans_axis0[0].cached_tot_size = row_ids_dim;

  int32_t *mem_data = mem.Data();
  auto lambda_set_row_info = [=] __host__ __device__ (int32_t i) {
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

RaggedShape Stack(int32_t num_srcs, RaggedShape **src, int32_t axis) {
  CHECK_GT(num_srcs, 0);
  CHECK(axis >= 0 && axis <= 1);;

  ContextPtr c = src[0]->Context();

  std::vector<RaggedShape> unsqueezed(num_srcs);
  std::vector<RaggedShape*> unsqueezed_ptrs(num_srcs);
  {
    ParallelRunner pr(c);
    for (int32_t i = 0; i < num_srcs; i++) {
      With(pr.NewStream()) _w;
      unsqueezed[i] = Unsqueeze(src, 0);
      unsqueezed_ptrs[i] = &unsqueezed[i];
    }
    // destructor will wait for work in those launched streams to finish.
    // (well it won't actually wait, but it will force the current stream to wait.
  }

  RaggedShape ans = Append(num_srcs, &(unsqueezed_ptrs[0]), 0);
  if (axis = 1)
    ans = Transpose(ans);
  return ans;
}




}  // namespace k2
