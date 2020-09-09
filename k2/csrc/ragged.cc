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

Array1<int32_t> &RaggedShape::RowIds(int32_t axis) {
  K2_CHECK_GE(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  RaggedShapeDim &rsd = axes_[axis - 1];
  auto &row_splits = rsd.row_splits;
  auto &row_ids = rsd.row_ids;
  // there must be row_splits.Dim() >=1 according to the definition of
  // RaggedShapeDim.
  if (row_splits.Dim() != 1 && row_ids.Dim() == 0) {
    // create row_ids as it does not exist
    row_ids =
        Array1<int32_t>(row_splits.Context(), row_splits[row_splits.Dim() - 1]);
    const int32_t *row_splits_data = row_splits.Data();
    int32_t *row_ids_data = row_ids.Data();
    RowSplitsToRowIds(row_splits.Context(), row_splits.Dim() - 1,
                      row_splits_data, row_ids.Dim(), row_ids_data);
    // set cached_tot_size
    rsd.cached_tot_size = row_ids.Dim();
  }
  return row_ids;
}

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
void PrintRaggedShapePart(std::ostream &stream, RaggedShape &shape,
                          int32_t axis, int32_t begin_pos, int32_t end_pos) {
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() && begin_pos >= 0 &&
      begin_pos <= end_pos && end_pos <= shape.TotSize(axis));
  for (int32_t d = begin_pos; d < end_pos; d++) {
    if (axis == shape.NumAxes() - 1) {
      stream << d << " ";
    } else {
      stream << "[ ";
      const int32_t *row_splits = shape.RowSplits(axis + 1).Data();
      K2_DCHECK(d < shape.RowSplits(axis + 1).Dim());
      int32_t row_start = row_splits[d], row_end = row_splits[d + 1];
      PrintRaggedShapePart(stream, shape, axis + 1, row_start, row_end);
      stream << "] ";
    }
  }
}

// prints a RaggedShape as e.g. [ [ 0 1 ] [ 2 ] [] ].  Note, the 'values'
// are just the positions in the array, this is for readability.
std::ostream &operator<<(std::ostream &stream, RaggedShape &shape) {
  if (shape.Context()->GetDeviceType() != kCpu) {
    return stream << shape.To(GetCpuContext());
  } else {
    stream << "[ ";
    PrintRaggedShapePart(stream, shape, 0, 0, shape.Dim0());
    stream << "]";
    return stream;
  }
}

RaggedShape RaggedShapeFromTotSizes(int32_t num_axes, int32_t *tot_sizes) {
  // TODO
  std::vector<RaggedShapeDim> axes;
  return RaggedShape(axes);
}

RaggedShape RaggedShapeFromTotSizes(ContextPtr &c,
    int32_t num_axes, int32_t *tot_sizes) {
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

void RaggedShape::Populate() {}

RaggedShape RaggedShape::To(ContextPtr ctx) {
  return RaggedShape();
}

RaggedShape RaggedShape::Index(int32_t axis, int32_t i) {
  // only support `axis == 0` for now
  K2_CHECK_EQ(axis, 0);
  K2_CHECK_GE(i, 0);
  int32_t num_axes = NumAxes();
  K2_CHECK_GE(num_axes, 2);
  std::vector<RaggedShapeDim> axes(num_axes - 1);
  for (int32_t axis = 0; axis < num_axes; ++i) {
    // test
    //
  }
  RaggedShape shape(axes, true);
  return shape;
  // leave row_ids unset
}

RaggedShape RaggedShape2(Array1<int32_t> *row_splits,
    Array1<int32_t> *row_ids, int32_t cached_tot_size) {
  if (!row_splits && !row_ids) {
    K2_LOG(FATAL) << "At least one of row_splits and row_ids must be defined";
  }
  if (cached_tot_size != -1) {
    if (row_ids != nullptr) K2_CHECK_EQ(cached_tot_size, row_ids->Dim() - 1);
    if (row_splits != nullptr) {  // caution: next check may be slow...
      const auto &row_splits_ref = *row_splits;
      K2_CHECK_EQ(cached_tot_size, row_splits_ref[row_splits->Dim() - 1]);
    }
  }
  std::vector<RaggedShapeDim> axes(1);
  if (row_splits) axes[0].row_splits = *row_splits;
  if (row_ids) axes[0].row_ids = *row_ids;
  axes[0].cached_tot_size = cached_tot_size;
  return RaggedShape(axes);
}

RaggedShape ComposeRaggedShapes(RaggedShape &a,
                                RaggedShape &b) {
  if (a.NumElements() != b.Dim0()) {
    K2_LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: "
               << a.NumElements() << " vs. " << b.Dim0();
  }
  auto a_axes = a.Axes(), b_axes = b.Axes();
  std::vector<RaggedShapeDim> axes(a_axes.size() + b_axes.size());
  size_t a_size = a_axes.size(), b_size = b_axes.size();
  for (size_t i = 0; i < a_size; i++)
    axes[i] = a_axes[i];
  for (size_t i = 0; i < b_size; i++)
    axes[i + a_size] = b_axes[i];
  return RaggedShape(axes);
}

RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2) {
  // This is a slightly lazy implementation, could save a couple copies of
  // metadata by
  // implementing it directly.
  auto shape1 = RaggedShape2(row_splits1, row_ids1, cached_tot_size1);
  auto shape2 = RaggedShape2(row_splits2, row_ids2, cached_tot_size2);
  return ComposeRaggedShapes(shape1, shape2);
}

RaggedShapeIndexIterator RaggedShape::Iterator() {
  return RaggedShapeIndexIterator(*this);
}

Array1<int32_t *> GetRowSplitsPtr(RaggedShape &src) {
  Array1<int32_t *> array;
  // TODO(haowen): implement
  return array;
}

int32_t RaggedShape::operator[](const std::vector<int32_t> &indexes) {
  K2_CHECK_EQ(indexes.size(), NumAxes());
  K2_CHECK_EQ(Context()->GetDeviceType(), kCpu);
  int32_t cur_idx = indexes[0];
  for (int32_t i = 1; i < NumAxes(); i++) {
    Array1<int32_t> &row_splits = axes_[i - 1].row_splits;
    K2_CHECK(cur_idx >= 0 && cur_idx + 1 < row_splits.Dim());
    cur_idx = row_splits[cur_idx];
    cur_idx += indexes[i];
  }
  return cur_idx;
}

void GetRowInfo(RaggedShape &src,
                Array1<int32_t*> *row_splits,
                Array1<int32_t*> *row_ids) {
  // TODO
}

void GetInfoMulti(int32_t num_src, RaggedShape **src,
    Array2<int32_t *> *row_splits, Array2<int32_t *> *row_ids,
    Array2<int32_t *> *offsets, std::vector<int32_t> *tot_sizes) {
  // TODO: implement this

}

void GetRowInfoMulti(int32_t num_src, RaggedShape **src,
                     Array2<int32_t *> *row_splits,
                     Array2<int32_t *> *row_ids) {
  // TODO
}

struct RowInfoWithOffsets {
  int32_t *row_splits;
  int32_t *row_ids;
  int32_t num_rows;
  int32_t num_elems;
  int32_t row_splits_offset;
  int32_t row_ids_offset;
};

RaggedShape RemoveAxis(RaggedShape &src, int32_t axis) {
  K2_CHECK_GT(src.NumAxes(), 2);
  K2_CHECK(axis >= 0 && axis < src.NumAxes());

  // note, `axes` is of dim src.NumAxes() - 1.
  // Also note: axes_in[i] pertains to the relationship between
  // axes i and i+1 in the source.
  src.Populate();
  const std::vector<RaggedShapeDim> &axes_in = src.Axes();

  std::vector<RaggedShapeDim> axes_out(axes_in.size() - 1);

  for (int32_t i = 0; i < axis - 1; i++) axes_out[i] = axes_in[i];

  if (axis > 0 && axis + 1 < src.NumAxes()) {
    axes_out[axis - 1].row_ids =
        axes_in[axis - 1].row_ids[axes_in[axis].row_ids];
    axes_out[axis - 1].row_splits =
        axes_in[axis].row_splits[axes_in[axis - 1].row_splits];
  }
  for (int32_t i = axis; i < axes_out.size(); i++) axes_out[i] = axes_in[i + 1];
  return RaggedShape(axes_out);
}

RaggedShape Stack(int32_t num_srcs, RaggedShape **src, int32_t axis) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK(axis >= 0 && axis <= 1);

  ContextPtr c = src[0]->Context();

  std::vector<RaggedShape> unsqueezed(num_srcs);
  std::vector<RaggedShape *> unsqueezed_ptrs(num_srcs);
  {
    ParallelRunner pr(c);
    for (int32_t i = 0; i < num_srcs; i++) {
      With w(pr.NewStream());
      unsqueezed[i] = Unsqueeze(*src[i], 0);
      unsqueezed_ptrs[i] = &unsqueezed[i];
    }
    // destructor will wait for work in those launched streams to finish.
    // (well it won't actually wait, but it will force the current stream to
    // wait.
  }

  RaggedShape ans = Append(num_srcs, &(unsqueezed_ptrs[0]), 0);
  if (axis == 1) ans = Transpose(ans);
  return ans;
}

}  // namespace k2
