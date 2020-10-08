/**
 * @brief
 * ragged
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cub/cub.cuh>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
namespace {

// will be used in RaggedShape::MaxSize(int32_t axis) to call
// cub::DeviceReduce::Max
struct RowSplitsDiff {
  const int32_t *row_splits_data;
  explicit RowSplitsDiff(const int32_t *row_splits)
      : row_splits_data(row_splits) {}
  // operator[] and operator+ are required by cub::DeviceReduce::Max
  __device__ int32_t operator[](int32_t i) const {
    return row_splits_data[i + 1] - row_splits_data[i];
  }
  __device__ RowSplitsDiff operator+(int32_t n) const {
    RowSplitsDiff tmp(*this);
    tmp.row_splits_data += n;
    return tmp;
  }
};

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

namespace std {
// vaule_type is required by cub::DeviceReduce::Max
template <>
struct iterator_traits<::RowSplitsDiff> {
  typedef int32_t value_type;
};
}  // namespace std

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
  std::vector<RaggedShapeDim> axes(num_axes - 1);
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

// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos < shape.TotSize(axis).

void PrintRaggedShapePart(std::ostream &stream, RaggedShape &shape,
                          int32_t axis, int32_t begin_pos, int32_t end_pos) {
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() && begin_pos >= 0 &&
           begin_pos <= end_pos && end_pos <= shape.TotSize(axis));
  for (int32_t d = begin_pos; d < end_pos; ++d) {
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

Array1<int32_t> &RaggedShape::RowIds(int32_t axis) {
  K2_CHECK_GT(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  RaggedShapeDim &rsd = axes_[axis - 1];
  auto &row_splits = rsd.row_splits;
  auto &row_ids = rsd.row_ids;
  // there must be row_splits.Dim() >=1 according to the definition of
  // RaggedShapeDim.
  K2_CHECK_GE(row_splits.Dim(), 1);
  if (!row_ids.IsValid()) {
    if (rsd.cached_tot_size < 0)
      rsd.cached_tot_size = row_splits[row_splits.Dim() - 1];
    // create row_ids as it does not exist
    row_ids = Array1<int32_t>(Context(), rsd.cached_tot_size);
    const int32_t *row_splits_data = row_splits.Data();
    int32_t *row_ids_data = row_ids.Data();
    RowSplitsToRowIds(Context(), row_splits.Dim() - 1, row_splits_data,
                      row_ids.Dim(), row_ids_data);
  }
  return row_ids;
}

int32_t RaggedShape::MaxSize(int32_t axis) {
  K2_CHECK_GT(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  const auto &row_splits = axes_[axis - 1].row_splits;
  const int32_t num_rows = row_splits.Dim() - 1;
  if (num_rows == 0) return 0;
  const int32_t *row_splits_data = row_splits.Data();
  ContextPtr c = Context();
  if (c->GetDeviceType() == kCpu) {
    int32_t max_value = 0;
    for (int32_t i = 0; i < num_rows; ++i) {
      int32_t value = row_splits_data[i + 1] - row_splits_data[i];
      if (value > max_value) max_value = value;
    }
    return max_value;
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    ::RowSplitsDiff row_splits_diff(row_splits_data);
    Array1<int32_t> max_array(Context(), 1, 0);
    int32_t *max_value = max_array.Data();

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // the first time is to determine temporary device storage requirements
    K2_CUDA_SAFE_CALL(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                                             row_splits_diff, max_value,
                                             num_rows, c->GetCudaStream()));
    void *deleter_context;
    d_temp_storage = c->Allocate(temp_storage_bytes, &deleter_context);
    K2_CUDA_SAFE_CALL(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                                             row_splits_diff, max_value,
                                             num_rows, c->GetCudaStream()));
    c->Deallocate(d_temp_storage, deleter_context);
    // this will convert to memory on CPU
    return max_array[0];
  }
}

RaggedShape RaggedShape::Index(int32_t axis, int32_t i) {
  // only support `axis == 0` for now
  K2_CHECK_EQ(axis, 0);
  K2_CHECK_GE(i, 0);
  int32_t num_axes = NumAxes();
  K2_CHECK_GE(num_axes, 2);
  const auto &src_axes = Axes();
  K2_CHECK_LT(i + 1, src_axes[0].row_splits.Dim());

  int32_t idx = src_axes[0].row_splits[i];
  int32_t idx_next = src_axes[0].row_splits[i + 1];
  std::vector<RaggedShapeDim> axes(src_axes.size() - 1);
  ContextPtr c = Context();
  for (int32_t i = 2; i < num_axes; ++i) {
    const Array1<int32_t> &src_row_splits = src_axes[i - 1].row_splits;
    int32_t num_rows = idx_next - idx;
    int32_t offset = idx;
    idx = src_row_splits[idx];
    idx_next = src_row_splits[idx_next];
    // allocate new memory here as we need to change the values,
    // i.e. subtracts the offset.
    axes[i - 2].row_splits = Array1<int32_t>(c, num_rows + 1);
    int32_t *data = axes[i - 2].row_splits.Data();
    const int32_t *src_data = src_row_splits.Data();
    auto lambda_set_values = [=] __host__ __device__(int32_t i) -> void {
      data[i] = src_data[i + offset] - idx;
    };
    Eval(c, num_rows + 1, lambda_set_values);
    // leave row_ids and cached_tot_size unset
    axes[i - 2].cached_tot_size = -1;
  }
  RaggedShape shape(axes, true);
  return shape;
}

void RaggedShape::Populate() {
  int32_t num_axes = NumAxes();
  for (int32_t i = 1; i < num_axes; ++i) {
    // ignore return values of the following calls.
    this->TotSize(i);
    this->RowIds(i);
  }
}

RaggedShape RaggedShape::To(ContextPtr ctx) const {
  if (ctx->IsCompatible(*Context())) return *this;
  std::vector<RaggedShapeDim> axes(axes_.size());
  int32_t num_axes = NumAxes();
  for (int32_t i = 1; i < num_axes; ++i) {
    axes[i - 1].row_splits = axes_[i - 1].row_splits.To(ctx);
    // leave row_ids and cached_tot_size unset
    axes[i - 1].cached_tot_size = -1;
  }
  return RaggedShape(axes);
}

RaggedShapeIndexIterator RaggedShape::Iterator() {
  return RaggedShapeIndexIterator(*this);
}

int32_t RaggedShape::operator[](const std::vector<int32_t> &indexes) {
  K2_CHECK(indexes.size() == NumAxes());
  K2_CHECK(Context()->GetDeviceType() == kCpu);
  int32_t cur_idx = indexes[0];
  for (int32_t i = 1; i < NumAxes(); i++) {
    Array1<int32_t> &row_splits = axes_[i - 1].row_splits;
    K2_CHECK(cur_idx >= 0 && cur_idx + 1 < row_splits.Dim());
    cur_idx = row_splits[cur_idx];
    cur_idx += indexes[i];
  }
  return cur_idx;
}

int32_t RaggedShape::TotSize(int32_t axis) const {
  K2_CHECK_GE(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  if (axis == 0)
    return Dim0();
  else {
    const RaggedShapeDim &rsd = axes_[axis - 1];
    if (rsd.cached_tot_size >= 0) {
      return rsd.cached_tot_size;
    } else {
      // if we had row_ids set up, we should have set cached_tot_size.
      K2_CHECK_EQ(rsd.row_ids.Dim(), 0);
      K2_CHECK_GT(rsd.row_splits.Dim(), 0);
      const_cast<RaggedShapeDim &>(rsd).cached_tot_size = rsd.row_splits.Back();
      return rsd.cached_tot_size;
    }
  }
}

// TODO(dan): change this so that on error it prints a warning if
// print_warnings==true, and then returns false.
bool RaggedShape::Validate(bool print_warnings) {
  ContextPtr c = Context();
  int32_t num_axes = axes_.size();
  for (int32_t axis = 0; axis < num_axes; ++axis) {
    RaggedShapeDim &rsd = axes_[axis];
    K2_CHECK_GE(rsd.row_splits.Dim(), 0);
    if (rsd.cached_tot_size >= 0) {
      K2_CHECK(rsd.row_splits.Dim() == 0 ||
               rsd.cached_tot_size == rsd.row_splits.Back());
      K2_CHECK(rsd.row_ids.Dim() == 0 ||
               rsd.cached_tot_size == rsd.row_ids.Dim());
    } else {
      K2_CHECK_EQ(rsd.cached_tot_size, -1);
      K2_CHECK_EQ(rsd.row_ids.Dim(), 0);
    }

    int32_t num_elems;
    // Check row_splits.
    {
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
          if (next_idx < this_idx) *ok_data = 0;
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
                      << "], row_splits = " << rsd.row_splits;
      }
      if (rsd.cached_tot_size > 0 && rsd.cached_tot_size != num_elems) {
        K2_LOG(FATAL) << "Problem validating row-splits: for axes_[" << axis
                      << "], row_splits[-1] = " << num_elems
                      << " but cached_tot_size == " << rsd.cached_tot_size;
      }
    }
    if (axis + 1 < num_axes) {
      int32_t next_num_rows = axes_[axis + 1].row_splits.Dim() - 1;
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
      Array1<int32_t> meta(c, 2, 1);
      int32_t *ok_data = meta.Data(), *bad_index_data = ok_data + 1;

      const int32_t *row_splits_data = rsd.row_splits.Data(),
                    *row_ids_data = rsd.row_ids.Data();
      int32_t num_elems_from_row_ids = rsd.row_ids.Dim(),
              num_rows = rsd.row_splits.Dim() - 1;

      K2_CHECK_EQ(num_elems, num_elems_from_row_ids);
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
                      << meta[1] << " of row_ids, whose dim is "
                      << rsd.row_ids.Dim();
      }
    }
    if (axis + 1 < axes_.size()) {
      K2_CHECK(IsCompatible(rsd.row_splits, axes_[axis + 1].row_splits));
    }
  }
  return true;
}

RaggedShape RaggedShape2(Array1<int32_t> *row_splits, Array1<int32_t> *row_ids,
                         int32_t cached_tot_size) {
  K2_CHECK(row_splits != nullptr || row_ids != nullptr)
      << "At least one of row_splits and row_ids must be defined";
  ContextPtr ctx = ::GetContext(row_splits, row_ids);
  if (cached_tot_size != -1) {
    if (row_ids != nullptr) K2_CHECK_EQ(cached_tot_size, row_ids->Dim());
    if (row_splits != nullptr) {
      // may be slow as it may copy memory from device to host
      K2_DCHECK_EQ(cached_tot_size, row_splits->Back());
    }
  }
  std::vector<RaggedShapeDim> axes(1);
  if (row_splits != nullptr) {
    axes[0].row_splits = *row_splits;
  } else {
    // we need to work out row_splits as we always require row_splits is not
    // empty for RaggedShape. Note here we suppose the last element in row_ids
    // is num_rows - 1, i.e. there's no empty rows after row `row_ids[-1]`.
    int32_t num_rows = row_ids->Dim() == 0 ? 0 : row_ids->Back() + 1;
    Array1<int32_t> row_splits_array(ctx, num_rows + 1);
    RowIdsToRowSplits(*row_ids, row_splits_array);
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
  if (a.NumElements() != b.Dim0()) {
    K2_LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: " << a.NumElements()
                  << " vs. " << b.Dim0();
  }
  const auto &a_axes = a.Axes();
  const auto &b_axes = b.Axes();
  std::vector<RaggedShapeDim> axes(a_axes.size() + b_axes.size());
  std::size_t a_size = a_axes.size(), b_size = b_axes.size();
  for (std::size_t i = 0; i < a_size; ++i) axes[i] = a_axes[i];
  for (std::size_t i = 0; i < b_size; ++i) axes[i + a_size] = b_axes[i];
  return RaggedShape(axes);
}

RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2) {
  K2_CHECK(row_splits1 != nullptr || row_ids1 != nullptr)
      << "At least one of row_splits1 and row_ids1 must be defined";
  K2_CHECK(row_splits2 != nullptr || row_ids2 != nullptr)
      << "At least one of row_splits2 and row_ids2 must be defined";

  // check context
  ContextPtr ctx1 = ::GetContext(row_splits1, row_ids1);
  ContextPtr ctx2 = ::GetContext(row_splits2, row_ids2);
  K2_CHECK(ctx1->IsCompatible(*ctx2));

  // check row_splits and row_ids of axis-1
  if (cached_tot_size1 != -1) {
    if (row_ids1 != nullptr) K2_CHECK_EQ(cached_tot_size1, row_ids1->Dim());
    if (row_splits1 != nullptr) {
      // may be slow as it may copy memory from device to host
      K2_DCHECK_EQ(cached_tot_size1, row_splits1->Back());
    }
  }

  // check row_splits and row_ids of axis-2
  if (cached_tot_size2 != -1) {
    if (row_ids2 != nullptr) K2_CHECK_EQ(cached_tot_size2, row_ids2->Dim());
    if (row_splits2 != nullptr) {
      // may be slow as it may copy memory from device to host
      K2_DCHECK_EQ(cached_tot_size2, row_splits2->Back());
    }
  }

  std::vector<RaggedShapeDim> axes(2);
  // set row_splits and row_ids for axis 1
  if (row_splits1 != nullptr) {
    axes[0].row_splits = *row_splits1;
  } else {
    // work out row_splits1, see code in RaggedShape2 above for the reason
    int32_t num_rows = row_ids1->Dim() == 0 ? 0 : row_ids1->Back() + 1;
    Array1<int32_t> row_splits_array(ctx1, num_rows + 1);
    RowIdsToRowSplits(*row_ids1, row_splits_array);
    axes[0].row_splits = row_splits_array;
  }
  if (row_ids1 != nullptr) axes[0].row_ids = *row_ids1;
  if (cached_tot_size1 == -1) {
    cached_tot_size1 =
        row_ids1 != nullptr ? row_ids1->Dim() : axes[0].row_splits.Back();
  }
  axes[0].cached_tot_size = cached_tot_size1;

  // set row_splits and row_ids for axis 2
  if (row_splits2 != nullptr) {
    axes[1].row_splits = *row_splits2;
  } else {
    // work out row_splits1, see code in RaggedShape2 above for the reason
    int32_t num_rows = row_ids2->Dim() == 0 ? 0 : row_ids2->Back() + 1;
    Array1<int32_t> row_splits_array(ctx1, num_rows + 1);
    RowIdsToRowSplits(*row_ids2, row_splits_array);
    axes[1].row_splits = row_splits_array;
  }
  if (row_ids2 != nullptr) axes[1].row_ids = *row_ids2;
  if (cached_tot_size2 == -1) {
    cached_tot_size2 =
        row_ids2 != nullptr ? row_ids2->Dim() : axes[1].row_splits.Back();
  }
  axes[1].cached_tot_size = cached_tot_size2;

  // we don't check here if
  // row_splits1[row_splits1.Dim() - 1] == row_ids1.Dim()
  //   == (row_splits2.Dim() - 1)
  //   >= (row_ids2[row_ids2.Dim() - 1] + 1)
  // but RaggedShape(axes) below will check this.
  return RaggedShape(axes);
}

RaggedShape RaggedShapeFromTotSizes(ContextPtr &c, int32_t num_axes,
                                    int32_t *tot_sizes) {
  K2_CHECK_GE(num_axes, 2);
  std::vector<RaggedShapeDim> axes(num_axes - 1);
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

  ContextPtr c = src.Context();
  K2_CHECK(axis >= 0 && axis <= src.NumAxes());

  const std::vector<RaggedShapeDim> &axes_in = src.Axes();
  int32_t num_axes_in = src.NumAxes();

  // Note: in RaggedShape, the vector of RaggedShapeDim is of length
  // num_axes - 1, so the output will have one more axis than the input.
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
  } else {
    int32_t tot_size = src.TotSize(axis);
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
  axes_out[axis].cached_tot_size = row_ids_dim;
  for (int32_t i = 0; i < axis; ++i) axes_out[i] = axes_in[i];
  // Note: the returned array has `num_axes_in + 1` axes, so its
  // array of RaggedShapeDim is of length `num_axes_in`.
  for (int32_t i = axis + 1; i < num_axes_in; ++i) axes_out[i] = axes_in[i - 1];
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
  RaggedShape ans = RaggedShapeFromTotSizes(c, num_axes, tot_sizes_out.data());

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

/*
  Extract meta-info from the shape (this will include populating any row_ids and
  row_splits that were not already populated).  This is used inside algorithms
  when we need to transfer meta-info to GPU.

     @param [in]   src   Ragged shape that we're extracting meta-info from
     @param [out] row_splits  This will be set to an array of size
                              src.NumAxes()-1, containing pointers to the
                              row_splits' Data() vectors. The array will be
                              allocated on the same device as `src`.
     @param [out] row_ids     This will be set to an array of size
                              src.NumAxes()-1, containing pointers to the
                              row_ids' Data() vectors. The array will be
                              allocated on the same device as `src`.
*/
void GetRowInfo(RaggedShape &src, Array1<int32_t *> *row_splits,
                Array1<int32_t *> *row_ids) {
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

/*
  Get some meta-info for an array of RaggedShape, and transfer them
  to the device that `src` is located on. Just same with `GetRowInfo`
  above, but for multiple RaggedShapes.

     @param [in] num_srcs  Number of source arrays to process.
     @param [in] src      Source arrays.  All of them must have same num_axes
                          and on the same device, but we just check this in
                          debug mode.
     @param [in] row_splits  Output array of row_splits pointers,
                          will be of dimension num_axes-1 by num_src
     @param [in] row_splits  Output array of row_splits pointers,
                          will be of dimension num_axes-1 by num_src
*/
void GetRowInfoMulti(int32_t num_srcs, RaggedShape **src,
                     Array2<int32_t *> *row_splits,
                     Array2<int32_t *> *row_ids) {
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

RaggedShape Append(int32_t axis, int32_t num_srcs, RaggedShape **src) {
  K2_CHECK_EQ(axis, 0) << "Append() with axis > 0 not yet supported";
  K2_CHECK_GT(num_srcs, 1);
  int32_t num_axes = src[0]->NumAxes();
  ContextPtr c = src[0]->Context();

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
  return ans;
}

RaggedShape RemoveAxis(RaggedShape &src, int32_t axis) {
  K2_CHECK_GT(src.NumAxes(), 2);
  K2_CHECK(axis >= 0 && axis < src.NumAxes());

  // note, `axes_in` is of dim src.NumAxes() - 1.
  // Also note: axes_in[i] pertains to the relationship between
  // axes i and i+1 in the source.
  src.Populate();

  const std::vector<RaggedShapeDim> &axes_in = src.Axes();

  std::vector<RaggedShapeDim> axes_out(axes_in.size() - 1);
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

// transpose axes 0 and 1.
RaggedShape Transpose(RaggedShape &src) {
  K2_CHECK_GT(src.NumAxes(), 2);
  int32_t src_dim0 = src.Dim0(), src_tot_size1 = src.TotSize(1);
  K2_CHECK_EQ(src_tot_size1 % src_dim0, 0)
      << "Transpose(): all dims on axis 0 must be the same.";
  int32_t src_dim1 = src_tot_size1 / src_dim0;
  RaggedShape src_no_axis0 = RemoveAxis(src, 0);
  K2_CHECK_EQ(src_no_axis0.Dim0(), src_tot_size1);
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
  int32_t *mem_data = mem.Data();
  auto lambda_set_row_info = [=] __host__ __device__(int32_t i) {
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
  };
  Eval(c, row_splits_dim + row_ids_dim, lambda_set_row_info);
  ans_axis0[0].row_splits = mem.Range(0, row_splits_dim);
  ans_axis0[0].row_ids = mem.Range(row_splits_dim, row_ids_dim);
  ans_axis0[0].cached_tot_size = row_ids_dim;

  RaggedShape temp(ans_axis0);
  return ComposeRaggedShapes(temp, src_no_axis0_renumbered);
}

RaggedShape Stack(int32_t axis, int32_t num_srcs, const RaggedShape **src) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK(axis >= 0 && axis <= 1);

  ContextPtr c = src[0]->Context();
  int32_t num_axes = src[0]->NumAxes();

  // Check if they have the same num-axes and compatible context
  for (int32_t i = 1; i < num_srcs; ++i) {
    K2_CHECK_EQ(num_axes, src[i]->NumAxes());
    K2_CHECK(c->IsCompatible(*src[i]->Context()));
  }

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
    // wait.)
  }

  RaggedShape ans = Append(0, num_srcs, unsqueezed_ptrs.data());
  // Transpose will check if all src->Dim0() has the same value.
  if (axis == 1) ans = Transpose(ans);
  return ans;
}

RaggedShape TrivialShape(ContextPtr &c, int32_t num_elems) {
  // row_splits= [
  Array1<int32_t> row_splits = Range<int32_t>(c, 2, 0, num_elems);
  int32_t *row_splits_data = row_splits.Data();

  Array1<int32_t> row_ids(c, num_elems, 0);
  return RaggedShape2(&row_splits, &row_ids, num_elems);
}

}  // namespace k2
