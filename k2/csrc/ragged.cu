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

}  // namespace

namespace std {
// vaule_type is required by cub::DeviceReduce::Max
template <>
struct iterator_traits<::RowSplitsDiff> {
  typedef int32_t value_type;
};
}  // namespace std

namespace k2 {


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

}  // namespace k2
