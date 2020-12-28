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
#include "k2/csrc/macros.h"
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

void PrintRaggedShapePart(std::ostream &stream, const RaggedShape &shape,
                          int32_t axis, int32_t begin_pos, int32_t end_pos) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() && begin_pos >= 0 &&
           begin_pos <= end_pos && end_pos <= shape.TotSize(axis));
  for (int32_t d = begin_pos; d < end_pos; ++d) {
    if (axis == shape.NumAxes() - 1) {
      stream << "x ";
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
std::ostream &operator<<(std::ostream &stream, const RaggedShape &shape) {
  if (shape.Context()->GetDeviceType() != kCpu) {
    return stream << shape.To(GetCpuContext());
  } else {
    bool print_warnings = false;
    if (shape.Validate(print_warnings)) {
      stream << "[ ";
      PrintRaggedShapePart(stream, shape, 0, 0, shape.Dim0());
      stream << "]";
      return stream;
    } else {
      // For non-valid shapes, print the raw info.
      stream << "Invalid RaggedShape: { ";
      stream << " num-axes = " << shape.NumAxes();
      for (int32_t i = 1; i < shape.NumAxes(); i++) {
        const RaggedShapeLayer &layer = shape.Layers()[i - 1];
        if (layer.row_splits.IsValid())
          stream << " RowSplits(" << i << ")=" << layer.row_splits;
        if (layer.row_ids.IsValid())
          stream << "RowIds(" << i << ")=" << layer.row_ids;
        stream << "cached_tot_size[" << i << "]=" << layer.cached_tot_size;
      }
      return stream << " }";
    }
  }
}

Array1<int32_t> &RaggedShape::RowIds(int32_t axis) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  RaggedShapeLayer &rsd = layers_[axis - 1];
  auto &row_splits = rsd.row_splits;
  auto &row_ids = rsd.row_ids;
  // there must be row_splits.Dim() >=1 according to the definition of
  // RaggedShapeLayer.
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
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  const auto &row_splits = layers_[axis - 1].row_splits;
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

    size_t temp_storage_bytes = 0;
    // the first time is to determine temporary device storage requirements
    K2_CUDA_SAFE_CALL(cub::DeviceReduce::Max(nullptr, temp_storage_bytes,
                                             row_splits_diff, max_value,
                                             num_rows, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceReduce::Max(
        d_temp_storage.Data(), temp_storage_bytes, row_splits_diff, max_value,
        num_rows, c->GetCudaStream()));
    // this will convert to memory on CPU
    return max_array[0];
  }
}

RaggedShape RaggedShape::Index(int32_t axis, int32_t i,
                               int32_t *value_offset /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  // only support `axis == 0` for now
  K2_CHECK_EQ(axis, 0);
  K2_CHECK_GE(i, 0);
  int32_t num_axes = NumAxes();
  K2_CHECK_GT(num_axes, 2);
  const auto &src_axes = Layers();
  K2_CHECK_LT(i + 1, src_axes[0].row_splits.Dim());

  if (i == 0 && Dim0() == 1) {
    // Just remove first axis.  Common case so we make it efficient.
    std::vector<RaggedShapeLayer> ans_axes(src_axes.begin() + 1,
                                           src_axes.end());
    if (value_offset) *value_offset = 0;
    return RaggedShape(ans_axes, false);
  }

  int32_t idx_begin = (i != 0 ? src_axes[0].row_splits[i] : 0),
          idx_end = src_axes[0].row_splits[i + 1];
  std::vector<RaggedShapeLayer> axes(src_axes.size() - 1);
  ContextPtr &c = Context();
  for (int32_t i = 2; i < num_axes; ++i) {
    const Array1<int32_t> &src_row_splits = RowSplits(i),
                          &src_row_ids = RowIds(i);
    // TODO(fangjun): see https://github.com/k2-fsa/k2/pull/547
    // for how to optimize it (do all transfer in a single kernel).
    int32_t idx_begin_next = (idx_begin != 0 ? src_row_splits[idx_begin] : 0),
            idx_end_next = src_row_splits[idx_end];

    axes[i - 2].row_splits =
        src_row_splits.Range(idx_begin, idx_end - idx_begin + 1);
    if (idx_begin_next != 0)
      axes[i - 2].row_splits = Minus(axes[i - 2].row_splits, idx_begin_next);

    axes[i - 2].row_ids =
        src_row_ids.Range(idx_begin_next, idx_end_next - idx_begin_next);
    if (idx_begin != 0)
      axes[i - 2].row_ids = Minus(axes[i - 2].row_ids, idx_begin);
    axes[i - 2].cached_tot_size = idx_end_next - idx_begin_next;
    idx_begin = idx_begin_next;
    idx_end = idx_end_next;
  }
  if (value_offset) *value_offset = idx_begin;
  return RaggedShape(axes);
}

void RaggedShape::Populate() {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = NumAxes();
  ParallelRunner pr(this->Context());
  for (int32_t i = 1; i < num_axes; ++i) {
    With w(pr.NewStream());
    // ignore return values of the following calls.
    this->TotSize(i);
    this->RowIds(i);
  }
}

RaggedShape RaggedShape::To(ContextPtr ctx) const {
  NVTX_RANGE(K2_FUNC);
  if (ctx->IsCompatible(*Context())) return *this;
  std::vector<RaggedShapeLayer> axes(layers_.size());
  int32_t num_axes = NumAxes();
  for (int32_t i = 1; i < num_axes; ++i) {
    axes[i - 1].row_splits = layers_[i - 1].row_splits.To(ctx);
    // leave row_ids and cached_tot_size unset
    axes[i - 1].cached_tot_size = -1;
  }
  return RaggedShape(axes);
}

RaggedShapeIndexIterator RaggedShape::Iterator() {
  return RaggedShapeIndexIterator(*this);
}

int32_t RaggedShape::operator[](const std::vector<int32_t> &indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(static_cast<int32_t>(indexes.size()), NumAxes());
  K2_CHECK_EQ(Context()->GetDeviceType(), kCpu);
  int32_t cur_idx = indexes[0];
  for (int32_t i = 1; i < NumAxes(); i++) {
    Array1<int32_t> &row_splits = layers_[i - 1].row_splits;
    K2_CHECK(cur_idx >= 0 && cur_idx + 1 < row_splits.Dim());
    cur_idx = row_splits[cur_idx];
    cur_idx += indexes[i];
  }
  return cur_idx;
}

int32_t RaggedShape::TotSize(int32_t axis) const {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(axis, 0);
  K2_CHECK_LT(axis, NumAxes());
  if (axis == 0)
    return Dim0();
  else {
    const RaggedShapeLayer &rsd = layers_[axis - 1];
    if (rsd.cached_tot_size >= 0) {
      return rsd.cached_tot_size;
    } else {
      // if we had row_ids set up, we should have set cached_tot_size.
      K2_CHECK_EQ(rsd.row_ids.Dim(), 0);
      K2_CHECK_GT(rsd.row_splits.Dim(), 0);
      const_cast<RaggedShapeLayer &>(rsd).cached_tot_size =
          rsd.row_splits.Back();
      return rsd.cached_tot_size;
    }
  }
}

// TODO(dan): change this so that on error it prints a warning if
// print_warnings==true, and then returns false.
bool RaggedShape::Validate(bool print_warnings) const {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = Context();
  int32_t num_axes = layers_.size();

  ParallelRunner pr(c);
  for (int32_t axis = 0; axis < num_axes; ++axis) {
    With w(pr.NewStream());
    const RaggedShapeLayer &rsd = layers_[axis];
    K2_CHECK_GE(rsd.row_splits.Dim(), 0);
    if (rsd.cached_tot_size >= 0) {
      if (!(rsd.row_splits.Dim() == 0 ||
            rsd.cached_tot_size == rsd.row_splits.Back())) {
        if (print_warnings)
          K2_LOG(WARNING)
              << "Ragged shape validation failed, row_splits.Back()="
              << rsd.row_splits.Back()
              << " vs. cached-tot-size=" << rsd.cached_tot_size;
        return false;
      }
      if (!((rsd.row_ids.Dim() == 0 ||
             rsd.cached_tot_size == rsd.row_ids.Dim()))) {
        if (print_warnings)
          K2_LOG(WARNING) << "Ragged shape validation failed, row_ids.Dim()="
                          << rsd.row_ids.Dim()
                          << " vs. cached-tot-size=" << rsd.cached_tot_size;
        return false;
      }
    } else {
      if (rsd.cached_tot_size != -1 || rsd.row_ids.Dim() != 0) {
        if (print_warnings)
          K2_LOG(WARNING) << "Ragged shape validation failed, cached_tot_size="
                          << rsd.cached_tot_size
                          << ", row-ids.Dim()=" << rsd.row_ids.Dim();
        return false;
      }
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

      K2_EVAL(
          c, num_rows + 1, lambda_check_row_splits, (int32_t i)->void {
            int32_t this_idx = row_splits_data[i];
            if (i == 0 && this_idx != 0) *ok_data = 0;
            if (i < num_rows) {
              int32_t next_idx = row_splits_data[i + 1];
              if (next_idx < this_idx) *ok_data = 0;
            } else {
              K2_CHECK(i == num_rows);
              *num_elems_data = this_idx;
            }
          });
      meta = meta.To(GetCpuContext());
      num_elems = meta[1];
      int32_t ok = meta[0];
      if (!ok) {
        K2_LOG(FATAL) << "Problem validating row-splits: for layers_[" << axis
                      << "], row_splits = " << rsd.row_splits;
      }
      if (rsd.cached_tot_size > 0 && rsd.cached_tot_size != num_elems) {
        K2_LOG(FATAL) << "Problem validating row-splits: for layers_[" << axis
                      << "], row_splits[-1] = " << num_elems
                      << " but cached_tot_size == " << rsd.cached_tot_size;
      }
    }
    if (axis + 1 < num_axes) {
      int32_t next_num_rows = layers_[axis + 1].row_splits.Dim() - 1;
      if (num_elems != next_num_rows) {
        K2_LOG(FATAL) << "Ragged shape has num_elems for layers_[" << axis
                      << "] == " << num_elems << " and num-rows for layers_["
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
      // TODO: could do this and the other one in separate streams.
      K2_EVAL(
          c, num_elems, lambda_check_row_ids, (int32_t i)->void {
            int32_t this_row = row_ids_data[i];
            if (this_row < 0 || this_row >= num_rows ||
                i < row_splits_data[this_row] ||
                i >= row_splits_data[this_row + 1]) {
              *ok_data = 0;
              *bad_index_data = i;
            }
          });
      meta = meta.To(GetCpuContext());  // since we have 2 accesses, this should
                                        // be faster.
      int32_t ok = meta[0];
      if (!ok) {
        K2_LOG(FATAL) << "Problem validating row-ids: for layers_[" << axis
                      << "], row_splits = " << rsd.row_splits
                      << ", row_ids = " << rsd.row_ids << ", see index "
                      << meta[1] << " of row_ids, whose dim is "
                      << rsd.row_ids.Dim();
      }
    }
    if (axis + 1 < (int32_t)layers_.size()) {
      K2_CHECK(IsCompatible(rsd.row_splits, layers_[axis + 1].row_splits));
    }
  }
  return true;
}

bool Equal(const RaggedShape &a, const RaggedShape &b) {
  NVTX_RANGE(K2_FUNC);
  if (a.NumAxes() != b.NumAxes()) return false;
  for (int32_t i = 1; i < a.NumAxes(); i++) {
    if (a.RowSplits(i).Dim() != b.RowSplits(i).Dim() ||
        !Equal(a.RowSplits(i), b.RowSplits(i)))
      return false;
  }
  return true;
}

std::istream &operator>>(std::istream &is, RaggedShape &shape) {
  NVTX_RANGE(K2_FUNC);
  // Note: element 0 of 'row_splits' will end up being
  // discarded; the others will become the axes of `shape`.
  std::vector<std::vector<int32_t>> row_splits;
  int32_t cur_level = 0, num_elems = 0;
  while (1) {
    is >> std::ws;  // eat whitespace
    if (!is.good()) {
      is.setstate(std::ios::failbit);
      return is;
    }
    int c = is.get();
    if (c == static_cast<int32_t>('[')) {
      cur_level++;
      while (row_splits.size() < static_cast<size_t>(cur_level)) {
        if (num_elems != 0) {
          is.setstate(std::ios::failbit);
          return is;
        }
        row_splits.push_back(std::vector<int32_t>(1, 0));
      }
    } else if (c == static_cast<int32_t>(']')) {
      cur_level--;
      if (cur_level <= 0) {   // Done; return...
        if (cur_level < 0) {  // ']' without '['.
          is.setstate(std::ios::failbit);
          return is;
        }
        row_splits.erase(row_splits.begin());
        if (row_splits.empty()) {
          // Assume 2 axes even though the num-axes is ambiguous from the input.
          // row_splits is 0 0.
          row_splits.push_back(std::vector<int32_t>(1, 0));
        }
        std::vector<RaggedShapeLayer> axes(row_splits.size());
        for (size_t i = 0; i < row_splits.size(); i++) {
          axes[i].row_splits = Array1<int32_t>(GetCpuContext(), row_splits[i]);
          axes[i].cached_tot_size = -1;
        }
        shape = RaggedShape(axes);
        return is;
      }
      row_splits[cur_level].push_back(
          (cur_level + 1 >= (int32_t)row_splits.size())
              ? num_elems
              : (row_splits[cur_level + 1].size() - 1));
    } else if (c == static_cast<int32_t>('x')) {
      if (cur_level != static_cast<int32_t>(row_splits.size()) ||
          cur_level < 2) {
        is.setstate(std::ios::failbit);
        return is;
      }
      num_elems++;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
}

}  // namespace k2
