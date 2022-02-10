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

#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/cub.h"
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
  __device__ __forceinline__ int32_t operator[](int32_t i) const {
    return row_splits_data[i + 1] - row_splits_data[i];
  }
  __device__ __forceinline__ RowSplitsDiff operator+(int32_t n) const {
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


std::ostream &OutputBadRaggedShape(std::ostream &stream,
                                   const RaggedShape &shape) {
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

// prints a RaggedShape as e.g. [ [ 0 1 ] [ 2 ] [] ].  Note, the 'values'
// are just the positions in the array, this is for readability.
std::ostream &operator<<(std::ostream &stream, const RaggedShape &shape) {
  if (shape.Context()->GetDeviceType() != kCpu) {
    return stream << shape.To(GetCpuContext());
  } else {
    try {
      shape.Check();
      stream << "[ ";
      PrintRaggedShapePart(stream, shape, 0, 0, shape.Dim0());
      stream << "]";
      return stream;
    } catch (...) {
      return OutputBadRaggedShape(stream, shape);
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
                               int32_t *value_offset /*= nullptr*/) const {
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

RaggedShape RaggedShape::To(ContextPtr ctx,
                            bool copy_all) const {
  NVTX_RANGE(K2_FUNC);
  if (ctx->IsCompatible(*Context())) return *this;
  std::vector<RaggedShapeLayer> layers(layers_.size());
  int32_t num_layers = layers.size();
  for (int32_t i = 0; i < num_layers; i++) {
    layers[i].row_splits = layers_[i].row_splits.To(ctx);
    layers[i].cached_tot_size = layers_[i].cached_tot_size;
    if (copy_all && layers_[i].row_ids.IsValid())
      layers[i].row_ids = layers_[i].row_ids.To(ctx);
  }
  return RaggedShape(layers);
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

// This is for accessing info about a ragged shape on GPU.
struct RaggedShapeLayerInfo {
  int32_t num_rows;
  const int32_t *row_splits;  // [num_rows]
  int32_t num_elems;  // or -1 if unknown
  const int32_t *row_ids;  // [num_elems], or NULL if not populated
};

void RaggedShape::Check() const {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = Context();
  int32_t num_layers = layers_.size();

  /*
    If debugging ragged-shape validation failures, uncommenting the following may help
    (although it would be slow).
  if (c->GetDeviceType() != kCpu) {
    this->To(GetCpuContext(), true).Check();
    return;
  }
  */
  try {
    if (c->GetDeviceType() == kCpu) {
      // This branch should be more efficient on CPU, although
      // it would also work fairly fast for GPU.
      for (int32_t layer = 0; layer < num_layers; ++layer) {
        const RaggedShapeLayer &rsd = layers_[layer];
        K2_CHECK_GT(rsd.row_splits.Dim(), 0);
        if (rsd.cached_tot_size >= 0) {
          if (!(rsd.row_splits.Dim() == 0 ||
                rsd.cached_tot_size == rsd.row_splits.Back())) {
            K2_LOG(FATAL)
                << "Ragged shape validation failed, row_splits.Back()="
                << rsd.row_splits.Back()
                << " vs. cached-tot-size=" << rsd.cached_tot_size;
          }
          if (!((rsd.row_ids.Dim() == 0 ||
                 rsd.cached_tot_size == rsd.row_ids.Dim()))) {
            K2_LOG(FATAL) << "Ragged shape validation failed, row_ids.Dim()="
                          << rsd.row_ids.Dim()
                          << " vs. cached-tot-size=" << rsd.cached_tot_size;
          }
        } else {
          if (rsd.cached_tot_size != -1 || rsd.row_ids.Dim() != 0) {
            K2_LOG(FATAL) << "Ragged shape validation failed, cached_tot_size="
                          << rsd.cached_tot_size
                          << ", row-ids.Dim()=" << rsd.row_ids.Dim();
          }
        }

        // Check row_splits.
        {
          const int32_t *row_splits_data = rsd.row_splits.Data();
          int32_t num_rows = rsd.row_splits.Dim() - 1,
              cached_tot_size = rsd.cached_tot_size;
          K2_EVAL(
              c, num_rows + 1, lambda_check_row_splits, (int32_t i)->void {
                int32_t this_idx = row_splits_data[i];
                if (i == 0 && this_idx != 0) {
                  K2_LOG(FATAL) << "Error validating row-splits, i=0, this_idx="
                                << this_idx << ", num_rows=" << num_rows
                                << ", layer=" << layer;
                }
                if (i < num_rows) {
                  int32_t next_idx = row_splits_data[i + 1];
                  if (next_idx < this_idx) {
                    K2_LOG(FATAL) << "Error validating row-splits, i="
                                  << num_rows
                                  << "row_splits[i]=" << this_idx
                                  << "row_splits[i+1]=" << next_idx;
                  }
                } else {
                  K2_CHECK(i == num_rows);
                  if (cached_tot_size != -1 && this_idx != cached_tot_size) {
                    K2_LOG(FATAL) << "Error validating row-splits, i="
                                  << num_rows
                                  << "==num_rows, row_splits[i]=" << this_idx
                                  << " but expected it to equal cached-tot-size"
                                  "=" << cached_tot_size << "; layer=" << layer;
                  }
                }
              });
        }
        if (rsd.row_ids.Dim() != 0) {  // check row_ids.
          K2_CHECK(IsCompatible(rsd.row_ids, rsd.row_splits));
          // 1st elem is `ok` (1 or 0); 2nd elem is location of bad index
          // into row_splits

          const int32_t *row_splits_data = rsd.row_splits.Data(),
              *row_ids_data = rsd.row_ids.Data();
          int32_t num_elems = rsd.row_ids.Dim(),
              num_rows = rsd.row_splits.Dim() - 1;

          // TODO: could do this and the other one in separate streams.
          K2_EVAL(
              c, num_elems, lambda_check_row_ids, (int32_t i)->void {
                int32_t this_row = row_ids_data[i];
                if (this_row < 0 || this_row >= num_rows ||
                    i < row_splits_data[this_row] ||
                    i >= row_splits_data[this_row + 1]) {
                  K2_LOG(FATAL) << "Failed checking row_ids: row_ids[" << i
                                << "]=" << this_row
                                << ", row_splits_data[n,n+1]="
                                << row_splits_data[this_row] << ","
                                << row_splits_data[this_row+1] << ", layer="
                                << layer;
                }
              });
        }


        if (layer + 1 < num_layers) {
          // Check the num-elems on this layer == num_rows on next layer.
          int32_t next_num_rows = layers_[layer + 1].row_splits.Dim() - 1;
          if (rsd.cached_tot_size != -1) {
            int32_t num_elems = rsd.cached_tot_size;
            if (num_elems != next_num_rows) {
              K2_LOG(FATAL) << "Ragged shape has num_elems for layer " << layer
                            << " == " << num_elems << " vs. num-rows for layer "
                            << (layer + 1) << " == " << next_num_rows;
            }
          } else {
            const int32_t *num_elems_ptr = rsd.row_splits.Data() +
                rsd.row_splits.Dim() - 1;
            K2_EVAL(c, 1, lambda_check_num_elems, (int32_t i) -> void {
                if (*num_elems_ptr != next_num_rows) {
                  K2_LOG(FATAL) << "Ragged shape has num_elems="
                                << *num_elems_ptr << " for layers_[" << layer
                                << "], vs. num_rows=" << next_num_rows
                                << "for the next layer.";
                }});
          }
        }
        if (layer + 1 < (int32_t)layers_.size()) {
          K2_CHECK(IsCompatible(rsd.row_splits, layers_[layer + 1].row_splits));
        }
      }
    } else {
      // This branch is more optimized for GPU, by using a single kernel.
      // Note: the error message may appear later if we are not syncing kernel
      // so the stack trace won't be informative. You would need to set the
      // environment variable K2_SYNC=1 before running, to get the correct
      // stack trace.
      constexpr int MAX_DIM = 6;
      K2_CHECK_LE(num_layers, MAX_DIM);

      // put all info into this form so we can access it from GPU.
      SmallVec<RaggedShapeLayerInfo, MAX_DIM> layers_info;

      int32_t max_size = 0;
      for (int32_t layer = 0; layer < num_layers; ++layer) {
        const RaggedShapeLayer &rsd = layers_[layer];
        RaggedShapeLayerInfo &info = layers_info.data[layer];
        K2_CHECK_GT(rsd.row_splits.Dim(), 0);
        if (!rsd.row_splits.Context()->IsCompatible(*c)) {
          K2_LOG(FATAL) << "Incompatible contexts for different components "
              "of RaggedShape: row_splits of layer " << layer
                        << " vs. layer 0.";
        }
        info.row_splits = rsd.row_splits.Data();
        info.num_rows = rsd.row_splits.Dim() - 1;
        if (layer > 0) {
          int32_t prev_num_elems = layers_info.data[layer-1].num_elems;
          if (prev_num_elems >= 0 &&
              info.num_rows != prev_num_elems)
            K2_LOG(FATAL) << "Num-rows on layer " << layer << " is "
                          << info.num_rows << ", vs. num-elems on layer "
                          << (layer-1) << " = " << prev_num_elems;
        }
        if (info.num_rows + 1 > max_size)
          max_size = info.num_rows + 1;
        if (rsd.row_ids.IsValid()) {
          info.row_ids = rsd.row_ids.Data();
          info.num_elems = rsd.row_ids.Dim();
          if (!rsd.row_ids.Context()->IsCompatible(*c)) {
            K2_LOG(FATAL) << "Incompatible contexts for different components "
                "of RaggedShape: row_ids of layer " << layer
                          << " vs. row_splits of layer 0.";
            if (rsd.row_ids.Dim() != rsd.cached_tot_size) {
              K2_LOG(FATAL) << "Validating RaggedShape: error, on layer "
                            << layer << ", cached_tot_size="
                            << rsd.cached_tot_size << " vs. row_ids.Dim()="
                            << rsd.row_ids.Dim();
            }
          }
        } else {
          K2_CHECK_GE(rsd.cached_tot_size, -1);
          info.num_elems = rsd.cached_tot_size;
          info.row_ids = nullptr;
        }
        if (info.num_elems > max_size)
          max_size = info.num_elems;
      }

      int32_t block = 32;
      // round up max_size to a multiple of `block`; will help ensure
      // different branches of statements go in different warps.
      max_size = block * ((max_size + block - 1) / block);
      K2_EVAL2(c, 2 * num_layers, max_size, lambda_check_ragged_shape,
             (int32_t i, int32_t j) -> void {
                 int32_t layer = i / 2,
                     job_type = i % 2;
               const RaggedShapeLayerInfo &this_info = layers_info.data[layer];
               if (job_type == 0) {  // Checking row_splits
                 if (j > this_info.num_rows)
                   return;
                 int32_t this_elem = this_info.row_splits[j];
                 K2_CHECK_GE(this_elem, 0) << " layers[" << layer
                                           << "].row_splits should be >= 0.";
                   if (j == 0) {
                     K2_CHECK_EQ(this_elem, 0) << " layers[" << layer
                                               << "].row_splits[0] != 0";
                   }
                   if (j == this_info.num_rows) {
                     if (this_info.num_elems >= 0) {
                       K2_CHECK_EQ(this_elem, this_info.num_elems)
                           << " layers[" << layer << "]: last elem of "
                           << "row_splits does not have the expected value.";
                     } else if (layer + 1 < num_layers) {
                       int32_t next_layer_num_rows =
                           layers_info.data[layer+1].num_rows;
                       K2_CHECK_EQ(this_elem, next_layer_num_rows)
                           << " layers[" << layer << "]: last elem of "
                           << "row_splits does not have the expected value "
                           "vs. next layer's num-rows";
                     }
                   } else {
                     int32_t next_elem = this_info.row_splits[j+1];
                     K2_CHECK_GE(next_elem, this_elem)
                         << " layers[" << layer << "].row_splits is not "
                         "monotonic.";
                   }
                 } else {  // Checking row_ids
                   if (this_info.row_ids == nullptr)
                     return;  // row_ids is not set up.
                   if (j >= this_info.num_elems)
                     return;
                   int32_t this_row = this_info.row_ids[j];
                   K2_CHECK_GE(this_row, 0) << " layers[" << layer
                                            << "].row_ids[" << j << "] < 0.";
                   K2_CHECK_LT(this_row, this_info.num_rows)
                       << " layers[" << layer << "].row_ids[" << j
                       << "] >= num_rows.";
                   K2_CHECK_GE(j, this_info.row_splits[this_row])
                       << " j < layers[" << layer << "].row_splits["
                       << this_row << "];";
                   K2_CHECK_LT(j, this_info.row_splits[this_row+1])
                       << " j >= layers[" << layer << "].row_splits["
                       << this_row << "+1];";
                 }
               });
    }
  } catch (...) {
    std::ostringstream stream;
    OutputBadRaggedShape(stream, *this);
    K2_LOG(FATAL) << "Check of RaggedShape failed, see errors printed above:"
                  << stream.str();
  }
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
