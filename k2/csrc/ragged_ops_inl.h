/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu
 *                                                   Wei Kang)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_RAGGED_OPS_INL_H_
#define K2_CSRC_RAGGED_OPS_INL_H_

#ifndef IS_IN_K2_CSRC_RAGGED_OPS_H_
#error "this file is supposed to be included only by ragged_ops.h"
#endif

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "k2/csrc/array_ops.h"
#ifdef K2_WITH_CUDA
#include "k2/csrc/cudpp/cudpp.h"
#endif
#include "k2/csrc/macros.h"
#include "k2/csrc/moderngpu_allocator.h"

namespace k2 {

template <typename T, typename Op>
void SegmentedReduce(const Ragged<T> &src, T initial_value, Array1<T> *dst) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  K2_CHECK(IsCompatible(src.shape, *dst));

  int32_t last_axis = src.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = src.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;
  K2_CHECK_EQ(num_rows, dst->Dim());

  ContextPtr &c = src.Context();
  const int32_t *row_splits = row_splits_array.Data();
  const T *values_data = src.values.Data();
  T *output_data = dst->Data();
  Op op;

  if (c->GetDeviceType() == kCpu) {
    int32_t j = row_splits[0];
    for (int32_t i = 0; i < num_rows; ++i) {
      T val = initial_value;
      int32_t row_end = row_splits[i + 1];
      for (; j < row_end; ++j) {
        T elem = values_data[j];
        val = op(elem, val);
      }
      output_data[i] = val;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);

    // This code is based on the example here:
    // https://nvlabs.github.io/cub/structcub_1_1_device_segmented_reduce.html
    std::size_t temp_storage_bytes = 0;

    // the first time is to determine temporary device storage requirements
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_bytes, values_data, output_data, num_rows,
        row_splits, row_splits + 1, op, initial_value, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage.Data(), temp_storage_bytes, values_data, output_data,
        num_rows, row_splits, row_splits + 1, op, initial_value,
        c->GetCudaStream()));
  }
}

template <typename T>
Ragged<T> AddPerSublist(Ragged<T> &src, Array1<T> &value, T alpha) {
  NVTX_RANGE(K2_FUNC);

  ContextPtr &context = src.Context();
  int32_t num_axes = src.NumAxes();
  K2_CHECK_EQ(src.TotSize(num_axes - 2), value.Dim());

  Array1<T> ans_values(context, src.values.Dim());
  Ragged<T> ans(src.shape, ans_values);

  const int32_t *row_ids_data = src.RowIds(num_axes - 1).Data();
  T *ans_data = ans.values.Data();
  const T *src_data = src.values.Data(),
          *value_data = value.Data();

  K2_EVAL(
      context, ans_values.Dim(), lambda_do_add, (int32_t i)->void {
        int32_t row = row_ids_data[i];
        T value = value_data[row];

        ans_data[i] = src_data[i] + value * alpha;
      });
  return ans;
}

template <typename T>
Ragged<T> NormalizePerSublist(Ragged<T> &src, bool use_log) {
  NVTX_RANGE(K2_FUNC);
  K2_STATIC_ASSERT(
      (std::is_same<float, T>::value || std::is_same<double, T>::value));
  T negative_infinity = -std::numeric_limits<T>::infinity();
  T eps = std::numeric_limits<T>::epsilon();

  ContextPtr &context = src.Context();
  int32_t num_axes = src.NumAxes();
  Array1<T> values(context, src.TotSize(num_axes - 2));

  if (use_log) {
    LogSumPerSublist<T>(src, negative_infinity, &values);
  } else {
    SumPerSublist<T>(src, 0, &values);
  }

  const T *values_data = values.Data();
  const int32_t *row_ids_data = src.RowIds(num_axes - 1).Data();

  Array1<T> ans_values(context, src.values.Dim());
  Ragged<T> ans(src.shape, ans_values);

  T *ans_data = ans.values.Data();
  const T *src_data = src.values.Data();

  if (use_log) {
    K2_EVAL(
        context, ans_values.Dim(), lambda_do_normalization, (int32_t i)->void {
          int32_t row = row_ids_data[i];
          T normalizer = values_data[row];

          ans_data[i] = src_data[i] - normalizer;
        });
  } else {
    K2_EVAL(
        context, ans_values.Dim(), lambda_do_normalization, (int32_t i)->void {
          int32_t row = row_ids_data[i];
          T normalizer = values_data[row] + eps;

          ans_data[i] = src_data[i] / normalizer;
        });
  }
  return ans;
}

template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, Ragged<T> **src,
                Array1<uint32_t> *merge_map /* = nullptr */) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  Array1<uint32_t> merge_map_temp;
  Array1<uint32_t> *merge_map_ptr =
      (merge_map != nullptr ? merge_map : &merge_map_temp);
  std::vector<RaggedShape *> src_shapes(num_srcs);
  std::vector<const Array1<T> *> src_values(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) {
    src_shapes[i] = &(src[i]->shape);
    src_values[i] = &(src[i]->values);
  }
  RaggedShape ans_shape =
      Stack(axis, num_srcs, src_shapes.data(), merge_map_ptr);
  Array1<T> ans_values =
      MergeWithMap(*merge_map_ptr, num_srcs, src_values.data());
  return Ragged<T>(ans_shape, ans_values);
}

template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, Ragged<T> *src,
                Array1<uint32_t> *merge_map /* = nullptr */) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(axis == 0 || axis == 1);
  K2_CHECK_GT(num_srcs, 0);
  std::vector<Ragged<T> *> temp(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) temp[i] = src + i;
  return Stack(axis, num_srcs, temp.data(), merge_map);
}

template <typename T>
void Unstack(Ragged<T> src, int32_t axis, bool pad_right,
             std::vector<Ragged<T>> *out,
             std::vector<Array1<int32_t>> *split_map /* = nullptr */) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(out != nullptr);
  ContextPtr &c = src.Context();
  std::vector<Array1<int32_t>> split_map_tmp;
  std::vector<Array1<int32_t>> *split_map_ptr =
      (split_map != nullptr ? split_map : &split_map_tmp);
  std::vector<RaggedShape> shape_out;

  Unstack(src.shape, axis, pad_right, &shape_out, split_map_ptr);

  out->resize(shape_out.size());
  // +1 here because we need to do ExclusiveSum on this Array1 later
  Array1<int32_t> elem_nums(GetCpuContext(), shape_out.size() + 1);
  Array1<T *> values_ptr(GetCpuContext(), shape_out.size());
  Array1<int32_t *> map_ptr(GetCpuContext(), shape_out.size());
  int32_t *elem_nums_data = elem_nums.Data();
  T **values_ptr_data = values_ptr.Data();
  int32_t **map_ptr_data = map_ptr.Data();

  int32_t tot_elems = 0;
  // Can not avoid this for loop as we want to allocate memory separately.
  for (size_t i = 0; i < shape_out.size(); ++i) {
    int32_t elem_num = shape_out[i].NumElements();
    out->at(i) = Ragged<T>(shape_out[i], Array1<T>(c, elem_num));
    elem_nums_data[i] = elem_num;
    tot_elems += elem_num;
    values_ptr_data[i] = out->at(i).values.Data();
    map_ptr_data[i] = split_map_ptr->at(i).Data();
  }

  Array1<int32_t> row_splits(c, shape_out.size() + 1);
  ExclusiveSum(elem_nums.To(c), &row_splits);

  Array1<int32_t> row_ids(c, tot_elems);
  RowSplitsToRowIds(row_splits, &row_ids);

  const int32_t *row_splits_data = row_splits.Data(),
                *row_ids_data = row_ids.Data();
  const T *src_value_data = src.values.Data();
  // Transfer to GPU if we are using a GPU
  map_ptr = map_ptr.To(c);
  map_ptr_data = map_ptr.Data();
  values_ptr = values_ptr.To(c);
  values_ptr_data = values_ptr.Data();

  K2_EVAL(c, tot_elems, lambda_set_values, (int32_t idx01) {
      int32_t idx0 = row_ids_data[idx01],
              idx0x = row_splits_data[idx0],
              idx1 = idx01 - idx0x;
      values_ptr_data[idx0][idx1] = src_value_data[map_ptr_data[idx0][idx1]];
  });
}

template <typename T>
Ragged<T> Cat(int32_t axis, int32_t num_srcs, Ragged<T> **src,
              Array1<uint32_t> *merge_map /* = nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  Array1<uint32_t> merge_map_temp;
  Array1<uint32_t> *merge_map_ptr =
      (merge_map != nullptr ? merge_map : &merge_map_temp);
  std::vector<RaggedShape *> src_shapes(num_srcs);
  std::vector<const Array1<T> *> src_values(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) {
    src_shapes[i] = &(src[i]->shape);
    src_values[i] = &(src[i]->values);
  }
  RaggedShape ans_shape = Cat(axis, num_srcs, src_shapes.data(), merge_map_ptr);
  Array1<T> ans_values =
      MergeWithMap(*merge_map_ptr, num_srcs, src_values.data());
  return Ragged<T>(ans_shape, ans_values);
}

template <typename T>
Ragged<T> Cat(int32_t axis, int32_t num_srcs, Ragged<T> *src,
              Array1<uint32_t> *merge_map /* = nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(axis == 0 || axis == 1) << "Given: " << axis;
  K2_CHECK_GT(num_srcs, 0);
  std::vector<Ragged<T> *> temp(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) temp[i] = src + i;
  return Cat(axis, num_srcs, temp.data());
}

template <typename T>
Ragged<T> Merge(int32_t num_srcs, Ragged<T> **src,
                const Array1<uint32_t> &merge_map,
                Array1<uint32_t> *merge_map_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GT(num_srcs, 0);
  Array1<uint32_t> merge_map_temp;
  Array1<uint32_t> *merge_map_ptr =
      (merge_map_out != nullptr ? merge_map_out : &merge_map_temp);
  std::vector<RaggedShape *> src_shapes(num_srcs);
  std::vector<const Array1<T> *> src_values(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) {
    src_shapes[i] = &(src[i]->shape);
    src_values[i] = &(src[i]->values);
  }
  RaggedShape ans_shape =
      Merge(num_srcs, src_shapes.data(), merge_map, merge_map_ptr);
  Array1<T> ans_values =
      MergeWithMap(*merge_map_ptr, num_srcs, src_values.data());
  return Ragged<T>(ans_shape, ans_values);
}

template <typename T>
Ragged<T> RemoveValuesLeq(Ragged<T> &src, T cutoff) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  Renumbering r(c, src.NumElements());
  const T *values_data = src.values.Data();
  char *keep = r.Keep().Data();
  K2_EVAL(
      c, src.NumElements(), lambda_set_keep,
      (int32_t i)->void { keep[i] = (char)(values_data[i] > cutoff); });
  return SubsetRagged(src, r);
}

template <typename T>
Ragged<T> RemoveValuesEq(Ragged<T> &src, T target) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  Renumbering r(c, src.NumElements());
  const T *values_data = src.values.Data();
  char *keep = r.Keep().Data();
  K2_EVAL(
      c, src.NumElements(), lambda_set_keep,
      (int32_t i)->void { keep[i] = (char)(values_data[i] != target); });
  return SubsetRagged(src, r);
}

// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos <= shape.TotSize(axis).
template <typename T>
void PrintRaggedPart(std::ostream &stream, const Ragged<T> &ragged,
                     int32_t axis, int32_t begin_pos, int32_t end_pos) {
  NVTX_RANGE(K2_FUNC);
  const auto &shape = ragged.shape;
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() && begin_pos >= 0 &&
           begin_pos <= end_pos && end_pos <= shape.TotSize(axis));
  for (int32_t d = begin_pos; d < end_pos; d++) {
    if (axis == shape.NumAxes() - 1) {
      stream << ragged.values[d] << " ";
    } else {
      stream << "[ ";
      const int32_t *row_splits = shape.RowSplits(axis + 1).Data();
      K2_DCHECK(d < shape.RowSplits(axis + 1).Dim());
      int32_t row_start = row_splits[d], row_end = row_splits[d + 1];
      PrintRaggedPart(stream, ragged, axis + 1, row_start, row_end);
      stream << "] ";
    }
  }
}

// prints a Ragged array as e.g. [ [ 7 9 ] [ 10 ] [] ]
template <typename T>
std::ostream &operator<<(std::ostream &stream, const Ragged<T> &ragged) {
  if (ragged.values.GetRegion() == nullptr)
    return stream << "<invalid Ragged<T> >";

  if (ragged.Context()->GetDeviceType() != kCpu) {
    return stream << ragged.To(GetCpuContext());
  } else {
    stream << "[ ";
    PrintRaggedPart(stream, ragged, 0, 0, ragged.shape.Dim0());
    stream << "]";
    return stream;
  }
}

template <typename T>
Ragged<T> RandomRagged(T min_value, T max_value, int32_t min_num_axes,
                       int32_t max_num_axes, int32_t min_num_elements,
                       int32_t max_num_elements) {
  RaggedShape shape = RandomRaggedShape(true, min_num_axes, max_num_axes,
                                        min_num_elements, max_num_elements);
  ContextPtr c = GetCpuContext();
  Array1<T> values =
      RandUniformArray1(c, shape.NumElements(), min_value, max_value);
  return Ragged<T>(shape, values);
}

// TODO(fangjun): add test cases for `order`
template <typename T, typename Op>
static void SortSublistsCpu(Ragged<T> *src, Array1<int32_t> *order) {
  T *p = src->values.Data();
  Op comp = Op();

  if (order != nullptr)
    std::iota(order->Data(), order->Data() + order->Dim(), 0);

  auto lambda_comp = [p, comp](int32_t i, int32_t j) {
    return comp(p[i], p[j]);
  };

  Array1<int32_t> &row_splits = src->shape.RowSplits(src->NumAxes() - 1);
  for (int32_t i = 0; i < row_splits.Dim() - 1; ++i) {
    int32_t cur = row_splits[i];
    int32_t next = row_splits[i + 1];
    if (order != nullptr)
      std::stable_sort(order->Data() + cur, order->Data() + next, lambda_comp);

    std::stable_sort(p + cur, p + next, comp);
  }
}

template <typename T, typename Op /* = LessThan<T> */>
void SortSublists(Ragged<T> *src, Array1<int32_t> *order /* = nullptr */) {
  if (order) {
    K2_DCHECK(IsCompatible(src->values, *order));
    K2_DCHECK_EQ(src->values.Dim(), order->Dim());
  }
  K2_DCHECK_GE(src->NumAxes(), 2);

  if (src->values.Dim() == 0) return;

  if (src->Context()->GetDeviceType() == kCpu) {
    SortSublistsCpu<T, Op>(src, order);
    return;
  }

  K2_DCHECK_EQ(src->Context()->GetDeviceType(), kCuda);

  mgpu::context_t *context = GetModernGpuAllocator(src->Context());

  Array1<int32_t> &segment = src->shape.RowSplits(src->NumAxes() - 1);
  if (order)
    K2_CUDA_SAFE_CALL(
        mgpu::segmented_sort_indices(src->values.Data(),  // keys
                                     order->Data(),       // indices
                                     src->values.Dim(),   // count
                                     segment.Data(),      // segments
                                     segment.Dim() - 1,   // num_segments
                                     Op(),                // cmp
                                     *context));          // context
  else
    K2_CUDA_SAFE_CALL(mgpu::segmented_sort(src->values.Data(),  // keys
                                           src->values.Dim(),   // count
                                           segment.Data(),      // segments
                                           segment.Dim() - 1,   // num_segments
                                           Op(),                // cmp
                                           *context));          // context
}

template <typename T>
void Ragged<T>::Check() const {
  if (values.Dim() != shape.NumElements()) {
    K2_LOG(FATAL) << "Dimension mismatch: values.Dim() == " << values.Dim()
                  << " vs. shape.NumElements() == " << shape.NumElements();
  }
  shape.Check();
}

// Defined here and not in ragged.h because it needs RemoveAxis(RaggedShape&,
// int).
template <typename T>
Ragged<T> Ragged<T>::RemoveAxis(int32_t axis) {
  K2_CHECK(NumAxes() > 2 && axis >= 0 && axis < NumAxes() - 1);
  RaggedShape new_shape = ::k2::RemoveAxis(shape, axis);
  return Ragged<T>(new_shape, values);
}

template <typename T>
std::istream &operator>>(std::istream &is, Ragged<T> &r) {
  // Note: the top element of 'row_splits' will end up being
  // discarded; the others will become the axes of `r`
  std::vector<std::vector<int32_t>> row_splits;
  std::vector<T> elems;
  int32_t cur_level = 0;
  is >> std::ws;  // eat whitespace
  while (1) {
    // We exit the loop after reading the final '['.
    char c = is.peek();
    if (c == '[') {
      cur_level++;
      while (row_splits.size() < static_cast<size_t>(cur_level)) {
        if (!elems.empty()) {
          is.setstate(std::ios::failbit);
          return is;
        }
        row_splits.push_back(std::vector<int32_t>(1, 0));
      }
      is.get();  // consume character 'c'
    } else if (c == ']') {
      cur_level--;
      if (cur_level < 0) {  // ']' without '['.
        is.setstate(std::ios::failbit);
        return is;
      }
      row_splits[cur_level].push_back(
          (cur_level + 1 >= (int32_t)row_splits.size())
              ? static_cast<int32_t>(elems.size())
              : (row_splits[cur_level + 1].size() - 1));
      is.get();  // consume character 'c'
      if (cur_level == 0) break;
    } else if (c == ',') {
      is.get();  // consume character 'c'
    } else {
      InputFixer<T> t;
      is >> t;
      if (!is.good() || cur_level != static_cast<int32_t>(row_splits.size()) ||
          cur_level < 2) {
        is.setstate(std::ios::failbit);
        return is;
      }
      elems.push_back(t);
    }
    is >> std::ws;
  }

  if (row_splits.empty() || row_splits[0].size() != 2) {
    is.setstate(std::ios::failbit);
    return is;
  }
  row_splits.erase(row_splits.begin());
  if (row_splits.empty()) {
    // Assume 2 axes even though the num-axes is ambiguous from the input "[ ]"
    // row_splits is [ 0 ].
    row_splits.push_back(std::vector<int32_t>(1, 0));
  }
  std::vector<RaggedShapeLayer> axes(row_splits.size());
  for (size_t i = 0; i < row_splits.size(); i++) {
    axes[i].row_splits = Array1<int32_t>(GetCpuContext(), row_splits[i]);
    axes[i].cached_tot_size = -1;
  }
  r.shape = RaggedShape(axes);
  r.values = Array1<T>(GetCpuContext(), elems);
  K2_CHECK(r.values.Dim() == r.shape.NumElements());
  return is;
}

template <typename T>
Ragged<T> Index(Ragged<T> &src, Ragged<int32_t> &indexes, bool remove_axis) {
  Ragged<T> r = Index(src, 0, indexes.values);
  RaggedShape s = ComposeRaggedShapes(indexes.shape, r.shape);
  Ragged<T> ans(s, r.values);
  return (remove_axis ? RemoveAxis(ans, ans.NumAxes() - 2) : ans);
}

namespace argmax_internal {
template <typename T>
struct Pair {
  T t;
  int32_t idx;
};

template <typename T>
struct PairInputIterator {
  explicit PairInputIterator(const T *t) : t_(t), offset_(0) {}
  __device__ __forceinline__ PairInputIterator(const T *t, int32_t offset)
      : t_(t), offset_(offset) {}
  __device__ __forceinline__ Pair<T> operator[](int32_t idx) const {
    return Pair<T>{t_[idx + offset_], idx + offset_};
  }
  __device__ __forceinline__ PairInputIterator operator+(int32_t offset) {
    return PairInputIterator{t_, offset + offset_};
  }
  const T *t_;
  int32_t offset_;
};

template <typename T>
struct PairOutputIteratorDeref {  // this is what you get when you dereference
                                  // PairOutputIterator, it pretends to be a
                                  // Pair<T> but really only stores the `idx`
                                  // member.
  explicit __device__ __forceinline__ PairOutputIteratorDeref(int32_t *i)
      : i_(i) {}
  __device__ __forceinline__ PairOutputIteratorDeref &operator=(
      const Pair<T> &p) {
    *i_ = p.idx;
    return *this;
  }
  int32_t *i_;
};

template <typename T>
struct PairOutputIterator {  // outputs just the index of the pair.
  explicit PairOutputIterator(int32_t *i) : i_(i) {}
  __device__ __forceinline__ PairOutputIteratorDeref<T> operator[](
      int32_t idx) const {
    return PairOutputIteratorDeref<T>(i_ + idx);
  }
  __device__ __forceinline__ PairOutputIterator operator+(int32_t offset) {
    return PairOutputIterator{i_ + offset};
  }
  int32_t *i_;
};

template <typename T>
struct PairMaxOp {
  __device__ __forceinline__ Pair<T> operator()(const Pair<T> &a,
                                                const Pair<T> &b) const {
    // NOTE: could specialize this via a union, if T == int32_t, might be
    // marginally faster.
    if (a.t > b.t || (a.t == b.t && a.idx > b.idx)) return a;
    return b;
  }
};

}  // namespace argmax_internal
}  // namespace k2

namespace std {
// those below typedefs are required by cub::DeviceSegmentedReduce:Reduce
template <typename T>
struct iterator_traits<k2::argmax_internal::PairInputIterator<T>> {
  typedef k2::argmax_internal::Pair<T> value_type;
};
template <typename T>
struct iterator_traits<k2::argmax_internal::PairOutputIterator<T>> {
  typedef k2::argmax_internal::Pair<T> value_type;
  typedef k2::argmax_internal::PairOutputIteratorDeref<T> reference;
};
}  // namespace std

namespace k2 {
template <typename T>
void ArgMaxPerSublist(Ragged<T> &src, T initial_value, Array1<int32_t> *dst) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(src.NumAxes(), 2);
  K2_CHECK(IsCompatible(src.shape, *dst));

  int32_t last_axis = src.NumAxes() - 1;
  const Array1<int32_t> &row_splits_array = src.RowSplits(last_axis);
  int32_t num_rows = row_splits_array.Dim() - 1;
  K2_CHECK_EQ(num_rows, dst->Dim());

  ContextPtr &c = src.Context();
  const int32_t *row_splits = row_splits_array.Data();
  const T *values_data = src.values.Data();
  int32_t *output_data = dst->Data();

  if (c->GetDeviceType() == kCpu) {
    int32_t j = row_splits[0];
    for (int32_t i = 0; i < num_rows; ++i) {
      T val = initial_value;
      int32_t idx = -1;

      int32_t row_end = row_splits[i + 1];
      for (; j < row_end; ++j) {
        T elem = values_data[j];
        if (elem >= val) {
          val = elem;
          idx = j;
        }
      }
      output_data[i] = idx;
    }
  } else {
    K2_CHECK_EQ(c->GetDeviceType(), kCuda);
    argmax_internal::PairInputIterator<T> input_iter(values_data);
    argmax_internal::PairOutputIterator<T> output_iter(output_data);
    argmax_internal::PairMaxOp<T> op;
    argmax_internal::Pair<T> initial_pair{initial_value, -1};

    // This code is based on the example here:
    // https://nvlabs.github.io/cub/structcub_1_1_device_segmented_reduce.html
    std::size_t temp_storage_bytes = 0;

    // the first time is to determine temporary device storage requirements
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_bytes, input_iter, output_iter, num_rows,
        row_splits, row_splits + 1, op, initial_pair, c->GetCudaStream()));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage.Data(), temp_storage_bytes, input_iter, output_iter,
        num_rows, row_splits, row_splits + 1, op, initial_pair,
        c->GetCudaStream()));

    // Do the same thing as the code above, but it need one more kernel to
    // add offset to the result.
#if 0
    const Array1<int32_t> &row_ids_array = src.RowIds(last_axis);
    const int32_t *row_ids = row_ids_array.Data();
    size_t align = alignof(cub::KeyValuePair<int, T>);
    Array1<int8_t> out_storage(c,
        num_rows * sizeof(cub::KeyValuePair<int, T>) + align);
    size_t shift = reinterpret_cast<std::intptr_t>(out_storage.Data()) % align;
    shift = shift ? align - shift : 0;

    cub::KeyValuePair<int, T> *d_out_data =
      reinterpret_cast<cub::KeyValuePair<int, T>*>(out_storage.Data() + shift);

    std::size_t temp_storage_bytes = 0;

    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::ArgMax(
        nullptr, temp_storage_bytes, values_data, d_out_data, num_rows,
        row_splits, row_splits + 1));
    Array1<int8_t> d_temp_storage(c, temp_storage_bytes);
    K2_CUDA_SAFE_CALL(cub::DeviceSegmentedReduce::ArgMax(
        d_temp_storage.Data(), temp_storage_bytes, values_data, d_out_data,
        num_rows, row_splits, row_splits + 1));

    K2_EVAL(
        c, num_rows, lambda_add_offset, (int32_t idx0) -> void {
          if (row_splits[idx0 + 1] == row_splits[idx0])
            output_data[idx0] = -1;
          else
            output_data[idx0] = d_out_data[idx0].key + row_splits[idx0];
        });
#endif
  }
}

template <typename T>
void SegmentedExclusiveSum(Ragged<T> &src, Array1<T> *dst) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetContext(src, *dst);
  int32_t dim = dst->Dim();
  K2_CHECK_EQ(src.NumElements(), dim);

  const int32_t *row_splits_data = src.RowSplits(src.NumAxes() - 1).Data();
  const int32_t *row_ids_data = src.RowIds(src.NumAxes() - 1).Data();
  T *dst_data = dst->Data();
  if (c->GetDeviceType() == kCuda) {
    // there's roundoff problem for float type with the below implementation in
    // else branch.
    if (std::is_same<float, T>::value || std::is_same<double, T>::value) {
      // flags is similar to `tails` (see concepts in k2/csrc/utils)
      // But it indicates `heads` here. The very first segment always
      // starts at zero, so flags[0] is always 0.
      Array1<uint32_t> flags(c, dim);
      uint32_t *flags_data = flags.Data();
      K2_EVAL(
          c, dim, set_flags, (int32_t idx01)->void {
            int32_t idx0 = row_ids_data[idx01];
            int32_t idx0x = row_splits_data[idx0];
            int32_t idx0x_next = row_splits_data[idx0 + 1];
            if (idx0x < idx0x_next) {
              if (idx01 == idx0x)
                flags_data[idx01] = idx01 == 0 ? 0 : 1;
              else
                flags_data[idx01] = 0;
            }
          });
      SegmentedExclusiveSum(c, src.values.Data(), dim, flags_data, dst->Data());
    } else {
      Array1<T> exclusive_sum(c, dim);
      ExclusiveSum(src.values, &exclusive_sum);
      const T *exclusive_sum_data = exclusive_sum.Data();
      K2_EVAL(
          c, dim, set_ans_values, (int32_t idx01)->void {
            int32_t idx0 = row_ids_data[idx01];
            int32_t idx0x = row_splits_data[idx0];
            dst_data[idx01] =
                exclusive_sum_data[idx01] - exclusive_sum_data[idx0x];
          });
    }
  } else {
    // Though the above code for Cuda would be working for cpu as well, we still
    // add an implementation for cpu here as it only needs one iteration
    K2_CHECK_EQ(c->GetDeviceType(), kCpu);
    const T *src_values_data = src.values.Data();
    int32_t dim0 = src.TotSize(src.NumAxes() - 2);
    for (int32_t i = 0; i != dim0; ++i) {
      T sum = 0;
      int32_t row_begin = row_splits_data[i];
      int32_t row_end = row_splits_data[i + 1];
      for (int32_t n = row_begin; n != row_end; ++n) {
        auto prev = src_values_data[n];  // save a copy since src.values and
                                         // dest may share the underlying memory
        dst_data[n] = sum;
        sum += prev;
      }
    }
  }
}

template <typename T>
Ragged<T> CreateRagged2(const std::vector<std::vector<T>> &vecs) {
  NVTX_RANGE(K2_FUNC);
  std::vector<T> values;
  std::vector<int32_t> row_splits;
  row_splits.reserve(vecs.size() + 1);
  row_splits.push_back(0);
  for (const auto &vec : vecs) {
    values.insert(values.end(), vec.begin(), vec.end());
    row_splits.push_back(values.size());
  }
  ContextPtr context = GetCpuContext();
  Array1<int32_t> row_splits_array(context, row_splits);
  RaggedShape shape = RaggedShape2(&row_splits_array, nullptr, values.size());
  Array1<T> values_array(context, values);
  return Ragged<T>(shape, values_array);
}

template <typename T>
Array2<T> PadRagged(Ragged<T> &src, const std::string &mode, T padding_value) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src.NumAxes(), 2);

  bool is_constant = false;
  if (mode == "constant") {
    is_constant = true;
  } else if (mode == "replicate") {
    is_constant = false;
  } else {
    K2_LOG(FATAL) << "Unsupported mode: " << mode << ".\n"
                  << "Valid values are: constant, replicate";
  }

  ContextPtr &c = src.Context();
  int32_t row_num = src.Dim0(),
          col_num = src.shape.MaxSize(1);
  Array2<T> res(c, row_num, col_num);
  auto res_acc = res.Accessor();
  const T *src_values_data = src.values.Data();
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data();
  if (is_constant) {
    K2_EVAL2(
        c, res.Dim0(), res.Dim1(), lambda, (int32_t i, int32_t j)->void {
          int32_t idx0x = src_row_splits1_data[i],
                  idx0x_next = src_row_splits1_data[i + 1],
                  len = idx0x_next - idx0x;
          if (j >= len)
            res_acc(i, j) = padding_value;
          else
            res_acc(i, j) = src_values_data[idx0x + j];
        });
  } else {
    K2_EVAL2(
        c, res.Dim0(), res.Dim1(), lambda, (int32_t i, int32_t j)->void {
          int32_t idx0x = src_row_splits1_data[i],
                  idx0x_next = src_row_splits1_data[i + 1],
                  len = idx0x_next - idx0x;

          if (len == 0) {
            res_acc(i, j) = padding_value;
            return;
          }

          if (j >= len)
            // replicate the last element in this list
            res_acc(i, j) = src_values_data[idx0x_next - 1];
          else
            res_acc(i, j) = src_values_data[idx0x + j];
        });
  }
  return res;
}

/* Prune a three axes ragged tensor on axis1
 * This is a special case of PruneRagged with axis == 1 and src.NumAxes() == 3,
 * To get more details, please refer to the docs for PruneRagged in
 * ragged_ops.h.
 */
template <typename T>
Renumbering PruneRaggedAxis1(Ragged<T> &src, T beam,
                             int32_t max_elems) {
  K2_CHECK_EQ(src.NumAxes(), 3);
  ContextPtr &c = src.Context();
  int32_t total_elements = src.TotSize(1);
  Renumbering renumbering(c, total_elements);

  T negative_infinity = -std::numeric_limits<T>::infinity();
  Array1<T> sub_max(c, src.TotSize(1));
  MaxPerSublist<T>(src, negative_infinity, &sub_max);

  Array1<T> best_scores(c, src.TotSize(0));
  Ragged<T> ragged_sub_max(RemoveAxis(src.shape, 2), sub_max);
  MaxPerSublist<T>(ragged_sub_max, negative_infinity, &best_scores);

  bool prune_with_max_elems =
      max_elems > 0 && max_elems < total_elements;

  Array1<int32_t> order_map;
  const int32_t *order_map_data;
  if (prune_with_max_elems) {
    Ragged<T> sorted_sub_max = ragged_sub_max.Clone();
    order_map = Array1<int32_t>(c, total_elements);
    SortSublists<T, GreaterThan<T>>(&sorted_sub_max, &order_map);
    order_map_data = order_map.Data();
  }

  char *keep_data = renumbering.Keep().Data();
  const T *sub_max_data = sub_max.Data(),
          *best_scores_data = best_scores.Data(),
          *src_data = src.values.Data();
  const int32_t *row_ids1_data = src.RowIds(1).Data(),
                *row_splits1_data = src.RowSplits(1).Data();
  // prune_with_max_elems means we have sorted the sub-max ragged tensor
  if (prune_with_max_elems) {
    K2_EVAL(c, total_elements, lambda_set_keep_sorted, (int32_t idx01) {
                // idx01 is the index after sorting
        int32_t original_idx01 = order_map_data[idx01],
                // SortSublists wouldn't change idx0 and idx0x
                idx0 = row_ids1_data[original_idx01],
                idx0x = row_splits1_data[idx0],
                // idx1 is the index after sorting
                idx1 = idx01 - idx0x;
        bool pruned_by_max_elems = idx1 >= max_elems,
             pruned_by_beam =
                 sub_max_data[original_idx01] < best_scores_data[idx0] - beam;
        keep_data[original_idx01] =
            !(pruned_by_max_elems || pruned_by_beam);
    });
  } else {
    K2_EVAL(c, total_elements, lambda_set_keep, (int32_t idx01) {
        int32_t idx0 = row_ids1_data[idx01];
        keep_data[idx01] = sub_max_data[idx01] >= best_scores_data[idx0] - beam;
    });
  }
  return renumbering;
}

template <typename T>
Renumbering PruneRagged(Ragged<T> &src, int32_t axis, T beam,
                        int32_t max_elems) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  if (axis == 0) {
    auto new_shape = ComposeRaggedShapes(
            TrivialShape(c, src.TotSize(0)), src.shape);
    auto new_src = Ragged<T>(new_shape, src.values);
    while (new_src.NumAxes() > 3) {
      new_src = RemoveAxis(new_src, new_src.NumAxes() - 2);
    }
    return PruneRaggedAxis1(new_src, beam, max_elems);
  } else if (axis == src.NumAxes() - 1) {
    auto new_shape = ComposeRaggedShapes(src.shape,
         RegularRaggedShape(c, src.NumElements(), 1));
    auto new_src = Ragged<T>(new_shape, src.values);
    while (new_src.NumAxes() > 3) {
      new_src = RemoveAxis(new_src, 0);
    }
    return PruneRaggedAxis1(new_src, beam, max_elems);
  } else {
    auto new_src = src;
    while (--axis) {
     new_src = RemoveAxis(new_src, 0);
    }
    while (new_src.NumAxes() > 3) {
      new_src = RemoveAxis(new_src, new_src.NumAxes() - 2);
    }
    return PruneRaggedAxis1(new_src, beam, max_elems);
  }
}

}  // namespace k2

#endif  // K2_CSRC_RAGGED_OPS_INL_H_
