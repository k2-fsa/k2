/**
 * @brief
 * ragged_ops_inl
 *
 * @note
 * This is to be included only from ragged_ops.h.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
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
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "moderngpu/kernel_segsort.hxx"

namespace k2 {

template <typename T, typename Op>
void ApplyOpPerSublist(Ragged<T> &src, T initial_value, Array1<T> *dst) {
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
Ragged<T> NormalizePerSublist(Ragged<T> &src) {
  NVTX_RANGE(K2_FUNC);
  K2_STATIC_ASSERT(
      (std::is_same<float, T>::value || std::is_same<double, T>::value));
  T negative_infinity = -std::numeric_limits<T>::infinity();

  ContextPtr &context = src.Context();
  int32_t num_axes = src.NumAxes();
  Array1<T> values(context, src.TotSize(num_axes - 2));
  LogSumPerSublist(src, negative_infinity, &values);

  const T *values_data = values.Data();
  const int32_t *row_ids_data = src.RowIds(num_axes - 1).Data();

  Array1<T> ans_values(context, src.values.Dim());
  Ragged<T> ans(src.shape, ans_values);

  T *ans_data = ans.values.Data();
  const T *src_data = src.values.Data();

  K2_EVAL(
      context, ans_values.Dim(), lambda_do_normalization, (int32_t i)->void {
        int32_t row = row_ids_data[i];
        T normalizer = values_data[row];

        ans_data[i] = src_data[i] - normalizer;
      });
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
Ragged<T> Append(int32_t axis, int32_t num_srcs, Ragged<T> **src,
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
  RaggedShape ans_shape =
      Append(axis, num_srcs, src_shapes.data(), merge_map_ptr);
  Array1<T> ans_values =
      MergeWithMap(*merge_map_ptr, num_srcs, src_values.data());
  return Ragged<T>(ans_shape, ans_values);
}

template <typename T>
Ragged<T> Append(int32_t axis, int32_t num_srcs, Ragged<T> *src,
                 Array1<uint32_t> *merge_map /* = nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(axis == 0 || axis == 1);
  K2_CHECK_GT(num_srcs, 0);
  std::vector<Ragged<T> *> temp(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) temp[i] = src + i;
  return Append(axis, num_srcs, temp.data());
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
  ContextPtr &c = src.Context();
  Renumbering r(c, src.NumElements());
  const T *values_data = src.values.Data();
  char *keep = r.Keep().Data();
  K2_EVAL(
      c, src.NumElements(), lambda_set_keep,
      (int32_t i)->void { keep[i] = (char)(values_data[i] > cutoff); });
  return SubsampleRagged(src, r);
}

template <typename T>
Ragged<T> RemoveValuesEq(Ragged<T> &src, T target) {
  ContextPtr &c = src.Context();
  Renumbering r(c, src.NumElements());
  const T *values_data = src.values.Data();
  char *keep = r.Keep().Data();
  K2_EVAL(
      c, src.NumElements(), lambda_set_keep,
      (int32_t i)->void { keep[i] = (char)(values_data[i] != target); });
  return SubsampleRagged(src, r);
}

// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos <= shape.TotSize(axis).
template <typename T>
void PrintRaggedPart(std::ostream &stream, const Ragged<T> &ragged,
                     int32_t axis, int32_t begin_pos, int32_t end_pos) {
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
  NVTX_RANGE(K2_FUNC);
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
      std::sort(order->Data() + cur, order->Data() + next, lambda_comp);

    std::sort(p + cur, p + next, comp);
  }
}

template <typename T, typename Op /* = LessThan<T> */>
void SortSublists(Ragged<T> *src, Array1<int32_t> *order /* = nullptr */) {
  NVTX_RANGE(K2_FUNC);
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
bool Ragged<T>::Validate(bool print_warnings) const {
  NVTX_RANGE(K2_FUNC);
  if (values.Dim() != shape.NumElements()) {
    if (print_warnings) {
      K2_LOG(WARNING) << "Dimension mismatch: values.Dim() == " << values.Dim()
                      << " vs. shape.NumElements() == " << shape.NumElements();
    }
    return false;
  }
  return shape.Validate(print_warnings);
}

// Defined here and not in ragged.h because it needs RemoveAxis(RaggedShape&,
// int).
template <typename T>
Ragged<T> Ragged<T>::RemoveAxis(int32_t axis) {
  NVTX_RANGE(K2_FUNC);
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
  Ragged<T> r = Index(src, indexes.values);
  RaggedShape s = ComposeRaggedShapes(indexes.shape, r.shape);
  Ragged<T> ans(s, r.values);
  return (remove_axis ? RemoveAxis(ans, ans.NumAxes() - 2) : ans);
}

}  // namespace k2

#endif  // K2_CSRC_RAGGED_OPS_INL_H_
