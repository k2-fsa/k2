/**
 * @brief
 * ragged_ops_inl
 *
 * @note
 * This is to be included only from ragged.h.
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
#include <memory>
#include <numeric>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "moderngpu/kernel_segsort.hxx"

namespace k2 {

template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, const Ragged<T> **src) {
  K2_CHECK_EQ(axis, 0);
  K2_CHECK_GT(num_srcs, 0);  // can later relax this, maybe
  std::vector<const RaggedShape *> src_shapes(num_srcs);
  std::vector<const Array1<T> *> src_values(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) {
    src_shapes[i] = &(src[i]->shape);
    src_values[i] = &(src[i]->values);
  }
  // below line will check if srcs are compatible with each other or not, i.e.
  // context compatibility and num-axes compatibility.
  RaggedShape ans_shape = Stack(axis, num_srcs, src_shapes.data());
  Array1<T> ans_values = Append(num_srcs, src_values.data());
  return Ragged<T>(ans_shape, ans_values);
}

template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, const Ragged<T> *src) {
  K2_CHECK_EQ(axis, 0);
  K2_CHECK_GT(num_srcs, 0);  // can later relax this, maybe
  std::vector<const Ragged<T> *> temp(num_srcs);
  for (int32_t i = 0; i != num_srcs; ++i) temp[i] = src + i;
  return Stack(axis, num_srcs, temp.data());
}

// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos <= shape.TotSize(axis).
template <typename T>
void PrintRaggedPart(std::ostream &stream, Ragged<T> &ragged, int32_t axis,
                     int32_t begin_pos, int32_t end_pos) {
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
  if (ragged.Context().GetDeviceType() != kCpu) {
    return stream << ragged.To(GetCpuContext());
  } else {
    stream << "[ ";
    PrintRaggedPart(ragged, stream, 0, 0, ragged.shape.Dim0());
    stream << "]";
    return stream;
  }
}

template <typename T>
Ragged<T> RandomRagged(T min_value, T max_value, int32_t min_num_axes,
                       int32_t max_num_axes, int32_t min_num_elements,
                       int32_t max_num_elements) {
  RaggedShape shape = RandomRaggedShape(min_num_axes, max_num_axes,
                                        min_num_elements, max_num_elements);
  Array1<T> values = RandUniformArray1(GetCpuContext(), shape.NumElements(),
                                       min_value, max_value);
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
      std::sort(order->Data() + cur, order->Data() + next, lambda_comp);

    std::sort(p + cur, p + next, comp);
  }
}

template <typename T, typename Op /* = LessThan<T> */>
void SortSublists(Ragged<T> *src, Array1<int32_t> *order /* = nullptr */) {
  if (order) {
    K2_DCHECK(IsCompatible(src->values, *order));
    K2_DCHECK_EQ(src->values.Dim(), order->Dim());
  }
  K2_DCHECK_GE(src->NumAxes(), 2);
  if (src->Context()->GetDeviceType() == kCpu) {
    SortSublistsCpu<T, Op>(src, order);
    return;
  }

  K2_DCHECK_EQ(src->Context()->GetDeviceType(), kCuda);

  std::unique_ptr<mgpu::context_t> context =
      GetModernGpuAllocator(src->Context()->GetDeviceId());

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
bool Ragged<T>::Validate(bool print_warnings) {
  if (values.Dim() != shape.NumElements()) {
    if (print_warnings) {
      K2_LOG(WARNING) << "Dimension mismatch: values.Dim() == "
                      << values.Dim() << " vs. shape.NumElements() == "
                      << shape.NumElements();
    }
    return false;
  }
  return shape.Validate();
}

// Defined here and not in ragged.h because it needs RemoveAxis.
template <typename T>
Ragged<T> Ragged<T>::RemoveAxis(int32_t axis) {
  K2_CHECK(axis >= 0 && axis < NumAxes());
  RaggedShape new_shape = ::k2::RemoveAxis(shape, axis);
  return Ragged<T>(new_shape, values);
}


}  // namespace k2

#endif  // K2_CSRC_RAGGED_OPS_INL_H_
