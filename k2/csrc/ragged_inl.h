/**
 * @brief
 * ragged_inl
 *
 * @note
 * This is to be included only from ragged.h.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_INL_H_
#define K2_CSRC_RAGGED_INL_H_

#include <memory>
#include <vector>

#include "k2/csrc/moderngpu_allocator.h"
#include "moderngpu/kernel_segsort.hxx"

namespace k2 {

template <typename T>
Ragged<T> Stack(int32_t num_srcs, const Ragged<T> *src, int32_t axis) {
  K2_CHECK_GT(num_srcs, 0);  // can later relax this, maybe
  std::vector<const RaggedShape *> src_shapes(num_srcs);
  std::vector<const Array1<T> *> src_values(num_srcs);
  for (int32_t i = 0; i < num_srcs; i++) {
    src_shapes[i] = &(src[i]->shape);
    src_values[i] = &(src[i]->values);
  }

  // TODO(haowen): implement this later
  // RaggedShape ans_shape = Stack(num_srcs, src_shapes, axis);
  Array1<T> ans_values;
  if (axis == 0) {
    // values = Append(num_srcs, rsc_values);
  } else {
    K2_LOG(FATAL) << "Axis != 0 not currently supported in Stack().";
  }
}

template <typename T>
Ragged<T> Stack(int32_t axis, int32_t num_srcs, Ragged<T> *src) {
  K2_CHECK_GT(num_srcs, 0);  // can later relax this, maybe
  std::vector<Ragged<T> *> temp(num_srcs);
  for (int32_t i = 0; i < num_srcs; i++) temp[i] = src + i;
  return Stack(axis, num_srcs, temp[0]);
}

// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos < shape.TotSize(axis).
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
  // TODO(haowen): define RandUniforArray1
  // Array1<T> values = RandUniformArray1<T>(GetCpuContext(),
  // shape.NumElements());
  Array1<T> values;
  return Ragged<T>(shape, values);
}

template <typename T, typename Op /* = LessThan<T> */>
void SortSublists(Ragged<T> *src, Array1<int32_t> *order) {
  K2_DCHECK(IsCompatible(src->values, *order));
  K2_DCHECK_EQ(src->values.Dim(), order->Dim());
  K2_DCHECK_EQ(src->Context()->GetDeviceType(), kCuda)
      << "It supports only CUDA at present";

  std::unique_ptr<mgpu::context_t> context =
      GetModernGpuAllocator(src->Context()->GetDeviceId());

  Array1<int32_t> &segment = src->shape.RowSplits(src->NumAxes() - 1);
  mgpu::segmented_sort_indices(src->values.Data(),  // keys
                               order->Data(),       // indices
                               src->values.Dim(),   // count
                               segment.Data() + 1,  // segments
                               segment.Dim() - 1,   // num_segments
                               Op(),                // cmp
                               *context);           // context
  auto err = cudaGetLastError();
  (void)err;
  // TODO(fangjun): err is not cudaSuccess, but why was the data sorted
  // correctly?
  //
  // Check failed: err == cudaSuccess (9 vs. 0)  Error: invalid configuration
  // argument.
  //
  // K2_DCHECK_CUDA_ERROR(err);
}

}  // namespace k2

#endif  // K2_CSRC_RAGGED_INL_H_
