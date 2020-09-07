/**
 * @brief
 * ragged_inl
 *
 * @note
 * This is to be included only from ragged.h.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

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

  RaggedShape ans_shape = Stack(num_srcs, src_shapes, axis);
  Array1<T> ans_values;
  if (axis == 0) {
    values = Append(num_srcs, rsc_values);
  } else {
    K2_LOG(FATAL) << "Axis != 0 not currently supported in Stack().";
  }
}


// Recursive function that prints (part of) a ragged shape.
// 0 <=  begin_pos <= end_pos < shape.TotSize(axis).
template <typename T>
void PrintRaggedPart(std::ostream &stream, Ragged<T> &ragged,
                     int32_t axis,
                     int32_t begin_pos, int32_t end_pos) {
  K2_CHECK(axis >= 0 && axis < shape.NumAxes() &&
           begin_pos >= 0 && begin_pos <= end_pos &&
           end_pos <= shape.TotSize(axis));
  for (int32_t d = begin_pos; d < end_pos; d++) {
    if (axis == shape.NumAxes() - 1) {
      stream << ragged.values[d] << " ";
    } else {
      stream << "[ ";
      const int32_t *row_splits = shape.RowSplits(axis + 1).Data();
      K2_DCHECK(d < shape.RowSplits(axis + 1).Dim());
      int32_t row_start = row_splits[d],
          row_end = row_splits[d+1];
      PrintRaggedPart(stream, shape, axis + 1,  row_start, row_end);
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
Ragged<T> RandomRagged<T>(T min_value, T max_value,
                          int32_t min_num_axes , int32_t max_num_axes,
                          int32_t min_num_elements, int32_t max_num_elements) {
  RaggedShape shape = RandomRaggedShape(min_num_axes, max_num_axes,
                                        min_num_elements, max_num_elements);
  Array1<T> values = RandUniformArray1<T>(CpuContext(), shape.NumElements());
  return Ragged<T>(shape, values);
}



}  // namespace k2
