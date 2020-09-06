// k2/csrc/cuda/ragged_inl.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

// This is to be included only from ragged.h.

namespace k2 {



template <typename T>
Ragged<T> Stack(int32_t axis, int32_t src_size, const Ragged<T> *src) {
  CHECK_GT(src_size, 0);  // can later relax this, maybe
  std::vector<const RaggedShape *> src_shapes(src_size);
  std::vector<const Array1<T> *> src_values(src_size);
  for (int32_t i = 0; i < src_size; i++) {
    src_shapes[i] = &(src[i]->shape);
    src_values[i] = &(src[i]->values);
  }
  // TODO.
  if (axis == 0) {
    //  return Ragged<T>(Stack(axis, src_size, &(src_shapes[0])),
    //                  Append(src_size, src_values));
  } else {
    assert(0);  // Have to figure out whether it makes sense to
                // support this case here.
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
      const int32_t row_splits = shape.RowSplits(axis + 1).Data();
      K2_DCHECK(d < shape.RowSplits(axis + 1).Dim());
      int32_t row_start = row_splits[d],
          row_end = row_splits[d+1];
      PrintRaggedPart(stream, shape, axis + 1,
                      row_start, row_end);
      stream << "] ";
    }
  }
}

// prints a Ragged array as e.g. [ [ 7 9 ] [ 10 ] [] ]
template <typename T>
std::ostream &operator<<(std::ostream &stream, const Ragged<T> &ragged) {
  if (ragged.Context().GetDeviceType() != kCpuDevice) {
    return stream << ragged.To(CpuContext());
  } else {
    stream << "[ ";
    PrintRaggedPart(ragged, stream, 0, 0, ragged.shape.Dim0());
    stream << "]";
    return stream;
  }
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, const Ragged<T> &r) {
}

}  // namespace k2
