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

}  // namespace k2
