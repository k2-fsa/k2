// k2/csrc/array.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/array.h"

#include "k2/csrc/fsa.h"

namespace k2 {

K2_INSTANTIATE_ARRAY1(Arc *, int32_t);
K2_INSTANTIATE_ARRAY1(int32_t *, int32_t);
K2_INSTANTIATE_ARRAY1(float *, int32_t);

K2_INSTANTIATE_ARRAY2(Arc *, int32_t);

}  // namespace k2
