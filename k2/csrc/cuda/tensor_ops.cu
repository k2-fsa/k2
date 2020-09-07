// k2/csrc/cuda/tensor_ops.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/cuda/tensor_ops.h"

namespace k2 {


void CopyTensorElements(Tensor src, Tensor dest) {
  K2_CHECK(src.SameDim(dest));
  ContextPtr c = GetContext(src, dest);

  if (src.Ndim() > 2) {
    // For now, we only directly support copies of at most 2 dims.  Of course
    // this is not ideal.
    int32_t leading_dim = src.Dim(0);
    ParallelRunner pr(c);

    for (int32_t i = 0; i < leading_dim; i++) {
      Tensor src_part = src.Index(0, i),
          dest_part = dest.Index(0, i);
    }
  } else if (src.Ndim() == 2) {
    // TODO
  }  else {
    // src.Ndim() is 1 or 0.
    // TODO
  }

}




}  // namespace k2
