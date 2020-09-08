// k2/csrc/cuda/fsa.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors


#include "k2/csrc/fsa.h"

namespace k2 {

Fsa FsaFromTensor(const Tensor &t, bool *error) {
  *error = true;
  if (t.Dtype() != kInt32Dtype) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, wrong dtype, got "
                    << TraitsOf(t.Dtype()).Name() << " but expected "
                    << TraitsOf(kInt32Dtyt.Dtype()).Name();

}

}  // namespace k2
