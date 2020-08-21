// k2/csrc/cuda/algorithms.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_ALGORITHMS_H_
#define K2_CSRC_CUDA_ALGORITHMS_H_

#include "k2/csrc/cuda/array.h"

//  this really contains various utilities that are useful for k2 algorithms.
namespace k2 {

class Renumbering {
 public:
  Renumbering(int32_t num_old_elems);

  int32_t NumOldElems();
  int32_t NumNewElems();

  Array1<char> &Kept();

  Array1<int32_t> &New2Old();  // dim is NumNewElems()

  Array1<int32_t> &Old2New();  // dim is NumOldElems()

 private:
  Array1<char> kept;
  Array1<int32_t> new2old;
  Array1<int32_t> old2new;

};



}  // namespace k2

#endif  // K2_CSRC_CUDA_ALGORITHMS_H_
