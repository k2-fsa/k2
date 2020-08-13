// k2/csrc/cuda/ragged_shape.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/cuda/ragged_shape.h"

namespace k2 {

// Caution: this is really a .cu file.  It contains mixed host and device code.

RaggedShape4 MergeToAxis1(const std::vector<const RaggedShape3*> &src) {
  assert(src.size() != 0);
  Context c = src[0]->Context();
  int32_t osize0 = src[0]->Size0(),
      osize1 = src.size();   // note, osize1 is not the same as its TotSize1(),
                             // it is the size of each sub-list.
  // TODO: assert src[n]->Size0() == size0 for all n.

  // tot_size2 and tot_size3 are the total sizes on axes 2 and 3 of the
  // result, corresponding to the totals on axes 1 and 2 of src.
  int32_t tot_size2 = 0, tot_size3 = 0;
  for (int32_t i = 0; i < size1; i++) {
    tot_size2 += src[i]->TotSize1();
    tot_size3 += src[i]->TotSize2();
  }

  // we will transpose these arrays later; it's better for consolidated writes
  // to have it this way around initially.  TODO: need to make sure we can read
  // one past the end.
  // TODO: ensure that memory one past the end is readable.  May be easiest to
  // just ensure that this is the case always, in the constructor of Array2.
  Array2<int32_t> osizes2(c, osize1, osize0),
      tot_osizes3(c, osize1, osize0);
  // Note, we rely on the fact that a freshly-initialized Array2 will be contiguous, so
  // we can index an array x with size (A,B) as x.data()[a*B + b]
  int32_t *osizes2_data = osizes2.data(),
      *tot_osizes3_data = tot_osizes3.data();
  for (int32_t i = 0; i < osize1; i++) {
    RaggedShape3 &shape = *(src[i]);
    int32_t *row_splits1_data = shape.RowSplits1(),
        *row_splits2_data = shape.RowSplits2();
    // the j below is the top-level index0 into the shape; we want the total
    // number of index1's and index2's/elements associated with this index0,
    // which will be written to respectively sizes2_data and sizes3_data.
    auto lambda_get_sizes = __host__ __device__ [=] (int32_t j) -> void {
        int32_t begin1 = row_splits1_data[j], end1 = row_splits1_data[j+1];
        int32_t begin2 = row_splits2_data[begin1], end2 = row_splits1_data[end1];
        int32_t num_indexes1 = end1 - begin1,
            num_indexes2 = end2 - begin2;
        tot_osizes2_data[i * osize0 + j] = num_indexes1;
        tot_osizes3_data[i * osize0 + j] = num_indexes2;
    };
    Eval(c.Child(), osize0, lambda_get_sizes);
  }
  c.WaitForChildren();

  // Transpose those total sizes, so the indexes are the right way
  // around, i.e. [index0,index1].  Note: conversion to Array2 will
  // ensure the result is contiguous.
  Array2<int32_t> osizes2_t(osizes2.ToTensor().Transpose(0, 1)),
      tot_osizes3_t(osizes2.ToTensor().Transpose(0, 1));
  // osizes2_t and tot_osizes3_t are contiguous, so these will share
  // the underlying memory.
  Array1<int32_t> osizes2_t_linear = osizes2_t.Flatten(),
      tot_osizes3_t_linear = osizes3_t.Flatten();

  Array1<int32_t> offsets2_linear(size0*size1 + 1),
      offsets3_linear(size0*size1 + 1);
  ExclusiveSum(c.Child(), osizes2_t_linear, &offsets2_linear);
  ExclusiveSum(c.Child(), tot_osizes3_t_linear, &offsets3_linear);

  // Given idx0,idx1, interpreted as indexes into the returned value,
  // we'll index offsets2_linear and offsets3_linear as
  // data[idx0 * size1 + idx1].


  c.WaitForChildren();
}



}  // namespace k2
