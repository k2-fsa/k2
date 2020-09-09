/**
 * @brief
 * ragged_shape
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/ragged.h"

namespace k2 {


int32_t RaggedShape2::TotSize1() {
  if (cached_tot_size1_ >= 0) {
    return cached_tot_size1_;
  }
  // We set cached_tot_size1_ to -1 if we're not sure what its value is.
  Context *c = row_splits1_.Context().get();
  int32_t size = row_splits1_.Size();
  if (size == 0) {
    cached_tot_size1_ = 0;
  } else {
    // Most of the complexity is in the operator [], which handles stream
    // synchronization and GPU->CPU transfer, if this was on the GPU.
    cached_tot_size1_ = row_splits1_[size-1];
  }
}


RaggedShape3 MergeToAxis1(const std::vector<const RaggedShape2*> &src) {
}


RaggedShape4 MergeToAxis1(const std::vector<const RaggedShape3*> &src) {
  assert(src.size() != 0);
  Context c = src[0]->Context();

  RaggedShape3 temp = MergeToAxis1(reinterpret_cast<const std::vector<const RaggedShape2*> &>(src));

  Array1<int32_t*> src_row_splits1(c_cpu, n), src_row_splits2(c_cpu, n);
  // init arrays, convert to device tensors.

  int32_t **src_row_splits1_data = src_row_splits1.Data(),
      **src_row_splits2_data = src_row_splits2.Data();

  const int32_t *row_splits1_data = temp.RowSplits1().Data(),
      *row_ids1 = temp.RowIds1().Data(),
      *row_splits2 = temp.RowSplits2().Data(),
      *row_ids2 = temp.RowIds2().Data();

  // row_splits3_out_01 will be indexed by an idx01 of the output array, but actually
  // contains row-indexes on axis 3 (rather than on axis 2, which
  // something indexed by an idx01 normally would).  So it's like
  // idx01xx = row_splits3_out_01[idx01].
  // We want to do the
  // exclusive-sum on the smallest size we can, and are using the fact that
  // within given indexes 0,1 of the output, the input and output layouts are
  // the same.
  Array1<int32_t> row_splits3_out_01(temp.TotSize1() + 1);
  // sorry this naming is weird, but they are the sizes on axis 3, but the
  // row-splits for axis 2 (i.e. indexed by an idx012).
  int32_t *sizes3_out_data = row_splits3_out_01.Data();

  __host__ __device__ lambda_set_sizes3 = [=] (int32_t output_idx01) -> void {
     int32_t output_idx0 = row_ids1_data[output_idx01],
         output_idx0x = row_splits0_data[output_idx0],
         output_idx1 = output_idx01 - output_idx0x;

     int32_t *input_row_splits1 = src_row_splits1_data[output_idx1],
         *input_row_splits2 = src_row_splits2_data[output_idx1];

     int32_t input_idx0xx = input_row_splits2[input_row_splits1[output_idx0]],
         input_idx0xx_next = input_row_splits2[input_row_splits1[output_idx0 + 1]],
         input_size0xx = input_idx0xx_next - input_idx0xx;
     // input_size0xx is the total size on axis 2, but spanning one element on
     // axis 0, including whatever elements there are on axis1.  So the
     // num-elements in a list of lists.

     // Index 1 in the output is known as well; it corresponds to the index into 'src'.
     // So this is the total size on axis 3 of the output, given values on axes 0 and
     // 1 (but not 2; span all) of the input.
     int32_t output_size01xx = input_size0xx;
     // the size on axis 2 of the input becomes the size on axis 3 of the
     // output.
     sizes3_out_data[output_idx01] = output_size01xx;
  };
  Eval(c, temp.TotSize1(), lambda_set_sizes3);
  ExclusiveSum(row_splits3_out_01, &row_splits3_out_01);
  // the entries of row_splits3_out_01 would be written as output_idx01x,
  // i.e. they have the magnitude of an output_idx012 but where the last index
  // (axis 2) is always zero so we write x.
  int32_t *row_splits3_out_01 = row_splits3_out_01.Data();

  int32_t row_splits3_out_size = temp.TotSize2() + 1;
  Array1<int32_t> row_splits3_out(c, row_splits3_out_size);

  int32_t *row_splits3_out_data = row_splits3_out.Data();

  __host__ __device__ lambda_set_row_splits3 = [=] (int output_idx012) -> void {
     // 'offset' below is to avoid reading invalid memory, one past the end of
     // row_ids2_data; it'll still do the right thing.
     int32_t offset = (output_idx012 == row_splits3_size-1 ? -1 : 0);

     int32_t output_idx01 = row_ids2_data[output_idx012 + offset],
         output_idx0 = row_ids1_data[output_idx01],
         output_idx0x = row_splits1_data[output_idx0],
         output_idx1 = output_idx01 - output_idx0x,
         output_idx01x = row_splits2_data[output_idx01],
         output_idx2 = output_idx012 - output_idx01x;

     // Note: axis 0 of the output corresponds to axis 0 of the input, but axes
     // 2 and 3 of the output correspond to axes 1 and 2 of the input
     // respectively (axis 1 of the output corresponds to the index into 'src').
     int32_t *input_row_splits1 = src_row_splits1_data[output_idx1],
         *input_row_splits2 = src_row_splits2_data[output_idx1];

     int32_t input_idx_0x = input_row_splits0[output_idx0],
         input_idx01 = input_idx0x + output_idx2,
         input_idx01x = input_row_splits2[input_idx01],
         input_idx0xx = input_row_splits2[input_idx0x],
         input_idxx1x = input_idx01x - input_idx0xx;

     // index 1 of input becomes index 2 of the output.  the extra "x" is because
     // we inserted output_idx1.  Index  xx2x means it's dimensionally an index into
     // the elems of a Ragged4 array (includes 4 indexes) but it's a difference of
     // such things, namely like an 012x minus an 01xx.
     int32_t output_idxxx2x = input_idxx1x;

     int32_t output_idx01xx = row_splits3_out_data[output_idx01],
         output_idx012x = output_idx01xx + output_idxxx2x;

     row_splits3_out_data[output_idx012] = output_idx012x;
   };
  Eval(c, row_splits3_out_size, lambda_set_row_splits3);
}



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
    auto lambda_get_sizes = [=] __host__ __device__ (int32_t j) -> void {
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
  // osizes2_t and tot_osizes3_t are contiguous, so osizes2_t and tot_osizes3_t
  // will share the underlying memory.
  Array1<int32_t> osizes2_t_linear = osizes2_t.Flatten(),
      tot_osizes3_t_linear = osizes3_t.Flatten();
  Array1<int32_t> osizes1(c, osize0, osize1);

  Array1<int32_t> row_splits2_linear(size0*size1 + 1),
      row_splits3_linear(size0*size1 + 1);
  ExclusiveSum(c.Child(), osizes2_t_linear, &row_splits2_linear);
  ExclusiveSum(c.Child(), tot_osizes3_t_linear, &row_splits3_linear);

  // row_splits1_linear has size `size0`.
  Array1<int32_t> row_splits1_linear(
      row_splits2_linear.ToTensor().Range(0, size0, size1));

  c.WaitForChildren();

  return RaggedShape4FromRowSplits(row_splits1_linear,
                                   row_splits2_linear,
                                   row_splits3_linear);

}




}



}  // namespace k2
