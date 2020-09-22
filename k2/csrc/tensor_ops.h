// k2/csrc/cuda/tensor_ops.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_TENSOR_OPS_H_
#define K2_CSRC_TENSOR_OPS_H_

#include "k2/csrc/tensor.h"

namespace k2 {

/*
  This copies elements from `src` to `dest`.  They must be on the same device,
  and have the same dims and dtype, but not necessarily the same layout.  (In
  fact, if they have the same layout and are contiguous, there are faster ways
  to copy).
*/
void CopyTensorElements(Tensor src, Tensor dest);

/*
  Returns a contiguous version of `src`, i.e. with the same dims but that
  satisfies IsContiguous().  This will share the underlying data with
  `src` if `src` was already contiguous.    Internally calls
  `CopyTensorElements()`
*/
Tensor ToContiguous(const Tensor &src);

/*
  Cast tensor elements to a new dtype.  Only works if `src` was contiguous; will
  copy if that was not the case.

  If `new_dtype` was the same as the dtype of `src`, this may return a Tensor
  that uses the same underlying data as `src`.
 */
Tensor Cast(Tensor src, Dtype new_dtype);

}  // namespace k2

#endif  // K2_CSRC_TENSOR_OPS_H_
