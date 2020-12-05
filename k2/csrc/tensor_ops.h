// k2/csrc/cuda/tensor_ops.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_TENSOR_OPS_H_
#define K2_CSRC_TENSOR_OPS_H_

#include "k2/csrc/array.h"
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

/*
  Returns a Tensor that is a result of indexing `src` along its
  axis 0 with `indexes`, as if you had done src[indexes] in Pytorch.

     @param [in] src  Source tensor, to be indexed. Currently, it
                      supports only 1-D and 2-D tensors. If the
                      tensor is 2-D, it requires that its Stride(1) is 1.
     @param [in] indexes   Indexes to use; if allow_minus_one == false,
                     must satisfy 0 <= indexes[i] < src.Dim(0);
                     if allow_minus_one == true, -1 is also allowed
                     and the corresponding output values will be
                     set to zero.
     @param [in] allow_minus_one  If true, -1 is allowed as a member
                     of `indexes` and the corresponding output elements
                     will be zero.
     @return   Returns a Tensor with the same dtype as `src`, and
                     shape (indexes.Dim(), src.Dim(1), src.Dim(2), ...),
                     i.e. with the same num-axes as `src` and
                     axis 0's dim replaced with indexes.Dim().
 */
Tensor Index(Tensor &src, Array1<int32_t> &indexes,
             bool allow_minus_one = true);

/*
  IndexAdd() is the function you would use when doing backprop for Index().
  (Note: this is the non-in-place version, see also the other version of this
  function).
        @param [in] src   Source of elements to add (in backprop, this
                    would correspond to the derivative w.r.t. the output).
        @param [in] indexes  Indexes with `indexes.Dim() == src.Dim(0)`.
                    If allow_minus_one == false, these must
                    satisfy 0 <= indexes[i] < dim; if allow_minus_one == true,
                    -1 is also allowed (and nothing is done for those indexes).
        @param [in] dim  Will become the 1st dim of the result, i.e.
                    ans.Dim(0).
        @param [in] If true, -1 is allowed as an index; if false, that is
                    an error.

        @return  New tensor with shape `(dim, src.Dim(1), src.Dim(2), ...)`,
                 i.e. with same NumAxes() as src and ans.Dim(0) == dim.
                 Elements will be as if we first initialized it to zero
                 and then, for each tuple of indexes `(i,j,k..)` into
                 `src` where i != -1,
                 did: `ans[indexes[i],j,k,..] += src[i,j,k..]`.
 */
Tensor IndexAdd(Tensor &src, Array1<int32_t> &indexes, int32_t dim,
                bool allow_minus_one = true);

/*
  Version of IndexAdd() that does not allocate the tensor, but expects it
  to already be allocated (and set to zero, if appropriate).
           @param [in] src  Source tensor whose elements are to be added to
                            `dest`. It supports only 1-D and 2-D tensors.
           @param [in] indexes  Indexes with `indexes.Dim() == src.Dim(0)`.
                      If allow_minus_one == false, these must
                      satisfy 0 <= indexes[i] < dim; if allow_minus_one == true,
                      -1 is also allowed (and nothing is done for those
                      indexes).
           @param [in] allow_minus_one  If true, we allow -1 as an index,
                      and nothing is added for the corresponding elements.
           @param [out] dest  At exit it will be as if we had executed,
                      for each tuple of indexes `i,j,k..` into `src`,
                      `(*dest)[indexes[i],j,k..] += src[i,j,k]`
                      (if `indexes[i] != -1`).
 */
void IndexAdd(Tensor &src, Array1<int32_t> &indexes, bool allow_minus_one,
              Tensor *dest);

}  // namespace k2

#endif  // K2_CSRC_TENSOR_OPS_H_
