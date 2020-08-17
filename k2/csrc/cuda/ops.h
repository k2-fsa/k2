// k2/csrc/cuda/ops.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_OPS_H_
#define K2_CSRC_CUDA_OPS_H_

#include "k2/csrc/cuda/array.h"


// Note, I'm not sure about the name of this file, they are not ops like in TensorFlow, but
// procedures..

/*
  Transpose a matrix.  Require src.Size0() == dest.Size1() and src.Size1() ==
  dest.Size0().  This is not the only way to transpose a matrix, you can also
  do: dest = Array2<T>(src.ToTensor().Transpose(0,1)), which will likely call this
  function

     @param [in] c   Context to use, must satisfy `c.IsCompatible(src.Context())` and
                    `c.IsCompatible(dest->Context())`.
     @param [in] src  Source array to transpose
     @param [out] dest  Destination array; must satisfy `dest->Size1() == src.Size0()`
                     and `dest->Size0() == src.Size1()`.  At exit, we'll have
                     dest[i,j] == src[j,i].
 */
template <typename T> void Transpose(ContextPtr &c,
                                     Array2<T> &src,
                                     Array2<T> *dest);


/*
  Sets 'dest' to exclusive prefix sum of 'src'.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data.  Must satisfy dest.Size() == src.Size() or
                       dest.Size() == src.Size() + 1, but in the latter case we
                       require that the memory region inside src be allocated with
                       at least one extra element, because the exclusive-sum code
                       may read from it even though it doesn't affect the result.

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j].  Must
                       be on same device as src.
 */
template <typename S, typename T> void ExclusiveSum(ContextPtr &c,
                                                    Array1<S> &src,
                                                    Array1<T> *dest);


/*
  Sets 'dest' to exclusive prefix sum of 'src', along a specified axis.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data; allowed to be the same as src.
                       For axis==1, for example, at exit it will satisfy
                       dest[i][j] == sum_{k=0}^{j-1} src[i][k].
                       Must have the same size on the other axis; on the axis being
                       summed, must be either the same size as src, or one greater.
                       as src.
    @param [in] axis   Determines in what direction we sum, e.g. axis = 0 means summation
                       is over row axis (slower because we have to transpose), axis = 1
                       means summation is over column axis.
 */
template <typename T> void ExclusiveSum(ContextPtr &c,
                                        Array2<T> &src,
                                        Array2<T> *dest,
                                        int axis);



#endif  // K2_CSRC_CUDA_OPS_H_
