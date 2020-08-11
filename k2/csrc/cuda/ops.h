// k2/csrc/cuda/ops.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_OPS_H_
#define K2_CSRC_CUDA_OPS_H_

#include "k2/csrc/cuda/array.h"



/*
  Transpose a matrix.  Require src.Size0() == dest.Size1() and src.Size1() ==
  dest.Size0().
 */
template <typename T> void Transpose(Array2<T> &src,
                                     Array2<T> &dest);


/*
  Sets 'dest' to exclusive prefix sum of 'src'.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data.  Must satisfy dest.Size() == src.Size() or
                       dest.Size() == src.Size() + 1.  At exit will satisfy
                       dest[i] == sum_{j=0}^{i-1} src[j].  Must be on same device
                       as src.
 */
template <typename S, typename T> void ExclusiveSum(Array1<S> &src,
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
template <typename T> void ExclusiveSum(Array2<T> &src,
                                        Array2<T> *dest,
                                        int axis);



#endif  // K2_CSRC_CUDA_OPS_H_
