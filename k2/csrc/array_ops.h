/**
 * @brief
 * array_ops
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu
 *                                                   Meixu Song)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_OPS_H_
#define K2_CSRC_ARRAY_OPS_H_

#include <cassert>
#include <cstring>
#include <type_traits>

#include "k2/csrc/array.h"
#include "k2/csrc/context.cuh"
#include "k2/csrc/cuda_headers.cuh"
#include "k2/csrc/log.cuh"
#include "k2/csrc/ragged.h"

// Note, I'm not sure about the name of this file, they are not ops like in
// TensorFlow, but procedures..

namespace {
// TODO(haowen): manage/load block config with some classes? then we can get
// different configuration depending on num_elements and data type.
// block size for matrix transpose.
constexpr int32_t kTransTileDim = 32;
constexpr int32_t kTransBlockRows = 8;
}  // namespace

namespace k2 {
// TODO(haowen): move the implementations to file `op_inl.h` or
// `op.cu`(specialized on device and data type)?
template <typename T>
__global__ void TransposeKernel(int32_t rows, int32_t cols, const T *input,
                                T *output) {
  // TODO(haowen): here we need to handle different type of T to avoid bank
  // conflicts, the size of cache now is fine for type size with 32bit (e.g.
  // int32 or float).
  __shared__ T cache[kTransTileDim][kTransTileDim + 1];

  // input index, in a coalesced manner.
  int32_t x = threadIdx.x + blockIdx.x * kTransTileDim;
  int32_t y = threadIdx.y + blockIdx.y * kTransTileDim;

  for (int32_t i = 0; i < kTransTileDim; i += kTransBlockRows) {
    if (x < cols && (y + i) < rows) {
      cache[threadIdx.y + i][threadIdx.x] = input[(y + i) * cols + x];
    }
  }

  __syncthreads();

  // output index, in a coalesced manner
  x = threadIdx.x + blockIdx.y * kTransTileDim;
  y = threadIdx.y + blockIdx.x * kTransTileDim;
  for (int32_t i = 0; i < kTransTileDim; i += kTransBlockRows) {
    if (x < rows && (y + i) < cols) {
      output[(y + i) * rows + x] = cache[threadIdx.x][threadIdx.y + i];
    }
  }
}

/*
  Transpose a matrix.  Require src.Size0() == dest.Size1() and src.Size1() ==
  dest.Size0().  This is not the only way to transpose a matrix, you can also
  do: dest = Array2<T>(src.ToTensor().Transpose(0,1)), which will likely call
  this function

     @param [in] c   Context to use, must satisfy
                     `c.IsCompatible(src.Context())` and
                     `c.IsCompatible(dest->Context())`.
     @param [in] src  Source array to transpose
     @param [out] dest  Destination array; must satisfy
                        `dest->Size1() == src.Size0()` and
                        `dest->Size0() == src.Size1()`.
                        At exit, we'll have dest[i,j] == src[j,i].
*/
template <typename T>
void Transpose(ContextPtr &c, const Array2<T> &src, Array2<T> *dest) {
  assert(c->IsCompatible(*src.Context()));
  assert(c->IsCompatible(*dest->Context()));
  int32_t rows = src.Dim0();
  int32_t cols = src.Dim1();
  // TODO(haowen): limit the number of elements?
  assert(rows == dest->Dim1());
  assert(cols == dest->Dim0());
  const T *src_data = src.Data();
  T *dest_data = dest->Data();
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
  if (d == kCpu) {
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        dest_data[i * rows + j] = src_data[j * cols + i];
      }
    }
  } else {
    assert(d == kCuda);
    dim3 block_size(kTransTileDim, kTransBlockRows, 1);
    dim3 grid_size(NumBlocks(cols, kTransTileDim),
                   NumBlocks(rows, kTransTileDim));
    TransposeKernel<<<grid_size, block_size, 0, c->GetCudaStream()>>>(
        rows, cols, src_data, dest_data);
    auto ret = cudaDeviceSynchronize();
    K2_CHECK_CUDA_ERROR(ret);
  }
}

/*
  Sets 'dest' to exclusive prefix sum of 'src'.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data (possibly &src).  Must satisfy
                       dest.Size() == src.Size() or dest.Size() == src.Size() +
  1,
                       but in the latter case
                       we require that the memory region inside src be allocated
                       with at least one extra element, because the
                       exclusive-sum code may read from it even though it
                       doesn't affect the result.

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j].
                       Must be on same device as src.
 */
template <typename S, typename T>
void ExclusiveSum(Array1<S> &src, Array1<T> *dest);

/*  wrapper for the ExclusiveSum above.  Will satisfy
     ans[i] = sum_{k=0}^{i-1} src[i].
 */
template <typename T>
void ExclusiveSum(Array1<T> &src);


/*
  Sets 'dest' to exclusive prefix sum of the result of dereferinging the
  elements of 'src'.
    @param [in] src    Source data, to be dereferenced and then summed.
    @param [out] dest  Destination data.  Must satisfy dest.Size() == src.Size()
                       or dest.Size() == src.Size() + 1, but in the latter case
                       we require that the memory region inside src be allocated
                       with at least one extra element, because the
                       exclusive-sum code may read from it even though it
                       doesn't affect the result.

                       At exit, will satisfy dest[i] == sum_{j=0}^{i-1} src[j].
                       Must be on same device as src.
 */
template <typename T>
void ExclusiveSumDeref(Array1<T *> &src, Array1<T> *dest);

/*
  Sets 'dest' to exclusive prefix sum of 'src', along a specified axis.
    @param [in] src    Source data, to be summed.
    @param [out] dest  Destination data; allowed to be the same as src.
                       For axis==1, for example, at exit it will satisfy
                       dest[i][j] == sum_{k=0}^{j-1} src[i][k].
                       Must have the same size on the other axis; on the axis
                       being summed, must be either the same size as src,
                       or one greater. as src.
    @param [in] axis   Determines in what direction we sum, e.g. axis = 0 means
                       summation is over row axis (slower because we have to
                       transpose), axis = 1 means summation is over column axis.
 */
template <typename T>
void ExclusiveSum(ContextPtr &c,
     Array2<T> &src, Array2<T> *dest, int32_t axis);

template <typename T>
void ExclusiveSum(Array2<T> &src, Array2<T> *dest);

/*
  Append a list of Array1<T> to create a longer array.

  For now we can just use a simple loop; later there are lots of opportunities
  to optimize this, including multiple streams and using a single kernel making
  use of RaggedShape.
      @param [in] src_size  Number of arrays to append.  Must be > 0.
      @param [in] src     Array of pointers to arrays, of size `src_size`.
      @return       Returns the appended array
 */
template <typename T>
Array1<T> Append(int32_t src_size, const Array1<T> **src);

/*
  Transpose a matrix.  Require src.Size0() == dest.Size1() and src.Size1() ==
  dest.Size0().  This is not the only way to transpose a matrix, you can also
  do: dest = Array2<T>(src.ToTensor().Transpose(0,1)), which will likely call
  this function

     @param [in] c   Context to use, must satisfy
                     `c.IsCompatible(src.Context())` and
                     `c.IsCompatible(dest->Context())`.
     @param [in] src  Source array to transpose
     @param [out] dest  Destination array; must satisfy
                        `dest->Size1() == src.Size0()` and
                        `dest->Size0() == src.Size1()`.
                        At exit, we'll have dest[i,j] == src[j,i].
*/
template <typename T>
void Transpose(ContextPtr &c, const Array2<T> &src, Array2<T> *dest);

/*
   This is a little like Append(), but with special treatment of the last
   elements (it's intended for use with row_splits and row_ids vectors, which
   have a single "extra" last element).

   It appends the arrays with an offset.  Define:
        offset[i] = (sum of last element of src[j] for j < i).
   This function appends the arrays, while leaving out the last element
   of all but the last of the arrays in `src`, and also adding the
   offsets mentioned above for each array.

   arrays in 'src', except they all overlap by one element, and for i > 0
   we add an offset o[i] to the arrays in src[i], with the offsets being
   chosen so that in the overlapping elements there is no conflict between
   the two values being written (this means that src[i] is the sum of
   the final element of each of the arrays in src[j] for j < i).

      @param [in] src_size  Number of arrays to append
      @param [in] src     Array of pointers to arrays, of size `src_size`.
      @return       Returns the appended array

 */
Array1<int32_t> Splice(int32_t src_size, Array1<int32_t> **src);

/*
  Output to an array `max_values` the maximum of each sub-list in `src`
  i.e. the max taken over axis 1), or `default_value`, whichever was larger.

     @param [in] src            Input ragged array; must have src.NumAxes()
                                 == 2. Is allowed to be empty.
     @param [in] default_value  Value to use for maximum operation as a default
                                so max is taken over this and the elements
                                of sub-lists in `src`.
     @param [out] max_values    Array to which the maximum values will be
                                written. Must satisfy max_values->Dim() == src.
 */
template <typename T>
void MaxPerSublist(Ragged<T> &src, T default_value, Array1<T> *max_values);

/*
  Get the maximum value from the array `src`, or `default_value`, whichever is
  greater.
         @param [in] src   Array to find the '&'-based reduction of
         @param [in] default_value   Value to initialize the reduction with, and
                                     to use if src is empty.  Would typically be
                                     the most negative T possible.
         @param [out] dest  Output array, which must have dim == 1.
 */
template <typename T>
void Max(Array1<T> &src, T default_value, Array1<T> *dest);

/*
  Get the '&'-based (bitwise and) reduction of the array `src`, using
  `default_value` (e.g. all ones) to initialize the reduction.

         @param [in] src   Array to find the '&maximum of
         @param [in] default_value   Value to initialize the reduction with, and
                                     to use if src is empty.  Would typically be
                                     the most negative T possible.
         @param [out] dest  Output array, which must have dim == 1.
                            Note: it is allowable for the output array
                            to be an element of `src`.
 */
template <typename T>
void And(Array1<T> &src, T default_value, Array1<T> *dest);

/*
  Output to an array `and_values` the result of reducing each sub-list in
  `src` with operator &, i.e. bit-wise and.

     @param [in] src            Input ragged array; must have src.NumAxes()
                                == 2. Is allowed to be empty.
     @param [in] default_value  Value to initialize the reduction with; should
                                probably be all-ones.
     @param [out] and_values    Array to which the bitwise-and values will be
                                written. Must satisfy max_values->Dim() == src.

   TODO: implement this after debugging MaxPerSublist; it's mostly a matter of
   changing the reduction-operator object.
*/
template <typename T>
void AndPerSublist(Ragged<T> &src, T default_value, Array1<T> *and_values);

/*
  Returns a random Array1, uniformly distributed betwen `min_value` and
  `max_value`.  CAUTION: this will be randomly generated on the CPU, for now,
  and transferred to the CPU, so it will be slow if c is not a CPU context.

    @param[in] c  Context for this array; note, this function will be slow
                  if this is not a CPU context
    @param [in] dim    Dimension
    @param[in] min_value  Minimum value allowed in the array
    @param[in] max_value  Maximum value allowed in the array;
                           require max_value >= min_value.
    @return    Returns the randomly generated array

 */
template <typename T>
Array1<T> RandUniformArray1(ContextPtr &c, int32_t dim, T min_value, T max_value);

/*
  Return a newly allocated Array1 whose values form a linear sequence,
   so ans[i] = first_value + i * inc.
*/
template <typename T>
Array1<T> Range(ContextPtr &c, int32_t dim, T first_value, T inc = 1);


/*
   Validate a row_ids vector; this just makes sure its elements are nonnegative
   and non-decreasing.

     @param [in] row_ids  row_ids to validate
     @param [in] temp     The user may supply a nonempty array on the same device (or host)
                          as `row_ids` that can be used temporarily (just the first
                          element is needed).  This saves an allocation.
     @return   Returns true if `row_ids` is a plausible row_ids vector.
*/
bool ValidateRowIds(Array1<int32_t> &row_ids, Array1<int32_t> *temp = nullptr);

/*
   Validate a row_splits vector; this just makes sure its elements are
   non-decreasing, its dimension is at least 1 and row_splits[0] == 0.

     @param [in] row_splits  row_splits to validate
     @param [in] temp     The user may supply a nonempty array on the same device (or host)
                          as `row_splits` that can be used temporarily (just the first
                          element is needed).  This saves an allocation.
     @return   Returns true if `row_splits` is a plausible row_splits vector.
*/
bool ValidateRowSplits(Array1<int32_t> &row_splits, Array1<int32_t> *temp = nullptr);


/*
  Jointly validate row_splits and row_ids vectors, making sure they are
  plausible and consistent with each other.

     @param [in] row_splits  row_splits to validate
     @param [in] row_ids     row_ids to validate
     @param [in] temp     The user may supply a nonempty array on the same device (or host)
                          as `row_splits` that can be used temporarily (just the first
                          element is needed).  This saves an allocation.
     @return   Returns true if the vectors are plausible and agree with each other.
*/
bool ValidateRowSplitsAndIds(Array1<int32_t> &row_splits,
                             Array1<int32_t> &row_ids,
                             Array1<int32_t> *temp = nullptr);


}  // namespace k2

#define IS_IN_K2_CSRC_ARRAY_OPS_H_
#include "k2/csrc/array_ops_inl.h"
#undef IS_IN_K2_CSRC_ARRAY_OPS_H_

#endif  // K2_CSRC_ARRAY_OPS_H_
