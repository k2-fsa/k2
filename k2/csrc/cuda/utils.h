// k2/csrc/cuda/utils.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_UTILS_H_
#define K2_CSRC_CUDA_UTILS_H_

#include "k2/csrc/cuda/context.h"

#define IS_IN_K2_CSRC_CUDA_UTILS_H_
#include "k2/csrc/cuda/utils_inl.h"
#undef IS_IN_K2_CSRC_CUDA_UTILS_H_

namespace k2 {

// Some quite low-level utilities.
// CAUTION: this is not up to date, I will simplify this probably.



/*
  sizes concept

  A vector of sizes is a vector of non-negative integers, like [ 5 4 0 1 3 ].
  This just establishes a naming convention.  In most cases the 'sizes' will
  be the lengths of sub-lists in a list of lists.

  Relation to other concepts:
   A vector of sizes can be seen as the difference of successive elements of
  a vector of row_splits, i.e. sizes[i] = row_splits[i+1] - row_splits[i].
*/


/*
  row_splits concept / row-splits concept
  (Note, this has been named for compatibility with TensorFlow's RaggedTensor).

  An row_splits vector is a vector of the form, say, [ 0 5 9 9 10 13 ].
  i.e. it starts with 0 and is non-decreasing.  It will often be encountered
  as the exclusive-sum of a vector of 'sizes' (see 'sizes concept' above),
  with a size one greater than that of the corresponding 'sizes'.  It
  will represent the positions in a single linearized list, of where we
  put the elements of a list of sub-lists.  So in the example above,
  sub-list 0 occupies positions 0,1,2,3,4, sub-list 1 occupies positions 5,6,7,8,
  and so on.  Caution: the number of elements of the row_splits vector equals the number
  of sub-lists PLUS ONE.

  Relation to other concepts:
    See 'row_ids concept' where its relation to 'row_splits' is described.
    If row_splits = [ 0 2 3 5 ], the corresponding 'row_ids' is:
    row_ids = [ 0 0 1 2 2 ].
*/

/*
  row_ids concept / row-ids concept
  (Note, this has been named for compatibility with TensorFlow's RaggedTensor).


  A vector of row_ids is a vector of the form
    [ 0 0 0 1 1 2 2 2 ]
  or in the general case any nonnegative, non-decreasing list of integers.  Each
  group represents the index of the sub-list to which that position belongs; for
  instance, if we had a list-of-lists like [ a b c ] [ d e ] [ f g h ], the
  above vector of row_ids would describe its structure.

  Relation to other concepts:
    A vector of row_ids can arise as the cumulative sum of a vector of tails.
    A vector of row_ids and a vector of row_splits represent the same information
     in different ways, satisfying row_splits[row_ids[i]] <= i < row_splits[row_ids[i] + 1]
     and row_splits[row_splits.size() - 1] == row_ids.size().

*/


/*
  tails concept

  A vector of tails is a vector containing zeros and ones; each '1' represents
  the last element of a sub-list.  For example [ 0 1 0 0 0 1 1 ] to represent
  a list of sub-lists like [ x x ] [ y y y y ] [ z ].  The last element will always
  be 1.

  Relation to other concepts:
    The exclusive cumulative sum of a vector of tails is a vector of row_ids.
  E.g. the above example, with exclusive cumulative sum, is:
    [ 0 0 1 1 1 1 2 ].
 */

/*

  Index naming scheme

  In a ragged tensor with n axes (say, 3) the actual elements will be written
  in a linear array we'll have various levels of indexes that allow us to
  look up an element given the hierarchical indexes and vice versa.  A 3-d
  ragged tensor will have RowIds1(), RowSplits1(), RowIds2(), RowSplits2(),
  and the actual elements.  We have a naming scheme that expresses what information
  is packed into a single integer.

  Some entry-level facts about the naming scheme are:

     - The hierarchical indexes into the tensor (3 of them for a tensor with 3
       axes), we call idx0, idx1 and idx2
     - The linear index into the elements, we call idx012 because it includes
       all 3 values.
     - The RowSplits1() would map from an idx0 to an idx0x.  The x here
       takes the place of a 1 and that replacement means "actually the index
       here is definitely zero".  Any specific idx0x that we have will be
       for a particular idx0.

   For more details, it's best to use an example.


     # below is pseudocode
     RaggedTensor3 t = [ [ [ 1 2 ] [ 5 ] ] [ [ 7 8 9 ] ] ]

     # which will give us:
     t.row_splits1 == [ 0 2 3 ]    # indexed by idx0, elements are idx0x
     t.row_ids1 == [ 0 0 1 ]       # indexed by idx01, elements are idx0
     t.row_splits2 == [ 0 2 3 6 ]  # indexed by idx01, elements are idx01x
     t.row_ids2 == [ 0 0 1 2 2 2 ] # indexed by idx012, elements are idx01
     t.values == [ 1 2 5 7 8 9 ]   # indexed by idx012, elements are whatever
                                   # values we're storing.

     Sometimes we'll want to know the number of elements in sub-lists, and we
     have a notation for the computations involved in that.  Suppose we want to
     know the number of elements in T[0].  We'll compute:
       int idx0 = 0,
           idx0x = t.row_splits1[idx0],
           idx0x_next = t.row_splits1[idx0 + 1],
           idx0xx = t.row_splits2[idx0],
           idx0xx_next = t.row_splits2[idx0x_next],
           size_0xx = idx0xx_next - idx0xx
     (The _next suffix is used when we're querying the most specific known index
     plus one, in this case index 0 but for instance, idx01x_next would mean
     that we were querying idx01x after incrementing the index on axis 1.)

     We also might sometimes want to know an offset of an element within the
     part of the array that starts with a particular prefix of that index.
     E.g. suppose we want the offset of element t[ idx0, idx1, idx2 ]
     relative to the start of the sub-array t[idx0].  We'd do this as
     follows:
        int idx0, idx1, idx2;  # provided
        int idx0x = t.row_splits1[idx0],
            idx01 = idx0x + idx1,
            idx01x = t.row_splits2[idx1],
            idx012 = idx01x + idx2,
            idx0xx = t.row_splits2[idx0x],
            idx12 = idx012 - idx0xx;
     In the last line above, when we subtract idx012 - idx0xx we lose the
     leading "0" because the zeroth index was the same in the two things being
     subtracted.  Note: in an expression like idx0xx_next - idx0xx we don't get
     idxxxx because index zero is *different*.  However, the result of
     idx01x_next - idx01x would be written idx1x because index zero would be
     the same.

     The advantage of this naming scheme is that the 'type' that operations give
     is intuitively obvious and any mismatches will tend to be obvious in
     an individual line of code once you have understood the naming scheme
     and its rules.
*/




/**
   Perform exclusive cumulative sum: dest[i] = 0 + src[0] + src[1] + ... src[i-1] for
   0 <= i < n.  Note: although the input for 0 <= i < n-1 is all that affects
   the output, the last element src[n-1] may still be accessed in the CUDA kernel
   so you should make sure it was allocated as part of the array being summed
   even if the value was not set.

      @param [in] D     Device.  If n > 0 must be kCpu or kGpu; if n == 0, kUnk is also
                        allowed.

      @param [in] n     Number of elements in the input and output arrays (although
                        only items up to n-1 in the input array will affect the
                        result).  Must be >= 0

      @param [out] s       Array to which to write the exclusive sum (device pointer); size
                           must be at least t.size() + 1.
      @param [out] cpu_total  Optionally (if non-NULL), the sum of the elements
                           of t will be written to here, to a CPU address.
    This function may not wait for the kernel to terminate if cpu_total == NULL.

  IMPLEMENTATION NOTES:
     - If size of t is small enough that it makes sense to do it in one cooperative
       thread group (and maybe a small internal loop if needed), do that.
     - Otherwise, do it with 3 consecutive kernels:
       consider the input array to be made up of blocks of size BLOCK_SIZE, equal to
       some power of 2.  First, invoke the same kernel we used above to write
       the this-block-only partial sum for each block (note: only the 1st kernel
       should write the initial 0, to avoid race conditions), so e.g. if the
       input was [ 1 2 3 4 5 ] and BLOCK_SIZE = 2, the temporary array would be
       [ 0 1 3 3 7 5 ].  Then use a single thread block to inclusive-sum the
       values at multiples of BLOCK_SIZE, so the array looks like [ 0 1 3 3 10 5 ]
       (note: only the 7 changed, to 10 here).  Then use a single simple kernel
       to, for each index i that is not a multiple of BLOCK_SIZE, add to it the
       value at the most recent multiple of BLOCK_SIZE, so the array would look
       like [ 0 1 3 6 10 15 ].
 */
template <typename SrcPtr, typename DestPtr>
void ExclusiveSum(ContextPtr &c, int n, SrcPtr src, DestPtr dest);



/* Return the maximum value of the device array 't'.  Note: the sum will be
   initialized with T(0).

   Implementation notes: similar to ExclusiveSum.  We might at some point
   combine this with ExclusiveSum to (optionally) get the max value with little
   additional cost.
 */
template <typename T>
T MaxValue(Context *c, size_t nelems, T *t);

/*
  This is a rather special purpose function that is used for k2 Array.

  It sets x[i] to the index j to which position i 'belongs' according to
  the array 't'.  't' is expected to be an array containing the exclusive
  sum of a sequence of positive integers, so suppose there was a original
  sequence
       n = [ 2 1 4 ]
  and  t = [ 0 2 3 7 ]
  (and n_indexes = len(n) = 3, nelems = sum(n) = 7), then we would fill
  x with:
       x = [ 0 0 1 2 2 2 2 ]

  IMPLEMENTATION NOTES:
     One possibility: kernel runs for each element of x, and does some kind
    of search in `t`: either a binary search, or a smart one that tries to estimate
    the average number of elements per bin and hence converge a bit faster (in the
    normal case).

    Another possibility: two kernels.  Suppose the range we need to fill with a
    particular number (say, x) is from 1010 to 10000 inclusive (binary) The
    first kernel writes x to positions 1010, 1100, 10000; the significance of
    that sequence is we keep adding the smallest number we can add to get
    another zero at the end of the binary representation, until we exceed the
    range we're supposed to fill.  The second kernel: for a given index into x
    that is must fill (say, 1111), it asks "is the index currently here already
    the right one?", which it can test using the function is_valid_index()
    below; if it's not already corret, it searches in a sequence of positions:
    1110, 1100, 1000, 0000, like our sequence above but going downwards, again
    getting more zeros at the end of the binary representation, until it finds
    the correct value in the array at the searched position; then it copies the
    discovered value the original position requested (here, 1111).


    First kernel pseudocode: for each index 'i' into 't', it does:
      for (int n=0, j = t[i]; j < t[i+1]; n++) {
         x[j] = i;
         if (j & (1<<n))  j += (1 << n);
      }
    Second kernel pseudocode: for each element of x, it searches for the right index.  Suppose we're
    given num_indexes == length(n) == length(t) - 1.  Define is_valid_index as follows:
       // returns true if j is the value that we should be putting at position 'i' in x:
       // that is, if t[j] <= i < t[j+1].
       bool is_valid_index(i, j) {
          return (j >= 0 && j < num_indexes && t[j] <= i && i < t[j+1]);
       }
       // We suppose we are given i (the position into x that we're responsible for
       // setting:
       orig_i = i;
       for (int n=0; !is_valid_index(i, x[i]); n++) {
         if (i & (1<<n))  i -= (1 << n);
       }
       x[orig_i] = x[i];
*/
template <typename T>
T RowSplitsToRowIds(ContextPtr &c, T *row_splits, T *row_ids);


/*
  See above for 'row_ids concept' and 'row_splits' concept.
  This function turns a vector of row_ids into a vector of row_splits,
  e.g. given [ 0 0 1 1 1 2 ] it would produce [ 0 2 5 6 ].

   @param [in] row_ids  Input DeviceVec representing row_ids
   @param [out] row_splits  Output DeviceVecW representing row_splits;
                     its size must equal row_ids[row_ids.size() - 1] + 2.

   At exit (from the kernel, of course), for each 0 <= i <= row_ids.size(), if i
   == 0 or i == row_ids.size() or row_ids[i] != row_ids[i+1], row_splits[row_ids[i]]
   will be set to i.  Note: the row_splits must be consecutive (no gaps,
   i.e. nothing like [ 0 0 2 2 3 ], or certain elements of the output
   `row_splits` will be undefined.
 */
template <typename T>
T RowIdsToRowSplits(Context *c, T *row_ids, T *row_splits);


 __host__ __device__ __forceinline__ int32_t FloatAsInt(float f) {
   union { float f; int i; } u;
   u.f = f;
   return u.i;
 }

 __host__ __device__ __forceinline__ float IntAsFloat(int32_t i) {
   union { float f; int i; } u;
   u.i = i;
   return u.f;
 }


/*
 1:1 Conversion float <---> sortable int We convert floats to sortable ints in
 order to use native atomics operation, which are way faster than looping over
 atomicCAS
*/
__host__ __device__ __forceinline__ int32_t FloatToOrderedInt(float f) {
  int32_t i = FloatAsInt(f);
  return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__host__ __device__ __forceinline__ float OrderedIntToFloat(int32_t i) {
  return IntAsFloat((i >= 0) ? i : i ^ 0x7FFFFFFF);
}

/*
  Host version of Cuda's atomicMax function, marked __host__ (the default) for
  clarity.  So we can use this in lambdas that run on both host and device.
 */
__host__ int32_t atomicMax(int32_t* address, int32_t val) {
  int32_t old = *address;
  if (old < val)
    *address = val;
  return old;
}

}  // namespace k2

#endif  // K2_CSRC_CUDA_UTILS_H_
