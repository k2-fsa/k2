#include <cub/cub.cuh>

// Some quite low-level utilities.
// CAUTION: this is not up to date, I will simplify this probably.



/*
  sizes concept

  A vector of sizes is a vector of nonnegative integers, like [ 5 4 0 1 3 ].
  This just establishes a naming convention.  In most cases the 'sizes' will
  be the lengths of sub-lists in a list of lists.

  Relation to other concepts:
   A vector of sizes can be seen as the difference of successive elements of
  a vector of row_splits, i.e. sizes[i] = row_splits[i+1] - row_splits[i].
*/


/*
  row_splits concept

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
  row_ids concept

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
  sub-indexes concept

  sub-indexes represent the positions within the sub-lists, of a linearized list-of-lists.
  An example vector of sub-indexes is: [ 0 1 0 1 2 0 1 ].

  Relation to row_ids and row_splits:
    sub_indexes[i] = row_splits[row_ids[i] - i
  We will generally not explicitly write 'sub-indexes' to arrays but will compute them
  on the fly from the corresponding row_splits and row_ids.
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
void ExclusivePrefixSum(Context *c, int n, SrcPtr src, DestPtr dest);



/* Return the maximum value of the device array 't'.  Note: the sum will be
   initialized with T(0).

   Implementation notes: similar to ExclusiveSum.  We might at some point
   combine this with ExclusiveSum to (optionally) get the max value with little
   additional cost.
 */
template <type T>
T MaxValue(size_t nelems, T *t)

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
T RowSplitsToRowIds(Context *c, T *row_splits, T *row_ids);


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
