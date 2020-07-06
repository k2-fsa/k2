// Some quite low-level utilities.


/**
   "DeviceVec type"
   This is a concept not a specific type.   When we do:

   template <class DeviceVec> ...

   we expect DeviceVec to be some object that supports
   the following interface:
     class DeviceVec {
        public:
         using value_type = ...;
         __host__ __device__ size_t size() const;
         __device__ ValueType operator [] const;
         __device__  DeviceVec(const DeviceVec &other);
     };

   The object will in the simplest case contain just a size and a
   pointer to the data it contains.
*/


/**
   "DeviceVecW type"
   This is a concept not a specific type.  The W stands for
   writable.  When we do:

   template <class DeviceVecW> ...

   we expect DeviceVecW to be some object that supports the following interface.
   Note, the only difference from DeviceVec is that it returns a reference not a
   value, so you can write to it if needed.  Note: you can treat DeviceVecW
   as a specialization of DeviceVec.


     class DeviceVecW {
        public:
         using value_type = ...;
         __host__ __device__ size_t size() const;
         __device__ ValueType & operator [];
         const __device__ ValueType & operator [] const;
         __device__  DeviceVecW(const DeviceVecW &other);
     };

   The object will in the simplest case contain just a size and a
   pointer to the data it contains.
*/


/*
   "DevicePtr type"

   When we do
   template <class DevicePtr> ...
   we expect either a pointer or something that supports being indexed
   with an integer, like a pointer.  It's like a weaker form of DeviceVec,
   without the size() member.

   DevicePtrW is like a weaker version of DeviceVecW, without the size()
   member.

   When we want to extract the value_type from DevicePtr and DevicePtrW,
   we will do something like:
      using value_type = ExtractPointerType<DevicePtr>::value_type;
   where we'll define the template ExtractPointerType appropriately in order
   to accomplish this.  The purpose of this is to allow DevicePtr and
   DevicePtrW to be actual raw device pointers.
 */

/*
  sizes concept

  A vector of sizes is a vector of nonnegative integers, like [ 5 4 0 1 3 ].
  This just establishes a naming convention.  In most cases the 'sizes' will
  be the lengths of sub-lists in a list of lists.

  Relation to other concepts:
   A vector of sizes can be seen as the difference of successive elements of
  a vector of offsets, i.e. sizes[i] = offsets[i+1] - offsets[i].
/

/*
  nsizes concept

  A vector of sizes is a vector of nonzero sizes, i.e. a vector of positive
  integers like [ 1 2 2 9 ].  It's like 'sizes' but excluding zero as a possibility.
*/

/*
  offset concept

  An offset vector is a vector of the form, say, [ 0 5 9 9 10 13 ].
  i.e. it starts with 0 and is non-decreasing.  It will often be encountered
  as the exclusive-sum of a vector of 'sizes' (see 'sizes concept' above),
  with a size one greater than that of the corresponding 'sizes'.  It
  will represent the positions in a single linearized list, of where we
  put the elements of a list of sub-lists.  So in the example above,
  sub-list 0 occupies positions 0,1,2,3,4, sub-list 1 occupies positions 5,6,7,8,
  and so on.

  Relation to other concepts:
    See 'groups concept' where its relation to 'offsets' is described.
    If offsets = [ 0 2 3 5 ], the corresponding 'groups' is:
    groups = [ 0 0 1 2 2 ].
*/

/*
  tails concept

  A vector of tails is a vector containing zeros and ones; each '1' represents
  the last element of a sub-list.  For example [ 0 1 0 0 0 1 1 ] to represent
  a list of sub-lists like [ x x ] [ y y y y ] [ z ].  The last element will always
  be 1.

  Relation to other concepts:
    The exclusive cumulative sum of a vector of tails is a vector of groups.
  E.g. the above example, with exclusive cumulative sum, is:
    [ 0 0 1 1 1 1 2 ].
 */

/*
  groups concept

  A vector of groups is a vector of the form
    [ 0 0 0 1 1 2 2 2 ]
  or in the general case any nonnegative, non-decreasing list of integers.  Each
  group represents the index of the sub-list to which that position belongs; for
  instance, if we had a list-of-lists like [ x x x ] [ y y ] [ z z z ], the
  above vector of groups would describe its structure.

  Relation to other concepts:
    A vector of groups can arise as the cumulative sum of a vector of tails.
    A vector of groups and a vector of offsets represent the same information
     in different ways, satisfying offsets[groups[i]] <= i < offsets[groups[i] + 1]
     and offsets[offsets.size() - 1] == groups.size().

*/

/*
  sub-indexes concept

  sub-indexes represent the positions within the sub-lists, of a linearized list-of-lists.
  An example vector of sub-indexes is: [ 0 1 0 1 2 0 1 ].

  Relation to groups and offsets:
    sub_indexes[i] = offsets[groups[i] - i
  We will generally not explicitly write 'sub-indexes' to arrays but will compute them
  on the fly from the corresponding offsets and groups.
*/


/**
   Perform exclusive cumulative sum: s[i] = 0 + t[0] + t[1] + ... t[i-1] for
   0 <= i <= ninputs.  Note: this only accesses the input
   t[i] for 0 <= i < ninputs.

      @param [in] t        Input array (device pointer); size given by t.size().
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
template <typename SrcPtr, typename DestVec, typename DestCategory>
ExclusiveCumSumImpl(SrcPtr &src, DestVec &dest,
                typename DestVec::ValueType *cpu_total);

template <typename SrcPtr, typename DestVec>
ExclusiveCumSum(SrcPtr &src, DestVec &dest,
                typename DestVec::ValueType *cpu_total) {

}



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
template <typename DeviceVec, typename DeviceVecW>
    T OffsetsToGroups(DeviceVec offsets, DeviceVecW groups);


/*
  See above for 'groups concept' and 'offsets' concept.
  This function turns a vector of groups into a vector of offsets,
  e.g. given [ 0 0 1 1 1 2 ] it would produce [ 0 2 5 6 ].

   @param [in] groups  Input DeviceVec representing groups
   @param [out] offsets  Output DeviceVecW representing offsets;
                     its size must equal groups[groups.size() - 1] + 2.

   At exit (from the kernel, of course), for each 0 <= i <= groups.size(), if i
   == 0 or i == groups.size() or groups[i] != groups[i+1], offsets[groups[i]]
   will be set to i.  Note: the offsets must be consecutive (no gaps,
   i.e. nothing like [ 0 0 2 2 3 ], or certain elements of the output
   `offsets` will be undefined.
 */
template <typename DeviceVec, typename DeviceVecW>
void GroupsToOffsets(DeviceVec groups, DeviceVecW offsets);
