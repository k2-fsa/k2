// k2/csrc/cuda/ragged_shape.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_RAGGED_SHAPE_H_
#define K2_CSRC_CUDA_RAGGED_SHAPE_H_

#include "k2/csrc/cuda/array.h"
#include "k2/csrc/cuda/algorithms.h"

namespace k2 {


class RaggedShape {
  int32_t Dim0() {
    CHECK_GT(0, axes_.size());
    return axes_[0].row_splits.Dim() - 1;
  }
  // total size on this axis (require 0 < axis < NumAxes()).
  int32_t TotSize(int32_t axis);

  Array1<int32_t> &RowSplits(int32_t axis) {
    CHECK_LT(static_cast<uint32_t>(axis - 1), axes_.size());
    return axes_[axis - 1].row_splits;
  }
  Array1<int32_t> &RowIds(int32_t axis) {
    CHECK_LT(static_cast<uint32_t>(axis - 1), axes_.size());
    // TODO: make sure this row_ids exists, create it if needed.
    return axes_[axis - 1].row_ids;
  }

  int32_t NumAxes() { return axes_.size() + 1; }

  ContextPtr &Context() { return axes_[0].row_splits.Context(); }

 private:
  struct RaggedShapeDim {
    Array1<int32_t> row_splits;
    Array1<int32_t> row_ids;
    int32_t cached_tot_size;
  };

  // indexed by axis-index minus one... axis 0 is special, its dim
  // equas axes_[0].row_splits.Dim()-1.
  std::vector<RaggedShapeDim> axes_;

};



// Shape of a 2-dimensional ragged array ( e.g. [ [ 0 ] [ 3 4 1 ] [] [ 5 ] ])
// Please see utils.h for an explanation of row_splits and row_ids.
class RaggedShape2 {
  // return dim on 0th axis.
  int32_t Dim0() const { return row_splits0_.size - 1; }
  Array1<int32_t> &RowSplits1() { return row_splits1_; }
  Array1<int32_t> &RowIds1();

  Array1<int32_t> Sizes1();

  // Return the *total* size (or dimension) of axis 1, summed across all rows.  Caution: if
  // you're using a GPU this call blocking the first time you call it for a
  // particular RaggedShape2(), so it may be best to do it in multiple threads
  // if you do this on many objects.  You can also get it from the last element
  // of RowSplits0() or the size of RowIds1() if that exists.
  int32_t TotSize1();
  // max_size1() returns the maximum of any element of Sizes1().
  int32_t MaxSize1();


  ContextPtr &Context() { return row_splits1_.Context(); }

 protected:
  Array1<int32_t> row_splits1_;
  Array1<int32_t> row_ids1_;  // populated on demand.  Size() ==
                              // cached_tot_size1_, assuming that value is not
                              // -1 (i.e. that it's known).
  int32_t cached_tot_size1_;
};


/* Returns the TotSize1() of an array of RaggedShape2, as an array,
   with the same context as the first of the 'src' array.
*/
Array1<int32_t> GetTotSize1(const std::vector<RaggedShape2*> &src);

/* Returns the TotSize1() of an array of RaggedShape2, as an array,
   with the same context as the first of the 'src' array.
*/
Array1<int32_t> GetTotSize2(const std::vector<RaggedShape3*> &src);


class RaggedShape3: public RaggedShape2 {

  RaggedShape3(Array<int32_t> &row_splits1,
               Array<int32_t> &row_splits2,
               int32_t cached_size2_ = -1,
               Array<int32_t> *row_ids1 = nullptr,
               Array<int32_t> *row_ids2 = nullptr);


  Array1<int32_t> &RowSplits2() { return row_splits2_; }
  Array1<int32_t> &RowIds2();

  Array1<int32_t> Sizes2();

  // Removes an axis by appending those lists; 'axis' must be 0 or 1
  RaggedShape2 RemoveAxis(int32_t axis);

  int32_t TotSize2() { return row_splits2_[-1]; }
  int32_t MaxSize2();

  ContextPtr &Context() { return row_splits1_.Context(); }
 protected:
  Array1<int32_t> row_splits2_;
  int32_t cached_size2_;  // TODO(Dan)?  can use to avoid gpu access ..
  Array1<int32_t> row_ids2_;
};


class RaggedShape4: public RaggedShape3 {
  Array1<int32_t> &RowSplits3() { return row_splits3_; }
  Array1<int32_t> &RowIds3();
  int32_t TotSize2() { return row_splits2_[-1]; }
  int32_t MaxSize2();

  // Removes an axis by appending those lists; 'axis' must be 0, 1 or 2.
  RaggedShape2 RemoveAxis(int32_t axis);

  ContextPtr &Context() { return row_splits1_.Context(); }


  // ...
};


typedef std::shared_ptr<RaggedShape2> RaggedShape2Ptr;
typedef std::shared_ptr<RaggedShape3> RaggedShape3Ptr;


/* Gets rid of an axis by appending those lists.  Contains
   the same underlying elements as 'src'.
     'axis' should be 0 or 1.
*/
RaggedShape2 AppendAxis(const RaggedShape3 &src, int32_t axis);



/*
  Create a RaggedShape2 from an array of elems and a array of row-ids
  (which may each element to its corresponding row).  The row-ids must
  be a nonempty vector, nonnegative and no-decreasing.

    @param [in]  num_rows   The number of rows (Dim0()) of the object to be created.
                 If a value <= 0 is supplied, it will use row_ids[-1]+1
                 if row_ids.size > 0, else 0.
    @param [in]  row_ids   The row-ids of the elements; must be nonnegative
                 and non-decreasing.
 */
RaggedShape2 RaggedShape2FromRowIds(int32_t num_rows,
                                    const Array<int32_t> &row_ids);


/* Create ragged3 shape from ragged2 shape and sizes
     @param [in]  shape2   The ragged2 shape that will give the top level
                       of the ragged3 shape.
     @param [in] sizes    An array with size shape2.TotSize1() and nonnegative
                       elements, will be the size of each corresponding element of
                       'shape2'.
 */
RaggedShape3 RaggedShape3FromSizes(const RaggedShape2 &shape2,
                                   const Array1<int32_t> &sizes);



/*
  Create a new RaggedShape4 that is a subsampling of 'src', i.e. some elements
  are kept
       @param [in] src   The original RaggedShape3 to subsample
       @param [in] kept0  .. TODO...


       for each axis: either nullptr if all lists or elements are
                         to be kept, or:  1 if this arc is to be kept, 0 otherwise.
                         The Size() of this must equal src.TotSize2()+1, but the last element

                         , but its
                         memory region must contain one extra element, to avoid
                         possible segmentation faults when we do the exclusive-sum.
                         This will be checked.
 */
RaggedShape3 RaggedShape3Subsampled(const RaggedShape3 &src,
                                    const Renumbering *r0,
                                    const Renumbering *r1,
                                    const Renumbering *r2);


/*
  Create a new RaggedShape3 that is a subsampling of 'src', i.e. some elements
  are kept
       @param [in] src   The original RaggedShape3 to subsample
       @param [in] kept0  .. TODO...


       for each axis: either nullptr if all lists or elements are
                         to be kept, or:  1 if this arc is to be kept, 0 otherwise.
                         The Size() of this must equal src.TotSize2()+1, but the last element

                         , but its
                         memory region must contain one extra element, to avoid
                         possible segmentation faults when we do the exclusive-sum.
                         This will be checked.
 */
RaggedShape4 RaggedShape4Subsampled(const RaggedShape4 &src,
                                    const Renumbering *r0,
                                    const Renumbering *r1,
                                    const Renumbering *r2,
                                    const Renumbering *r3);

/*
  Merge a list of RaggedShape3 to create a RaggedShape4.  This is a rather
  special case because it combines two issues: creating a list, and transposing,
  so instead of Dim0() of the result corresponding to src.size(), the
  dimensions on axis 1 all correspond to src.size().  There is a requirement
  that src[i]->Size() must all have the same value.

  Viewing the source and result as the shapes of n-dimensional arrays, we will have:
      result[i,j,k,l] = (*src[j])[i,k,l]
  (although of course no such operator actually exists at the C++ level).

 */
RaggedShape4 MergeToAxis1(const std::vector<const RaggedShape3*> &src);

/*
  Merge a list of RaggedShape2 to create a RaggedShape3.  This is a rather
  special case because it combines two issues: creating a list, and transposing,
  so instead of Dim0() of the result corresponding to src.size(), the
  dimensions on axis 1 all correspond to src.size().  There is a requirement
  that src[i]->Size() must all have the same value.

  Viewing the source and result as the shapes of n-dimensional arrays, we will have:
      result[i,j,k] = (*src[j])[i,k]
  (although of course no such operator actually exists at the C++ level).
*/
RaggedShape3 MergeToAxis1(const std::vector<const RaggedShape2*> &src);



/*
  This is as Ragged3Subsampled (and refer to its documentation, but as if you
  had done:
   <code>
       // Intead of:
       // ragged = Ragged3Subsampled(src, kept);
       Array1<int32_t> reorder(kept.size() + 1);
       ExclusiveSum(kept, &reorder);
       ragged = Ragged3SubsampledFromNumbering(src, reorder);
  </code>
     @param [in] src   The source shape from which we'll subsample this shape
     @param [in] reorder  A vector that will be used to map the old RowSplits2()
                     to the new RowSplits2(); its length must equal
                     src.TotSize2() + 1.
 */
RaggedShape3 RaggedShape3SubsampledFromNumbering(const RaggedShape3 &src,
                                                 const Array1<int32_t> &reorder);


/* Constructs RaggedShape4 from row splits, and sets up row-ids. */
RaggedShape4 RaggedShape4FromRowSplits(const Array<int32_t> &row_splits1,
                                       const Array<int32_t> &row_splits2,
                                       const Array<int32_t> &row_splits3);

/* Constructs RaggedShape4 from sizes, and sets up row-splits and row-ids. */
RaggedShape4 RaggedShape4FromSizes(const Array<int32_t> &sizes1,
                                   const Array<int32_t> &sizes2,
                                   const Array<int32_t> &sizes3);

/*
  Construct a Ragged4 from a Ragged3 and a row_splits3..
     @param [in] src   Ragged3 which will be the same as the base-class of the
                       returned Ragged4.
     @param [in] row_splits3  Vector of non-decreasing integers starting from zero;
                       size must equal TotSize2() of src.  Its last element is the
                       number of elements in the Ragged4 with this shape.
*/
RaggedShape4 RaggedShape4FromShape3AndRowSplits(const RaggedShape3 &src,
                                                const Array32<int32_t> &row_splits3);



}  // namespace k2

#endif  // K2_CSRC_CUDA_RAGGED_SHAPE_H_
