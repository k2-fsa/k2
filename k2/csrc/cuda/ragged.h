#include "k2/csrc/cuda/shape.h"


// Interfaces:
//  Indexable1 means operator () (int)
//  Indexable2 means operator () (int, int)


// 2-dimensional ragged array.  T should be a POD type.
// We always ensure 'shape' and 'values' have the same device type.
template <class T> class Ragged2 {
  RaggedShape2 shape;
  Array1<T> values;

  ContextPtr &Context() { return shape.Context(); }
};


// 3-dimensional ragged array.  T should be a POD type.
// We always ensure 'shape' and 'values' have the same device type.
template <class T> class Ragged3 {
  RaggedShape3 shape;
  Array1<T> values;

  ContextPtr &Context() { return shape.Context(); }
};



// ways to construct ragged arrays...
//  Array2  From: Array1<T>,


/*
  Construct a Ragged3 array from an existing Ragged3 array and an integer array
  that should be 1 if we are to keep this element, 0 otherwise.  Currently it will
  keep all the lists present in `src` but just make them have no elements if
  no corresponding elements of `to_keep` were nonzero.

      @param [in] src  The source shape
      @param [in] to_keep   An array containing 1 if we are to keep the corresponding
                      element of the source arary, and 0 otherwise.
 */
template <typename T, typename I>
Ragged3Shape<T> Ragged3ShapeFromBoolArray(const Ragged3Shape &src,
                                          Array1<I> &to_keep);

/*
  This is similar to Ragged3ShapeFromBoolArray, but imagine you took the
  'to_keep' vector and computed its exclusive-prefix-sum into a vector whose
  dimension was one greater.  I.e. 'reordering' maps from the old to the new
  indexes.

  The length of `reordering` must be greater than the largest (i.e. last)
  element of src.RowIds2(), to avoid out-of-bounds access, but it does not
  have to be exactly equal to src.RowIds2()[-1], e.g. it can be one greater.

  This will create a Ragged3Shape whose RowIds2() equals
  reordering[src.RowIds2()].
 */
template <typename T, typename I>
Ragged3Shape<T> Ragged3ShapeFromReordering(const Ragged3Shape &src,
                                           Array1<I> &reordering);




/*
  Create a Ragged2<T> from an array of elems and a array of row-ids
  (which may each element to its corresponding row).  The row-ids must
  be a nonempty vector, nonnegative and no-decreasing.

    @param [in]  num_rows   The number of rows (Size0()) of the object to be created.
                 If a value <= 0 is supplied, it will use row_ids[-1]+1
                 if row_ids.size > 0, else 0.
    @param [in]  elems    The elements in the array (will become the .elems
                 of the returned array)
    @param [in]  row_ids   The row-ids of the elements; must be nonnegative
                 and non-decreasing, and its .size() must equal elems.size().
 */
template <typename T>
Ragged2<T> Ragged2FromRowIds(int num_rows,
                             const Array<T> &elems,
                             const Array<int> &row_ids);


/*
  Construct a 3-dim ragged array from a 2-dim ragged array and an array `row_ids`
  says, for each element of the .elems of the ragged3 array, which sub-list of
  the ragged2 array it corresponds to.

     @param [in] shape2    The ragged2 shape that will dictate the top-level shape of
                       the ragged2 arary (e.g. will share the Size0() and RowSplits1()).
     @param [in] elems     The elements of the ragged3 array (The only constraint on
                       the size of this array is that if shape2 has TotSize1()==0,
                       it must be empty).
     @param [in] row_ids   A nondecreasing vector of integers 0 <= i < shape2.TotSize1(),
                        with row_ids.size() == elems.size(), that maps each element of
                        `elems` to positions in the elemsn of a Ragged2 array with shape
                        `shape2`.
 */
template <typename T>
Ragged3<T> Ragged3FromRowIds(const Ragged2Shape &shape2,
                             const Array<T> &elems,
                             const Array<int> &row_ids);

/*
  Construct a 3-dim ragged array from a 2-dim ragged array and an array `row_ids`
  says, for each element of the .elems of the ragged3 array, which sub-list of
  the ragged2 array it corresponds to.

     @param [in] size0    The leading dimension of this ragged array, must be >= 0
     @param [in] row_ids   A nondecreasing vector of integers 0 <= i < size0,
                        with row_ids.size() == elems.size(), that maps each element of
                        `elems` to its leading index.
     @param [in] elems     The elements of the ragged2 array (may be uninitialized,
                        if you intend to initialize later).
 */
template <typename T>
Ragged3<T> Ragged2FromRowIds(int size0,
                             const Array<int> &row_ids,
                             const Array<T> &elems);
