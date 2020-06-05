// k2/csrc/array.h

// Copyright (c)  2020  Xiaomi Corporation (author: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_ARRAY_H_
#define K2_CSRC_ARRAY_H_

#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace k2 {

/*
   We will use e.g. StridedPtr<int32_t, T> when the stride is not 1, and
   otherwise just T* (which presumably be faster).
*/
template <typename I, typename T>
struct StridedPtr {
  T *data;
  I stride;
  T &operator[](I i) { return data[i]; }
  StridedPtr(T *data, I stride) : data(data), stride(stride) {}
};

/* MIGHT NOT NEED THIS */
template <typename I, typename Ptr>
struct Array1 {
  // Irregular two dimensional array of something, like vector<vector<X> >
  // where Ptr is, or behaves like, X*.
  using IndexT = I;
  using PtrT = Ptr;

  // 'begin' and 'end' are the first and one-past-the-last indexes into `data`
  // that we are allowed to use.
  IndexT begin;
  IndexT end;

  PtrT data;
};

/*
  This struct stores the size of an Array2 object; it will generally be used as
  an output argument by functions that work out this size.
 */
template <typename I>
struct Array2Size {
  using IndexT = I;
  // `size1` is the top-level size of the array, equal to the object's .size
  // element
  I size1;
  // `size2` is the nunber of elements in the array, equal to
  // o->indexes[o->size] - o->indexes[0] (if the Array2 object o is
  // initialized).
  I size2;
};

template <typename I, typename Ptr>
struct Array2 {
  // Irregular two dimensional array of something, like vector<vector<X> >
  // where Ptr is, or behaves like, X*.
  using IndexT = I;
  using PtrT = Ptr;

  IndexT size;
  const IndexT *indexes;  // indexes[0,1,...size] should be defined; note, this
                          // means the array must be of at least size+1.  We
                          // require that indexes[i] <= indexes[i+1], but it is
                          // not required that indexes[0] == 0, it may be
                          // greater than 0.

  PtrT data;  // `data` might be an actual pointer, or might be some object
              // supporting operator [].  data[indexes[0]] through
              // data[indexes[size] - 1] must be accessible through this
              // object.

  /* initialized definition:

        An Array2 object is initialized if its `size` member is set and its
        `indexes` and `data` pointer allocated, and the values of its `indexes`
        array are set for indexes[0] and indexes[size].
  */
};

template <typename I, typename Ptr>
struct Array3 {
  // Irregular three dimensional array of something, like vector<vector<vetor<X>
  // > > where Ptr is or behaves like X*.
  using IndexT = I;
  using PtrT = Ptr;

  IndexT size;
  const IndexT *indexes1;  // indexes1[0,1,...size] should be defined; note,
                           // this means the array must be of at least size+1.
                           // We require that indexes[i] <= indexes[i+1], but it
                           // is not required that indexes[0] == 0, it may be
                           // greater than 0.

  const IndexT *indexes2;  // indexes2[indexes1[0]]
                           // .. indexes2[indexes1[size]-1] should be defined.

  Ptr data;  // `data` might be an actual pointer, or might be some object
             // supporting operator [].  data[indexes[0]] through
             // data[indexes[size] - 1] must be accessible through this
             // object.

  Array2<I, Ptr> operator[](I i) {
    // TODO(haowen): fill real data here
    Array2<I, Ptr> array;
    return array;
  }
};

// Note: we can create Array4 later if we need it.

}  // namespace k2

#endif  // K2_CSRC_ARRAY_H_
