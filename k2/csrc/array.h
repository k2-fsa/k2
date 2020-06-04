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
   We will use e.g. StridedPtr<int32_t, T> when the stride is not 1, and otherwise
   just T* (which presumably be faster).
*/
template <typename I, typename T>
class StridedPtr {
  T *data;
  I stride;
  T &operator [] (I i) { return data[i]; }
  StridedPtr(T *data, I stride): data(data), stride(stride) { }
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

  Ptr data;    // `data` might be an actual pointer, or might be some object
               // supporting operator [].  data[indexes[0]] through
               // data[indexes[size] - 1] must be accessible through this
               // object.
};


template <typename I, typename Ptr>
struct Array3 {
  using IndexT = I;
  using PtrT = Ptr;

  // Irregular three dimensional array of something, like vector<vector<X> >
  // where Ptr is or behaves like X*.
  using IndexT = I;
  using PtrT = Ptr;

  IndexT size;
  const IndexT *indexes1;  // indexes1[0,1,...size] should be defined; note, this
                           // means the array must be of at least size+1.  We
                           // require that indexes[i] <= indexes[i+1], but it is
                           // not required that indexes[0] == 0, it may be
                           // greater than 0.

  const IndexT *indexes2;  // indexes2[indexes1[0]]
                           // .. indexes2[indexes1[size]-1] should be defined.

  Ptr data;    // `data` might be an actual pointer, or might be some object
               // supporting operator [].  data[indexes[0]] through
               // data[indexes[size] - 1] must be accessible through this
               // object.


  Array2 operator [] (I i) {
    // ...
  }

};


// Note: we can create Array4 later if we need it.


// we'd put the following in fsa.h
using Cfsa = Array2<int32_t, Arc>;
using CfsaVec = Array3<int32_t, Arc>;





}  // namespace k2

#endif  // K2_CSRC_ARRAY_H_
