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
template <typename I> struct Array2Size {
  using IndexT = I;
  // `size1` is the top-level size of the array, equal to the object's .size
  // element
  I size1;
  // `size2` is the nunber of elements in the array, equal to
  // o->indexes[o->size] - o->indexes[0] (if the Array2 object o is
  // initialized).
  I size2;
}


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

  PtrT data;    // `data` might be an actual pointer, or might be some object
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
  using IndexT = I;
  using PtrT = Ptr;

  // Irregular three dimensional array of something, like vector<vector<vetor<X> > >
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






// we'd put the following in fsa.h
using Cfsa = Array2<int32_t, Arc>;
using CfsaVec = Array3<int32_t, Arc>;




class FstInverter {
  /* Constructor.  Lightweight. */
  FstInverter(const Fsa &fsa_in, const AuxLabels &labels_in);

  /*
    Do enough work that know now much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the FSA
                                will be written to here
        @param [out] aux_size   The number of lists in the AuxLabels
                                output (==num-arcs) and the number of
                                elements will be written to here.
  */
  void GetSizes(Array2Size<int32_t> *fsa_size,
                Array2Size<int32_t> *aux_size);

  /*
    Finish the operation and output inverted FSA to `fsa_out` and
    auxiliary labels to `labels_out`.
       @param [out]  fsa_out  The inverted FSA will be written to
                         here.  Must be initialized; search for
                         'initialized definition' in class Array2
                         in array.h for meaning.
       @param [out]  labels_out  The auxiliary labels will be written to
                         here.  Must be initialized; search for
                         'initialized definition' in class Array2
                         in array.h for meaning.
   */
  void GetOutput(Fsa *fsa_out,
                 AuxLabels *labels_out);
 private:
  // ...
};

// Note: we can create Array4 later if we need it.


void InvertFst(const Fsa &fsa_in, const AuxLabels &labels_in, Fsa *fsa_out,
               AuxLabels *aux_labels_out) {


/*
void RmEpsilonsPrunedMax(const WfsaWithFbWeights &a, float beam, Fsa *b,
                         std::vector<std::vector<int32_t>> *arc_derivs);
*/



}  // namespace k2

#endif  // K2_CSRC_ARRAY_H_
