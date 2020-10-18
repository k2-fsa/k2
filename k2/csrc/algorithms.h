/**
 * @brief
 * algorithms
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ALGORITHMS_H_
#define K2_CSRC_ALGORITHMS_H_

#include "k2/csrc/array.h"
#include "k2/csrc/log.h"

//  this really contains various utilities that are useful for k2 algorithms.
namespace k2 {

class Renumbering {
 public:
  Renumbering() = default;
  Renumbering(ContextPtr c, int32_t num_old_elems) { Init(c, num_old_elems); }

  void Init(ContextPtr c, int32_t num_old_elems) {
    // make the underlying region allocate an extra element as we'll do
    // exclusive sum in New2Old() and Old2New()
    Array1<char> temp = Array1<char>(c, num_old_elems + 1);
    keep_ = temp.Range(0, num_old_elems);
  }

  int32_t NumOldElems() const { return keep_.Dim(); }
    
  int32_t NumNewElems() {
    if (!old2new_.IsValid()) ComputeOld2New();
    return num_new_elems_;
  }  

  // 0 if not kept, 1 if kept (user will write to here).  Its dimension is
  // `num_old_elems` provided in the constructor.  we allocate initially with an
  // extra element because ExclusiveSum reads one past the end (since we get the
  // result with 1 extra element).
  Array1<char> &Keep() { return keep_; }

  /* Return a mapping from new index to old index.  This is created on
     demand (must only be called after the Keep() array has been populated).

       @return    Returns an array mapping the new indexes to the old
                  (pre-renumbering) indexes. Its dimension is the number of
                  new index (i.e. the number of 1 in keep_). Will be allocated
                  with the same context that keep_ holds (should be same with
                  the context passed in the constructor).
  */
  Array1<int32_t> &New2Old() {
    if (!new2old_.IsValid()) ComputeNew2Old();
    return new2old_;
  }

  /* Return a mapping from old index to new index. This is created on demand
     (must only be called after the Keep() array has been populated).

       @return    Returns an array mapping the old indexes to the new indexes.
                  Its dimension is the number of old indexes (i.e. keep_.Dim()
                  or NumOldElems()). It is just the exclusive sum of Keep().
                  It gives the mapping for indexes that are kept; ignore the 
                  non-kept elements of it.
                  Will be allocated with the same context as keep_.
  */
  Array1<int32_t> &Old2New() {
    if (!old2new_.IsValid()) ComputeOld2New();
    return old2new_;
  }
private:
  
  void ComputeOld2New();
  // ComputeNew2Old() also computes old2new_ if needed.
  void ComputeNew2Old();
  
  Array1<char> keep_;
  Array1<int32_t> old2new_;
  int32_t num_new_elems_;    // equals last element of old2new_; set when
                             // old2new_ is created.
  Array1<int32_t> new2old_; 
};

}  // namespace k2

#endif  // K2_CSRC_ALGORITHMS_H_
