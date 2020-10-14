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
    num_old_elems_ = keep_.Dim();
  }

  int32_t NumOldElems() const { return num_old_elems_; }

  // Returns the number of kept elements, should be only called after New2Old()
  // or Old2New()
  int32_t NumNewElems() const { return num_new_elems_; }

  // 0 if not kept, 1 if kept (user will write to here).
  // Its dimension is `num_old_elems` provided in the constructor.
  Array1<char> &Keep() { return keep_; }

  /* Return a mapping from new index to old index.  This is created on
     demand (must only be called after the Keep() array has been populated).

       @return    Returns an array mapping the new indexes to the old
                  (pre-renumbering) indexes. Its dimension is the number of
                  new index (i.e. the number of 1 in keep_). Will be allocated
                  with the same context that keep_ holds (should be same with
                  the context passed in the constructor).
  */
  Array1<int32_t> New2Old();

  /* Return a mapping from old index to new index. This is created on demand
     (must only be called after the Keep() array has been populated).

       @return    Returns an array mapping the old indexes to the new indexes.
                  Its dimension is the number of old index (i.e. keep_.Dim()
                  or NumOldElems()). `-1` in it means the value is dropped in
                  new indexes. Will be allocated with the same context
                  that keep_ holds (should be same with the context passed
                  in the constructor).
  */
  Array1<int32_t> Old2New();

 private:
  Array1<char> keep_;
  int32_t num_old_elems_;
  int32_t num_new_elems_;
};

}  // namespace k2

#endif  // K2_CSRC_ALGORITHMS_H_
