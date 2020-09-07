/**
 * @brief
 * algorithms
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ALGORITHMS_H_
#define K2_CSRC_ALGORITHMS_H_

#include "k2/csrc/array.h"

//  this really contains various utilities that are useful for k2 algorithms.
namespace k2 {

class Renumbering {
 public:
  Renumbering(int32_t num_old_elems);

  int32_t NumOldElems();
  int32_t NumNewElems();

  Array1<char> &Keep();  // dim is NumOldElems().  0 if not kept, 1 if kept
                         // (user will write to here).


  /* Return a mapping from new index to old index.  This is created on
     demand (must only be called after the Keep() array has been populated).

       @param include_final_value   If true the dimension of the result
                        will be NumNewElems(), the number of new elements
                        in the renumbering (the last element will be
                        NumOldElems().  If false, the last element
                        is omitted.
       @return    Returns an array mapping the new indexes to the old
                 (pre-renumbering) indexes.
  */
  Array1<int32_t> New2Old(bool include_final_value = true);


  /* Return a mapping from old index to new index (this is the exclusive-sum of
     `Keep()`).  This is created on demand (must only be called after the Keep()
     array has been populated).

       @param include_final_value   If true the dimension of the result
                      will be NumNewElems(), the number of new elements
                      in the renumbering (the last element will be
                      NumOldElems().  If false, the last element
                      is omitted.
       @return    Returns an array mapping the new indexes to the old
                 (pre-renumbering) indexes.
  */
  Array1<int32_t> Old2New(bool include_final_value = true);

 private:
  Array1<char> keep_;
  Array1<int32_t> new2old_;
  Array1<int32_t> old2new_;

};



}  // namespace k2

#endif  // K2_CSRC_ALGORITHMS_H_
