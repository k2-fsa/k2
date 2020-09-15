/**
 * @brief
 * arcsort
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_ARCSORT_H_
#define K2_CSRC_HOST_ARCSORT_H_

#include <vector>

#include "k2/csrc/host/fsa.h"

namespace k2host {
/**
    Sort arcs leaving each state in the input FSA on label first and
    then on `dest_state`
 */
class ArcSorter {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input FSA
  */
  explicit ArcSorter(const Fsa &fsa_in) : fsa_in_(fsa_in) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size) const;

  /*
    Finish the operation and output the arc-sorted FSA to `fsa_out` and
    arc mapping information to `arc_map` (if provided).
    @param [out] fsa_out   The output FSA; Will be arc-sorted;
                           Must be initialized; search for 'initialized
                           definition' in class Array2 in array.h for meaning.
    @param [out] arc_map   If non-NULL, will output a map from the arc-index
                           in `fsa_out` to the corresponding arc-index in
                           `fsa_in`.
                           If non-NULL, at entry it must be allocated with
                           size num-arcs of `fsa_out`, e.g. `fsa_out->size2`.
   */
  void GetOutput(Fsa *fsa_out, int32_t *arc_map = nullptr);

 private:
  const Fsa &fsa_in_;
};

// In-place version of ArcSorter; see its documentation;
// Note that if `arc_map` is non-NULL, then at entry it must be allocated with
// size num-arcs of `fsa`, e.g. `fsa->size2`
void ArcSort(Fsa *fsa, int32_t *arc_map = nullptr);

}  // namespace k2host

#endif  // K2_CSRC_HOST_ARCSORT_H_
