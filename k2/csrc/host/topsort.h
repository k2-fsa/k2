/**
 * @brief
 * topsort
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_TOPSORT_H_
#define K2_CSRC_HOST_TOPSORT_H_

#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"

namespace k2host {

/**
    Sort the input fsa topologically.
 */
class TopSorter {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input FSA
  */
  explicit TopSorter(const Fsa &fsa_in) : fsa_in_(fsa_in) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
  */
  void GetSizes(Array2Size<int32_t> *fsa_size);

  /*
    Finish the operation and output the top-sorted FSA to `fsa_out` and
    arc mapping information to `arc_map` (if provided).
    @param [out]  fsa_out Output fsa. It is set to empty if the input fsa had
                          cycles other than self-loops; otherwise it contains
                          the top-sorted fsa.
                          Must be initialized; search for 'initialized
                          definition' in class Array2 in array.h for meaning.
    @param [out]  arc_map If non-NULL, Maps from arc indexes in the output
                              fsa to arc indexes in input fsa.
                              If non-NULL, at entry it must be allocated with
                              size num-arcs of `fsa_out`,
                              e.g. `fsa_out->size2`.
    @return Returns true on success (i.e. the output will be topsorted).
            The only failure condition is when the input had cycles that were
            not self loops. Noted we may remove those states in the
            input Fsa which are not accessible or co-accessible.
            Caution: true return status does not imply that the returned FSA
            is nonempty.
   */
  bool GetOutput(Fsa *fsa_out, int32_t *arc_map = nullptr);

 private:
  const Fsa &fsa_in_;

  bool is_acyclic_;  // if the input FSA is acyclic (may have self-loops)
  std::vector<int32_t> arc_indexes_;  // arc_index of fsa_out
  std::vector<Arc> arcs_;             // arcs of fsa_out
  std::vector<int32_t> arc_map_;
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_TOPSORT_H_
