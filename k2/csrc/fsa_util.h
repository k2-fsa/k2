// k2/csrc/fsa_util.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_UTIL_H_
#define K2_CSRC_FSA_UTIL_H_

#include <vector>

#include "k2/csrc/fsa.h"

namespace k2 {

/*
  Computes lists of arcs entering each state (needed for algorithms that
  traverse the Fsa in reverse order).

  Requires that `fsa` be valid and top-sorted, i.e.  CheckProperties(fsa,
  KTopSorted) == true.

    @param [out] arc_index   A list of arc indexes.
                             For states 0 < s < fsa.NumStates(),
                             the elements arc_index[i] for
                             end_index[s-1] <= i < end_index[s] contain the
                             arc-indexes in fsa.arcs for arcs that enter
                             state s.
    @param [out] end_index   For each state, the `end` index in `arc_index`
                             where we can find arcs entering this state, i.e.
                             one past the index of the last element in
                             `arc_index` that points to an arc entering
                             this state.
*/
void GetEnteringArcs(const Fsa &fsa, std::vector<int32_t> *arc_index,
                     std::vector<int32_t> *end_index);

}  // namespace k2

#endif  // K2_CSRC_FSA_UTIL_H_
