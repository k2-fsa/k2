// k2/csrc/fsa_util.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa.h"

#ifndef K2_CSRC_FSA_UTIL_H_
#define K2_CSRC_FSA_UTIL_H_

namespace k2 {

/*
  Computes lists of arcs entering each state (needed for algorithms that
  traverse the Fsa in reverse order).

  Requires that `fsa` be valid and top-sorted, i.e.
  CheckProperties(fsa, KTopSorted) == true.
*/
void GetEnteringArcs(const Fsa& fsa, VecOfVec* entering_arcs);

}  // namespace k2

#endif  // K2_CSRC_FSA_UTIL_H_
