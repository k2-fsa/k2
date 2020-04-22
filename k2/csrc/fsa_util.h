// k2/csrc/fsa_util.h

// Copyright     2020  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

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
void GetEnteringArcs(const Fsa &fsa, VecOfVec *entering_arcs);

}  // namespace k2

#endif  // K2_CSRC_FSA_UTIL_H_
