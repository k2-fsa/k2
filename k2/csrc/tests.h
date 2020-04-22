// k2/csrc/tests.h

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

// TODO(fangjun): rename this file
// since tests.h is not a good name

#include <cstdint>
#include <vector>

#include "k2/csrc/fsa.h"

#ifndef K2_CSRC_TESTS_H_
#define K2_CSRC_TESTS_H_

namespace k2 {

/*
  Returns true if the Fsa `a` is equivalent to `b`.
  CAUTION: this one will be quite hard to implement.
 */
bool IsEquivalent(const Fsa& a, const Fsa& b);

/* Gets a random path from an Fsa `a` */
void RandomPath(const Fsa& a, Fsa* b, std::vector<int32_t>* state_map = NULL);

}  // namespace k2

#endif  // K2_CSRC_TESTS_H_
