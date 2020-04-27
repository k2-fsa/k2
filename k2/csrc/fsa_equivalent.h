// k2/csrc/fsa_equivalent.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#include <cstdint>
#include <vector>

#include "k2/csrc/fsa.h"

#ifndef K2_CSRC_FSA_EQUIVALENT_H_
#define K2_CSRC_FSA_EQUIVALENT_H_

namespace k2 {

/*
  Returns true if the Fsa `a` is equivalent to `b`.
  CAUTION: this one will be quite hard to implement.
 */
bool IsEquivalent(const Fsa &a, const Fsa &b);

/*
  Gets a random path from an Fsa `a`, returns true if we get one path
  successfully.
*/
bool RandomPath(const Fsa &a, Fsa *b,
                std::vector<int32_t> *state_map = nullptr);

}  // namespace k2

#endif  // K2_CSRC_FSA_EQUIVALENT_H_
