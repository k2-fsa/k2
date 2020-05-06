// k2/csrc/fsa_equivalent.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include <cstdint>
#include <vector>

#include "k2/csrc/fsa.h"

#ifndef K2_CSRC_FSA_EQUIVALENT_H_
#define K2_CSRC_FSA_EQUIVALENT_H_

namespace k2 {

/*
  Returns true if the Fsa `a` is stochastically equivalent to `b` by randomly
  generating `npath` paths from one of them and then checking if the
  paths exist in the other one.
 */
bool IsRandEquivalent(const Fsa &a, const Fsa &b, std::size_t npath = 100);

/*
  Gets a random path from an Fsa `a`, returns true if we get one path
  successfully.
*/
bool RandomPath(const Fsa &a, Fsa *b,
                std::vector<int32_t> *state_map = nullptr);

}  // namespace k2

#endif  // K2_CSRC_FSA_EQUIVALENT_H_
