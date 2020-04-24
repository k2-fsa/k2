// k2/csrc/tests.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

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
bool IsEquivalent(const Fsa &a, const Fsa &b);

/* Gets a random path from an Fsa `a` */
void RandomPath(const Fsa &a, Fsa *b, std::vector<int32_t> *state_map = NULL);

}  // namespace k2

#endif  // K2_CSRC_TESTS_H_
