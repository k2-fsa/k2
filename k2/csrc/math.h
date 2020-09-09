/**
 * @brief
 * math utilities
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_MATH_H_
#define K2_CSRC_MATH_H_

#include <algorithm>

#include "k2/csrc/context.h"

namespace k2 {

/*
  Returns index of highest bit set, in range -1..31.
  HighestBitSet(0) = -1,
  HighestBitSet(1) = 0,
  HighestBitSet(2,3) = 1
  ...
 */
int32_t HighestBitSet(int32_t i);


// returns random int32_t from [min..max]
int32_t RandInt(int32_t min, int32_t max);

// Returns random ints from a distribution that gives more weight to lower
// values.  I'm not implying this is a geometric distribution.  Anyway
// we aren't relying on any exact properties.
int32_t RandIntGeometric(int32_t min, int32_t max);



}


#endif  // K2_CSRC_MATH_H_
