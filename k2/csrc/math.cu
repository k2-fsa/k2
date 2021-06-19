/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>  // strncmp

#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"

namespace k2 {
/*
  Returns index of highest bit set, in range -1..30.
  HighestBitSet(0) = -1,
  HighestBitSet(1) = 0,
  HighestBitSet(2) = 1
  HighestBitSet(3) = 1
  ...
 */
int32_t HighestBitSet(int32_t i) {
  K2_CHECK_GE(i, 0);
#if defined(__clang__) || defined (__GNUC__)
  if (i == 0) return -1;
  return 31 - static_cast<int32_t>(__builtin_clz(i));
#else
  for (int64_t j = 0; j < 32; ++j) {
    if (i < ((int64_t)1 << j)) {
      return j - 1;
    }
  }
#endif
  K2_LOG(FATAL) << "Unreachable code";
  return 32;
}

int32_t HighestBitSet(int64_t i) {
  K2_CHECK_GE(i, 0);
#if defined(__clang__) || defined (__GNUC__)
  if (i == 0) return -1;
  return 63 - static_cast<int32_t>(__builtin_clzl(i));
#else
  for (int64_t j = 0; j < 64; ++j) {
    if (i < ((int64_t)1 << j)) {
      return j - 1;
    }
  }
#endif
  K2_LOG(FATAL) << "Unreachable code";
  return 64;
}

int32_t RoundUpToNearestPowerOfTwo(int32_t n) {
  K2_CHECK_GE(n, 0);
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// returns random int32_t from [min..max]
int32_t RandInt(int32_t min, int32_t max) {
  K2_CHECK_GE(max, min);
  // declare as static intentionally here to make it constructed only once and
  // retain its state between calls
  static RandIntGenerator geneartor(GetSeed());
  return geneartor(min, max);
}

// Returns random ints from a distribution that gives more weight to lower
// values.  I'm not implying this is a geometric distribution.  Anyway
// we aren't relying on any exact properties.
int32_t RandIntGeometric(int32_t min, int32_t max) {
  static RandIntGeometricGenerator geneartor(GetSeed());
  return geneartor(min, max);
}

int32_t GetSeed() {
  static const char *seed = std::getenv("K2_SEED");
  if (seed == nullptr) return 0;

  return atoi(seed);  // 0 is returned if K2_SEED is not a numeric string.
}

namespace internal {
template <typename Real>
Real FixedRead(std::istream &is) {
  is >> std::ws;
  char c = is.peek();
  if (c == '-') {
    is.get();
    return -FixedRead<Real>(is);
  } else if (c == 'i' || c == 'I') {
    char c[10];
    int pos = 0;
    while (pos < 9 && isalpha(is.peek())) c[pos++] = tolower(is.get());
    c[pos] = '\0';
    if (strncmp(c, "inf", 3) && strncmp(c, "infinity", 8))
      is.setstate(std::ios::failbit);
    return std::numeric_limits<Real>::infinity();
    // can handle NaN's later, with:
    //} else if (c == 'n' || c == 'N') {
    // (NaN's are printed in a more complicated way though.
  } else {
    Real r;
    is >> r;
    return r;
  }
}
// Instantiate the template above.
template float FixedRead(std::istream &is);
template double FixedRead(std::istream &is);
}  // namespace internal

}  // namespace k2
