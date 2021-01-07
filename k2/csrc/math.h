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
#include <limits>
#include <random>

#include "k2/csrc/context.h"

namespace k2 {

/*
  Returns index of highest bit set, in range -1..31.
  HighestBitSet(0) = -1,
  HighestBitSet(1) = 0,
  HighestBitSet(2,3) = 1
  ...

  Note we may delete this function later if there's no other usage.
 */
int32_t HighestBitSet(int32_t i);

int32_t RoundUpToNearestPowerOfTwo(int32_t n);

// Generate a uniformly distributed random variable of type int32_t.
class RandIntGenerator {
 public:
  // Set `seed` to non-zero for reproducibility.
  explicit RandIntGenerator(int32_t seed = 0) : gen_(rd_()) {
    if (seed != 0) gen_.seed(seed);
  }

  // Get the next random number on the **closed** interval [low, high]
  int32_t operator()(int32_t low = std::numeric_limits<int32_t>::min(),
                     int32_t high = std::numeric_limits<int32_t>::max()) {
    K2_CHECK_GE(high, low);
    std::uniform_int_distribution<int32_t> dis(low, high);
    return dis(gen_);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
};

// Generate a geometric distributed random variable of type int32_t.
class RandIntGeometricGenerator {
 public:
  // Set `seed` to non-zero for reproducibility.
  explicit RandIntGeometricGenerator(int32_t seed = 0)
      : dis_(0.1), gen_(rd_()) {
    if (seed != 0) gen_.seed(seed);
  }

  // Get the next random number on the **closed** interval [low, high]
  int32_t operator()(int32_t low = std::numeric_limits<int32_t>::min(),
                     int32_t high = std::numeric_limits<int32_t>::max()) {
    K2_CHECK_GE(high, low);
    // Note using modulo here may introduce some bias into the generated random
    // number, but this does not matter for our usages.
    return low + (dis_(gen_) % (high - low + 1));
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::geometric_distribution<int32_t> dis_;
};

// returns random int32_t from [min..max]
int32_t RandInt(int32_t min, int32_t max);

// Returns random ints from a distribution that gives more weight to lower
// values.  I'm not implying this is a geometric distribution.  Anyway
// we aren't relying on any exact properties.
int32_t RandIntGeometric(int32_t min, int32_t max);

/*
 You are supposed to use IoFixer by: instead of doing
   float f;
   istream >> f;
 you do:
   InputFixer<float> f;
   istream >> f;
 For most types this code will be equivalent to just using the unmodified
 type, but for types float and double it "fixes" the broken behavior of
 the C++ standard w.r.t. infinity allowing infinities to be parsed.
*/
template<class T> struct InputFixer {
  T t;
  // cast operator
  operator T() const { return t; }
};


namespace internal {
template <typename Real>
Real FixedRead(std::istream &is);
}

template <typename T>
inline std::istream &operator >>(std::istream &is, InputFixer<T> &f) {
  return is >> f.t;
}
template <>
inline std::istream &operator >>(std::istream &is, InputFixer<float> &f) {
  f.t = internal::FixedRead<float>(is);
  return is;
}
template <>
inline std::istream &operator >>(std::istream &is, InputFixer<double> &f) {
  f.t = internal::FixedRead<double>(is);
  return is;
}

// Return the seed that can be used for random generators.
//
// It reads the environment variable `K2_SEED` to get the seed.
// If `K2_SEED` is not set or is not a numeric string,
// it returns 0.
int32_t GetSeed();
}  // namespace k2

#endif  // K2_CSRC_MATH_H_
