// k2/csrc/util.h

// Copyright (c)  2020  Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_UTIL_H_
#define K2_CSRC_UTIL_H_

#include <cfloat>
#include <cmath>
#include <functional>
#include <utility>

#include "k2/csrc/fsa.h"

namespace k2 {

// boost::hash_combine
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {  // NOLINT
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);  // NOLINT
}

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    std::size_t result = 0;
    hash_combine(&result, pair.first);
    hash_combine(&result, pair.second);
    return result;
  }
};

static const double kMinLogDiffDouble = log(DBL_EPSILON);  // negative!
static const float kMinLogDiffFloat = log(FLT_EPSILON);    // negative!

// returns log(exp(x) + exp(y)).
inline double LogAdd(double x, double y) {
  double diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffDouble) {
    double res;
    res = x + log1p(exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

// returns log(exp(x) + exp(y)).
inline float LogAdd(float x, float y) {
  float diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
    res = x + log1p(exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

}  // namespace k2
#endif  // K2_CSRC_UTIL_H_
