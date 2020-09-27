/**
 * @brief
 * util
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_UTIL_H_
#define K2_CSRC_HOST_UTIL_H_

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

namespace k2host {

#define EXPECT_DOUBLE_ARRAY_APPROX_EQ(expected, actual, abs_error)          \
  ASSERT_EQ((expected).size(), (actual).size()) << "Different Array Size."; \
  for (std::size_t i = 0; i != (expected).size(); ++i) {                    \
    EXPECT_TRUE(ApproxEqual((expected)[i], (actual)[i], abs_error))         \
        << "expected value at index " << i << " is " << (expected)[i]       \
        << ", but actual value is " << (actual)[i];                         \
  }

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
static const float kMinLogDiffFloat = logf(FLT_EPSILON);   // negative!

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
  }

  return x;  // return the larger one.
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
    res = x + log1pf(expf(diff));
    return res;
  }

  return x;  // return the larger one.
}

inline bool ApproxEqual(double a, double b, double delta = 0.001) {
  // a==b handles infinities.
  if (a == b) return true;
  double diff = std::abs(a - b);
  if (diff == std::numeric_limits<double>::infinity() || diff != diff)
    return false;  // diff is +inf or nan.
  return diff <= delta;
}

inline bool DoubleApproxEqual(double a, double b, double delta = 1e-6) {
  return a <= b + delta && b <= a + delta;
}

void *MemAlignedMalloc(size_t nbytes, size_t alignment);
void MemFree(void *ptr);

}  // namespace k2host
#endif  // K2_CSRC_HOST_UTIL_H_
