// k2/csrc/util.h

// Copyright (c)  2020  Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_UTIL_H_
#define K2_CSRC_UTIL_H_

#include <functional>
#include <utility>

#include "k2/csrc/fsa.h"

namespace k2 {

// boost::hash_combine
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);
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

}  // namespace k2
#endif  // K2_CSRC_UTIL_H_
