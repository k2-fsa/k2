/**
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/test_utils.h"

namespace k2 {


Array1<int32_t> GenerateRandomIndexes(ContextPtr context, bool allow_minus_one,
                                      int32_t dim, int32_t max_value) {
  std::vector<int32_t> indexes(dim);
  int32_t start = allow_minus_one ? -1 : 0;
  for (int32_t &i : indexes) {
    int32_t tmp = RandInt(-max_value, max_value);
    i = std::max(tmp, start);
  }

  return Array1<int32_t>(context, indexes);
}

}  // namespace k2
