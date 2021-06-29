/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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
