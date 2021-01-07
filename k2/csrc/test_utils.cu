/**
 * @brief
 * test_utils
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

void ToNotTopSorted(Fsa *fsa) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsa->Context()->GetDeviceType(), kCpu);

  int32_t num_states = fsa->TotSize(0);
  std::vector<int32_t> order(num_states);
  std::iota(order.begin(), order.end(), 0);
  std::random_shuffle(order.begin() + 1, order.end() - 1);

  Array1<Arc> &arcs = fsa->values;
  Arc *arcs_data = arcs.Data();
  int32_t num_arcs = arcs.Dim();
  for (int32_t i = 0; i != num_arcs; ++i) {
    int32_t src_state = arcs_data[i].src_state;
    int32_t dest_state = arcs_data[i].dest_state;
    arcs_data[i].src_state = order[src_state];
    arcs_data[i].dest_state = order[dest_state];
  }

  auto lambda_comp = [](const Arc &a, const Arc &b) -> bool {
    return a.src_state < b.src_state;
  };
  std::sort(arcs_data, arcs_data + num_arcs, lambda_comp);
  for (int32_t i = 0; i != num_arcs; ++i) {
    arcs_data[i].score = i;
  }

  bool error = true;
  *fsa = FsaFromArray1(arcs, &error);
  K2_CHECK(!error);
}

Fsa GetRandFsa() {
  NVTX_RANGE(K2_FUNC);
  k2host::RandFsaOptions opts;
  opts.num_syms = 5 + RandInt(0, 100);
  opts.num_states = 10 + RandInt(0, 2000);
  opts.num_arcs = opts.num_states * 4 + RandInt(0, 100);
  opts.allow_empty = false;
  opts.acyclic = true;

  k2host::RandFsaGenerator generator(opts);
  k2host::Array2Size<int32_t> fsa_size;
  generator.GetSizes(&fsa_size);
  FsaCreator creator(fsa_size);
  k2host::Fsa host_fsa = creator.GetHostFsa();
  generator.GetOutput(&host_fsa);
  Fsa ans = creator.GetFsa();
  ToNotTopSorted(&ans);

  return ans;
}

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
