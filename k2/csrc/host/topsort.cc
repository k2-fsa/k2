/**
 * @brief
 * topsort
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/topsort.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {
void TopSorter::GetSizes(Array2Size<int32_t> *fsa_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;
  arc_indexes_.clear();
  arcs_.clear();
  arc_map_.clear();

  std::vector<int32_t> state_out_to_in;
  is_acyclic_ = ConnectCore(fsa_in_, &state_out_to_in);
  if (!is_acyclic_) return;

  auto num_states_out = state_out_to_in.size();
  arc_indexes_.resize(num_states_out + 1);
  arcs_.reserve(fsa_in_.size2);
  arc_map_.reserve(fsa_in_.size2);

  std::vector<int32_t> state_in_to_out(fsa_in_.NumStates(), -1);
  for (auto i = 0; i != num_states_out; ++i) {
    auto state_in = state_out_to_in[i];
    state_in_to_out[state_in] = i;
  }

  auto arc_begin = 0;
  auto arc_end = 0;
  const int32_t arc_begin_index = fsa_in_.indexes[0];
  for (auto i = 0; i != num_states_out; ++i) {
    auto state_in = state_out_to_in[i];
    arc_begin = fsa_in_.indexes[state_in];
    arc_end = fsa_in_.indexes[state_in + 1];

    arc_indexes_[i] = static_cast<int32_t>(arcs_.size());
    for (; arc_begin != arc_end; ++arc_begin) {
      auto arc = fsa_in_.data[arc_begin];
      auto dest_state = arc.dest_state;
      auto state_out = state_in_to_out[dest_state];
      if (state_out < 0) continue;  // dest_state is unreachable
      arc.src_state = i;
      arc.dest_state = state_out;
      arcs_.push_back(arc);
      arc_map_.push_back(arc_begin - arc_begin_index);
    }
  }
  arc_indexes_[num_states_out] = arc_indexes_[num_states_out - 1];

  K2_CHECK_EQ(arcs_.size(), arc_map_.size());
  fsa_size->size1 = num_states_out;
  fsa_size->size2 = arcs_.size();
}

bool TopSorter::GetOutput(Fsa *fsa_out, int32_t *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  if (!is_acyclic_) return false;

  // output FSA
  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_EQ(arc_indexes_.size(), fsa_out->size1 + 1);
  std::copy(arc_indexes_.begin(), arc_indexes_.end(), fsa_out->indexes);
  K2_CHECK_EQ(arcs_.size(), fsa_out->size2);
  std::copy(arcs_.begin(), arcs_.end(), fsa_out->data);

  // output arc map
  if (arc_map != nullptr) std::copy(arc_map_.begin(), arc_map_.end(), arc_map);

  return true;
}
}  // namespace k2host
