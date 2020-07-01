// k2/csrc/topsort.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/topsort.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace k2 {
void TopSorter::GetSizes(Array2Size<int32_t> *fsa_size) {
  CHECK_NOTNULL(fsa_size);
  fsa_size->size1 = fsa_size->size2 = 0;
  is_connected_ = true;
  is_acyclic_ = true;
  order_.clear();

  if (IsEmpty(fsa_in_)) return;
  if (!IsConnected(fsa_in_)) {
    is_connected_ = false;
    return;
  }
  order_.reserve(fsa_in_.NumStates());
  is_acyclic_ = IsAcyclic(fsa_in_, &order_);
  if (!is_acyclic_) return;

  CHECK_EQ(order_.size(), fsa_in_.NumStates());
  fsa_size->size1 = fsa_in_.size1;  // = fsa_in_.NumStates()
  fsa_size->size2 = fsa_in_.size2;
}

bool TopSorter::GetOutput(Fsa *fsa_out, int32_t *state_map /* = nullptr*/) {
  CHECK_NOTNULL(fsa_out);
  if (IsEmpty(fsa_in_)) return true;
  if (!is_connected_) return false;
  if (!is_acyclic_) return false;

  auto num_states = fsa_in_.NumStates();
  std::vector<int32_t> state_in_to_out(num_states);
  for (auto i = 0; i != num_states; ++i) {
    state_in_to_out[order_[num_states - 1 - i]] = i;
  }
  // start state maps to start state
  CHECK_EQ(state_in_to_out.front(), 0);
  // final state maps to final state
  CHECK_EQ(state_in_to_out.back(), fsa_in_.FinalState());

  int32_t arc_begin;
  int32_t arc_end;
  int32_t num_arcs = 0;
  for (auto state_out = 0; state_out != num_states; ++state_out) {
    auto state_in = order_[num_states - 1 - state_out];
    arc_begin = fsa_in_.indexes[state_in];
    arc_end = fsa_in_.indexes[state_in + 1];

    fsa_out->indexes[state_out] = num_arcs;
    for (; arc_begin != arc_end; ++arc_begin) {
      auto arc = fsa_in_.data[arc_begin];
      arc.src_state = state_out;
      arc.dest_state = state_in_to_out[arc.dest_state];
      fsa_out->data[num_arcs++] = arc;
    }
  }
  fsa_out->indexes[num_states] = num_arcs;

  if (state_map != nullptr) {
    std::reverse_copy(order_.begin(), order_.end(), state_map);
  }
  return true;
}

}  // namespace k2
