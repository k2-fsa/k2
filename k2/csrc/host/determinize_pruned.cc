/**
 * @brief
 * determinize
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "k2/csrc/host/determinize.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {

template <typename TracebackState>
void DeterminizerPruned<TracebackState>::GetSizes(
    Array2Size<int32_t> *fsa_size, Array2Size<int32_t> *arc_derivs_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  K2_CHECK_NE(arc_derivs_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;
  arc_derivs_size->size1 = arc_derivs_size->size2 = 0;

  arcs_.clear();
  arc_derivs_.clear();
  if (IsEmpty(fsa_in_.fsa)) return;

  DetStatePriorityQueue<TracebackState> queue;
  DetStateMap<TracebackState> map;
  using DS = DetState<TracebackState>;
  std::shared_ptr<DS> start_state(new DS());

  bool ans = map.GetOutputState(start_state.get(), fsa_in_.fsa);
  K2_CHECK(ans && start_state->state_id == 0);

  if (max_step_ <= 0) max_step_ = std::numeric_limits<int64_t>::max();
  int64_t num_steps = 0;
  double total_prob = fsa_in_.BackwardStateWeights()[0],
         prune_cutoff = total_prob - beam_;
  queue.push(std::move(start_state));
  while (num_steps < max_step_ && !queue.empty()) {
    std::shared_ptr<DS> state(queue.top());
    queue.pop();
    num_steps += state->ProcessArcs(fsa_in_, prune_cutoff, &arcs_, &arc_derivs_,
                                    &map, &queue);
  }

  // We may stopped early due to max_step
  effective_beam_ =
      queue.empty() ? beam_ : total_prob - queue.top()->forward_backward_prob;

  K2_CHECK_EQ(arcs_.size(), arc_derivs_.size());
  int32_t num_states_out = -1, num_derivs_out = 0;
  for (auto i = 0; i != arcs_.size(); ++i) {
    num_states_out = std::max(
        num_states_out, std::max(arcs_[i].src_state, arcs_[i].dest_state));
    num_derivs_out += arc_derivs_[i].size();
  }
  // as we suppose state-ids are starting from zero.
  ++num_states_out;
  fsa_size->size1 = num_states_out;
  fsa_size->size2 = arcs_.size();
  arc_derivs_size->size1 = arcs_.size();
  arc_derivs_size->size2 = num_derivs_out;
}

template <typename TracebackState>
float DeterminizerPruned<TracebackState>::GetOutput(
    Fsa *fsa_out,
    Array2<typename TracebackState::DerivType *, int32_t> *arc_derivs) {
  NVTX_RANGE(K2_FUNC);
  if (IsEmpty(fsa_in_.fsa)) return beam_;

  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_NE(arc_derivs, nullptr);

  std::vector<int32_t> arc_map;
  // output fsa
  K2_CHECK_EQ(arcs_.size(), fsa_out->size2);
  CreateTopSortedFsa(arcs_, fsa_out, &arc_map);
  K2_CHECK_EQ(arcs_.size(), arc_map.size());

  // output arc derivative information
  K2_CHECK_EQ(arc_derivs_.size(), arc_derivs->size1);
  int32_t num_derivs = 0;
  for (int32_t i = 0; i != arc_derivs->size1; ++i) {
    arc_derivs->indexes[i] = num_derivs;
    const auto &curr_arc_deriv = arc_derivs_[arc_map[i]];
    std::copy(curr_arc_deriv.begin(), curr_arc_deriv.end(),
              arc_derivs->data + num_derivs);
    num_derivs += curr_arc_deriv.size();
  }
  arc_derivs->indexes[arc_derivs->size1] = num_derivs;

  return effective_beam_;
}

// explicit instantiation here
template class DeterminizerPruned<MaxTracebackState>;
template class DeterminizerPruned<LogSumTracebackState>;

}  // namespace k2host
