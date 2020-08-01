// k2/csrc/determinize.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
// dpove@gmail.com, Haowen Qiu qindazhu@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/determinize.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace k2 {

template <typename TracebackState>
void Determinizer<TracebackState>::GetSizes(
    Array2Size<int32_t> *fsa_size, Array2Size<int32_t> *arc_derivs_size) {
  CHECK_NOTNULL(fsa_size);
  CHECK_NOTNULL(arc_derivs_size);
  fsa_size->size1 = fsa_size->size2 = 0;
  arc_derivs_size->size1 = arc_derivs_size->size2 = 0;

  arcs_.clear();
  arc_weights_.clear();
  arc_derivs_.clear();
  if (IsEmpty(fsa_in_.fsa)) return;

  DetStatePriorityQueue<TracebackState> queue;
  DetStateMap<TracebackState> map;
  using DS = DetState<TracebackState>;
  std::shared_ptr<DS> start_state(new DS());

  bool ans = map.GetOutputState(start_state.get(), fsa_in_.fsa);
  CHECK(ans && start_state->state_id == 0);

  if (max_step_ <= 0) max_step_ = std::numeric_limits<int64_t>::max();
  int64_t num_steps = 0;
  double total_prob = fsa_in_.BackwardStateWeights()[0],
         prune_cutoff = total_prob - beam_;
  queue.push(std::move(start_state));
  while (num_steps < max_step_ && !queue.empty()) {
    std::shared_ptr<DS> state(queue.top());
    queue.pop();
    num_steps += state->ProcessArcs(fsa_in_, prune_cutoff, &arcs_,
                                    &arc_weights_, &arc_derivs_, &map, &queue);
  }

  // We may stopped early due to max_step
  effective_beam_ =
      queue.empty() ? beam_ : total_prob - queue.top()->forward_backward_prob;

  CHECK_EQ(arcs_.size(), arc_derivs_.size());
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
float Determinizer<TracebackState>::GetOutput(
    Fsa *fsa_out, float *arc_weights_out,
    Array2<typename TracebackState::DerivType *, int32_t> *arc_derivs) {
  if (IsEmpty(fsa_in_.fsa)) return beam_;

  CHECK_NOTNULL(fsa_out);
  CHECK_NOTNULL(arc_weights_out);
  CHECK_NOTNULL(arc_derivs);

  std::vector<int32_t> arc_map;
  // output fsa
  CHECK_EQ(arcs_.size(), fsa_out->size2);
  CreateFsa(arcs_, fsa_out, &arc_map);
  CHECK_EQ(arcs_.size(), arc_map.size());

  // output arc weights
  ReorderCopyN(arc_map.begin(), arc_map.size(), arc_weights_.begin(),
               arc_weights_out);

  // output arc derivative information
  CHECK_EQ(arc_derivs_.size(), arc_derivs->size1);
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
template class Determinizer<MaxTracebackState>;
template class Determinizer<LogSumTracebackState>;

LogSumTracebackLink::LogSumTracebackLink(
    const std::shared_ptr<LogSumTracebackState> &src, int32_t arc_index,
    float arc_weight)
    : prev_state(src),
      arc_index(arc_index),
      forward_prob(arc_weight + src->forward_prob) {}

int32_t GetMostRecentCommonAncestor(
    std::unordered_set<LogSumTracebackState *> *cur_states) {
  int32_t ans = 0;
  std::unordered_set<LogSumTracebackState *> prev_states;
  for (; cur_states->size() != 1; ans++) {
    CHECK(!cur_states->empty());
    for (LogSumTracebackState *s : *cur_states) {
      for (LogSumTracebackLink &l : s->prev_elements) {
        prev_states.insert(l.prev_state.get());
      }
    }
    cur_states->clear();
    cur_states->swap(prev_states);
  }
  return ans;
}

int32_t GetMostRecentCommonAncestor(
    std::unordered_set<MaxTracebackState *> *cur_states) {
  int32_t ans = 0;
  std::unordered_set<MaxTracebackState *> prev_states;
  for (; cur_states->size() != 1; ans++) {
    CHECK(!cur_states->empty());
    for (MaxTracebackState *s : *cur_states) {
      prev_states.insert(s->prev_state.get());
    }
    cur_states->clear();
    cur_states->swap(prev_states);
  }
  return ans;
}

void TraceBack(std::unordered_set<LogSumTracebackState *> *cur_states,
               int32_t num_steps, const float *arc_weights_in,
               float *weight_out,
               std::vector<std::pair<int32_t, float>> *deriv_out) {
  std::unordered_set<LogSumTracebackState *> prev_states;
  assert(cur_states->size() == 1);
  // In the standard forward-backward algorithm for HMMs this backward_prob
  // would, mathematically, be 0.0, but if we set it to the negative of the
  // forward prob we can avoid having to subtract the total log-prob
  // when we compute posterior/occupation probabilities for arcs.
  double cur_forward_prob = (*(cur_states->begin()))->forward_prob;
  (*(cur_states->begin()))->backward_prob = -cur_forward_prob;
  deriv_out->clear();
  for (int32_t i = 0; i < num_steps; i++) {
    for (LogSumTracebackState *state_ptr : *cur_states) {
      double backward_prob = state_ptr->backward_prob;
      for (const auto &link : state_ptr->prev_elements) {
        auto arc_log_posterior =
            static_cast<float>(link.forward_prob + backward_prob);
        deriv_out->push_back(
            std::pair<int32_t, float>(link.arc_index, expf(arc_log_posterior)));
        LogSumTracebackState *prev_state = link.prev_state.get();
        double new_backward_prob =
            backward_prob + arc_weights_in[link.arc_index];
        if (prev_states.insert(prev_state).second) {  // newly inserted
          prev_state->backward_prob = new_backward_prob;
        } else {
          prev_state->backward_prob =
              LogAdd(new_backward_prob, prev_state->backward_prob);
        }
      }
    }
    cur_states->clear();
    cur_states->swap(prev_states);
  }
  // failure of the next assertion may indicate many kinds of bugs in the
  // algorithm.
  CHECK_EQ(cur_states->size(), 1);
  double prev_forward_prob = (*(cur_states->begin()))->forward_prob;
  *weight_out = static_cast<float>(cur_forward_prob - prev_forward_prob);
  // The following is mostly for ease of interpretability of the output;
  // conceptually the order makes no difference.
  // TODO(dpovey): maybe remove this, for efficiency?
  std::reverse(deriv_out->begin(), deriv_out->end());
}

void TraceBack(std::unordered_set<MaxTracebackState *> *cur_states,
               int32_t num_steps,
               const float *unused,  // arc_weights_in, unused.
               float *weight_out, std::vector<int32_t> *deriv_out) {
  (void)unused;
  CHECK_EQ(cur_states->size(), 1);
  MaxTracebackState *state = *(cur_states->begin());
  double cur_forward_prob = state->forward_prob;
  deriv_out->resize(num_steps);
  for (int32_t i = num_steps - 1; i >= 0; --i) {
    // `deriv_out` is just a list of arc indexes in the input FSA
    // that this output arc depends on (it's their sum).
    (*deriv_out)[i] = state->arc_id;
    state = state->prev_state.get();
  }
  double prev_forward_prob = state->forward_prob;
  *weight_out = static_cast<float>(cur_forward_prob - prev_forward_prob);
}

}  // namespace k2
