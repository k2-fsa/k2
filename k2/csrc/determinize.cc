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

#include "k2/csrc/fsa_algo.h"

namespace k2 {

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

template <>
double LogSumOrMax<MaxTracebackState>(double a, double b) {
  return std::max(a, b);
}
template <>
double LogSumOrMax<LogSumTracebackState>(double a, double b) {
  return LogAdd(a, b);
}

}  // namespace k2
