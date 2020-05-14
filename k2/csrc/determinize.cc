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
  (*(cur_states->begin()))->backward_prob = cur_forward_prob;
  deriv_out->clear();
  for (int32_t i = 0; i < num_steps; i++) {
    for (LogSumTracebackState *state_ptr : *cur_states) {
      double backward_prob = state_ptr->backward_prob;
      for (auto link : state_ptr->prev_elements) {
        float arc_log_posterior = link.forward_prob + backward_prob;
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
  *weight_out = cur_forward_prob - prev_forward_prob;
  // The following is mostly for ease of interpretability of the output;
  // conceptually the order makes no difference.
  // TODO(dpovey): maybe remove this, for efficiency?
  std::reverse(deriv_out->begin(), deriv_out->end());
}

void TraceBack(std::unordered_set<MaxTracebackState *> *cur_states,
               int32_t num_steps,
               const float *,  // arc_weights_in, unused.
               float *weight_out, std::vector<int32_t> *deriv_out) {
  CHECK_EQ(cur_states->size(), 1);
  MaxTracebackState *state = *(cur_states->begin());
  double cur_forward_prob = state->forward_prob;
  deriv_out->resize(num_steps);
  for (int32_t i = num_steps - 1; i >= 0; i--) {
    // `deriv_out` is just a list of arc indexes in the input FSA
    // that this output arc depends on (it's their sum).
    (*deriv_out)[i] = state->arc_id;
  }
  double prev_forward_prob = state->forward_prob;
  *weight_out = cur_forward_prob - prev_forward_prob;
}

template <class TracebackState>
bool NormalizeStateAndOutputArc(
    DetState<TracebackState> *state, const WfsaWithFbWeights &wfsa_in,
    float prune_cutoff, std::vector<Arc> *arcs_out,
    std::vector<float> *arc_weights_out,
    std::vector<std::vector<typename TracebackState::DerivType>>
        *derivs_per_arc,
    DetStateMap<TracebackState> *state_map) {
  float arc_weight;
  std::vector<typename TracebackState::DerivType> deriv_info;
  state->Normalize(wfsa_in, &arc_weight, &deriv_info);
  int32_t next_state_id;
  bool is_new_state = state_map->GetOutputState(state);
  arcs_out->push_back({this->state_id, next_state_id, label});
  arc_weights_out->push_back(arc_weight);
  derivs_per_arc->push_back(std::move(deriv_info));
  return is_new_state;
}

template <class TracebackState>
int32_t DetState<TracebackState>::ProcessArcs(
    const WfsaWithFbWeights &wfsa_in, double prune_cutoff,
    std::vector<Arc> *arcs_out, std::vector<float> *arc_weights_out,
    std::vector<std::vector<typename TracebackState::DerivType>>
        *derivs_per_arc,
    DetStateMap<TracebackState> *state_map,
    DetStatePriorityQueue<TracebackState> *queue) {
  int32_t num_steps = 0;

  std::unordered_map<int32_t, DetState<TracebackState> *> label_to_state;

  // The following loop populates `label_to_state`, creating successor
  // DetStates (unnormalized).
  const Fsa &fsa = wfsa_in.fsa;
  const float *arc_weights = wfsa_in.arc_weights;
  for (const std::shared_ptr<TracebackState> &state_ptr : elements) {
    int32_t state_id = state_ptr->state_id,
            begin_arc = fsa.arc_indexes[state_id],
            end_arc = fsa.arc_indexes[state_id + 1];
    num_steps += end_arc - begin_arc;
    for (int32_t a = begin_arc; a < end_arc; ++a) {
      const Arc &arc = fsa.arcs[a];
      float weight = arc_weights[a];
      int32_t label = arc.label;
      auto ret = label_to_state.insert({label, nullptr});
      auto iter = ret.first;
      if (ret.second) {  // Inserted -> this label was not a key in this map.
                         // Allocate new DetState.
        iter->second = new DetState<TracebackState>> (seq_len + 1);
      }
      DetState<TracebackState> *det_state = iter->second.get();
      det_state->Accept(state_ptr, a, arc.label, weight);
    }
  }
  CHECK(!label_to_state.empty() ||
        elements[0]->state_id == fsa.FinalState());  // I'm assuming the input
                                                     // FSA is connected.

  // The following loop normalizes successor det-states, outputs the arcs
  // that lead to them, and adds them to the queue if necessary.
  for (auto iter = label_to_state.begin(); iter != label_to_state.end();
       ++iter) {
    DetState<TracebackState> *det_state = iter->second;

    float arc_weight;
    std::vector<DerivType> deriv_info;
    det_state->Normalize(wfsa_in, &arc_weight, &deriv_info);
    if (det_state->forward_backward_prob >= prune_cutoff) {
      bool is_new_state = state_map->GetOutputState(det_state);
      arcs_out->push_back({this->state_id, next_state_id, label});
      arc_weights_out->push_back(arc_weight);
      derivs_per_arc->push_back(std::move(deriv_info));
      if (is_new_state)
        queue->push(std::unique_ptr<DetState<TracebackState>>(det_state));
    } else {
      delete det_state;
    }
  }
  return num_steps;
}

template <class TracebackState>
void DetState<TracebackState>::Normalize(const WfsaWithFbWeights &wfsa_in,
                                         float *removed_weight,
                                         std::vector<DerivType> *deriv_info) {
  std::unordered_set<TracebackState *> cur_states;

  double fb_prob = -std::numeric_limits<double>::infinity();
  for (auto p : elements) {
    TracebackState *state = p.second.get();
    fb_prob = LogSumOrMax<TracebackState>(
        fb_prob,
        state->forward_prob + wfsa_in.backward_state_weights[state->state_d]);
    cur_states.insert(state);
  }

  int32_t new_seq_len = GetMostRecentCommonAncestor(&cur_states);
  // now cur_states.size() == 1.
  CHECK_EQ(cur_states.size(), 1);
  CHECK_LE(new_seq_len, seq_len);

  const TracebackState *base_state = cur_states.front().get();
  // The following statement is a correction term that we add to
  // forward_backward_prob, in which we replace the forward_prob in the DetState
  // (which will have been computed in a path-dependent way) with the
  // forward_prob in wfsa_in.  Note: the values of state->forward_prob above can
  // be thought of as base_state->forward_prob plus some value that only depends
  // on the symbol sequence.  The point of this is to ensure that
  // this->forward_backward_prob (which is used for pruning) depends only on the
  // base_state and the symbol sequence, and not on "how we got here", i.e.  the
  // history of DetStates from which this one is derived via ProcessArcs().
  fb_prob += wfsa_in.forward_state_weights[base_state->state_id] -
             base_state->forward_prob;
  // set thi->forward_backward_prob; it will affect pruning.
  this->forward_backward_prob = fb_prob;
  this->seq_len = new_seq_len;

  // the following will set removed_weight and deriv_info.
  TraceBack(&cur_states, seq_len - new_seq_len, wfsa_in.arc_weights,
            removed_weight, deriv_info);

  normalized = true;
}

template <typename TracebackState>
float DeterminizePrunedTpl(
    const WfsaWithFbWeights &wfsa_in, float beam, int64_t max_step,
    Fsa *fsa_out, std::vector<float> *arc_weights_out,
    std::vector<std::vector<typename TracebackState::DerivType>>
        *arc_derivs_out) {
  CHECK_GT(beam, 0);
  CHECK(IsDeterministic(wfsa_in.fsa));
  CHECK(!IsEmpty(wfsa_in.fsa));

  DetStatePriorityQueue<TracebackState> queue;
  DetStateMap<TracebackState> map;
  using DS = DetState<TracebackState>;

  std::shared_ptr<DS> start_state = std::make_shared<DS>();

  std::vector<Arc> arcs_out;
  arc_weights_out->clear();
  arc_derivs_out->clear();

  bool ans = map.GetOutputState(start_state.get());
  CHECK(ans && ans->state_id == 0);

  if (max_step <= 0) max_step = std::numeric_limits<int64_t>::max();
  int64_t num_steps = 0;
  int32_t block_size = 32;  // process a number of queue elements at a time
                            // between certain checks..

  double total_prob = wfsa_in.backward_state_weights[0],
         prune_cutoff = total_prob - beam;
  while (num_steps < max_step && !queue.empty()) {
    std::shared_ptr<DS> state = queue.top();
    queue.pop();
    num_steps +=
        state->ProcessArcs(wfsa_in, prune_cutoff, arcs_out, arc_weights_out,
                           arc_derivs_out, &map, &queue);
  }
  if (!queue.empty()) {  // We stopped early due to max_step
    return total_prob - queue.top()->forward_backward_prob;
  } else {
    return beam;
  }
}

}  // namespace k2
