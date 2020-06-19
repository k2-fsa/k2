// k2/csrc/fsa_algo.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "glog/logging.h"
#include "k2/csrc/determinize.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace {

constexpr int8_t kNotVisited = 0;  // a node that has not been visited
constexpr int8_t kVisiting = 1;    // a node that is under visiting
constexpr int8_t kVisited = 2;     // a node that has been visited
// depth first search state
struct DfsState {
  int32_t state;      // state number of the visiting node
  int32_t arc_begin;  // arc index of the visiting arc
  int32_t arc_end;    // end of the arc index of the visiting node
};

using StatePair = std::pair<int32_t, int32_t>;

inline int32_t InsertIntersectionState(
    const StatePair &new_state, int32_t *state_index_c,
    std::queue<StatePair> *qstates,
    std::unordered_map<StatePair, int32_t, k2::PairHash> *state_pair_map) {
  auto result = state_pair_map->insert({new_state, *state_index_c + 1});
  if (result.second) {
    // we have not visited `new_state` before.
    qstates->push(new_state);
    ++(*state_index_c);
  }
  return result.first->second;
}

/**
   A TraceBack() function used in RmEpsilonsPrunedLogSum.  It finds derivative
   information for all arcs in a sub-graph. Generally, in
   RmEpsilonsPrunedLogSum, we actually get a sub-graph when we find a
   non-epsilon arc starting from a particular state `s` (from which we are
   trying to remove epsilon arcs). All leaving arcs of all states in this
   sub-graph are epsilon arcs except the last one. Then, from the last state, we
   need to trace back to state `s` to find the derivative information for all
   epsilon arcs in this graph.
       @param [in] curr_states   (This is consumed destructively, i.e. don't
                       expect it to contain the same set on exit).
                       A set of states, stored as a std::map that mapping
                       state_id in input FSA to the corresponding
                       LogSumTracebackState we created for this state;
                       we'll iteratively trace back this set one element
                       (processing all entering arcs) at a time.  At entry
                       it must have size() == 1 which contains the last
                       state mentioned above; it will also have size() == 1
                       at exit which contains the state `s` above.
       @param [in] arc_weights_in  Weights on the arcs of the input FSA
       @param [out] deriv_out  Some derivative information at the output
                       will be written to here, which tells us how the weight
                       of the non-epsilon arc we created from the above
                       sub-graph varies as a function of the weights on the
                       arcs of the input FSA; it's a list
                       (input_arc_id, deriv) where, mathematically,
                       0 < deriv <= 1 (but we might still get exact zeros
                       due to limitations of floating point representation).
 */
static void TraceBackRmEpsilonsLogSum(
    std::map<int32_t, k2::LogSumTracebackState *> *curr_states,
    const float *arc_weights_in,
    std::vector<std::pair<int32_t, float>> *deriv_out) {
  CHECK_EQ(curr_states->size(), 1);
  deriv_out->clear();

  // as the input fsa is top-sorted, we traverse states in a reverse order so we
  // can process them when they already have correct backward_prob (all leaving
  // arcs have been processed).
  k2::LogSumTracebackState *state_ptr = curr_states->rbegin()->second;
  // In the standard forward-backward algorithm for HMMs this backward_prob
  // would, mathematically, be 0.0, but if we set it to the negative of the
  // forward prob we can avoid having to subtract the total log-prob
  // when we compute posterior/occupation probabilities for arcs.
  state_ptr->backward_prob = -state_ptr->forward_prob;
  while (!state_ptr->prev_elements.empty()) {
    double backward_prob = state_ptr->backward_prob;
    for (const auto &link : state_ptr->prev_elements) {
      auto arc_log_posterior =
          static_cast<float>(link.forward_prob + backward_prob);
      deriv_out->emplace_back(link.arc_index, expf(arc_log_posterior));
      k2::LogSumTracebackState *prev_state = link.prev_state.get();
      double new_backward_prob = backward_prob + arc_weights_in[link.arc_index];
      auto result = curr_states->emplace(prev_state->state_id, prev_state);
      if (result.second) {
        prev_state->backward_prob = new_backward_prob;
      } else {
        prev_state->backward_prob =
            k2::LogAdd(new_backward_prob, prev_state->backward_prob);
      }
    }
    // we have processed all entering arcs of state curr_states->rbegin(),
    // we'll remove it now. As std::map.erase() does not support passing a
    // reverse iterator, we here pass --end();
    curr_states->erase(--curr_states->end());
    CHECK(!curr_states->empty());
    state_ptr = curr_states->rbegin()->second;
  }
  // we have reached the state from which we are trying to remove epsilon arcs.
  CHECK_EQ(curr_states->size(), 1);
}

}  // namespace

namespace k2 {

// This function uses "Tarjan's strongly connected components algorithm"
// (see
// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
// to find co-accessible states in a single pass of the graph.
//
// The notations "lowlink", "dfnumber", and "onstack" are from the book
// "The design and analysis of computer algorithms", on page 192,
// Fig. 5.15 "Procedure to compute LOWLINK", written by John E. Hopcroft.
//
// http://www.openfst.org/doxygen/fst/html/connect_8h_source.html
// is used as a reference while implementing this function.
bool ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map) {
  CHECK_NOTNULL(state_map);

  state_map->clear();
  if (IsEmpty(fsa)) return true;

  auto num_states = fsa.NumStates();

  std::vector<bool> accessible(num_states, false);
  std::vector<bool> coaccessible(num_states, false);
  std::vector<int8_t> state_status(num_states, kNotVisited);

  // scc is short for "strongly connected component"
  // the following block of variables are for
  // "Tarjan's strongly connected components algorithm"
  //
  // Refer to the comment above the function for the meaning of them
  std::vector<int32_t> scc_stack;
  scc_stack.reserve(num_states);
  std::vector<bool> onstack(num_states, false);
  std::vector<int32_t> dfnumber(num_states,
                                std::numeric_limits<int32_t>::max());
  auto df_count = 0;
  std::vector<int32_t> lowlink(num_states, std::numeric_limits<int32_t>::max());

  accessible.front() = true;
  coaccessible.back() = true;

  std::stack<DfsState> stack;
  stack.push({0, fsa.arc_indexes[0], fsa.arc_indexes[1]});
  state_status[0] = kVisiting;

  dfnumber[0] = df_count;
  lowlink[0] = df_count;
  ++df_count;
  scc_stack.push_back(0);
  onstack[0] = true;

  // map order to state.
  // state 0 has the largest order, i.e., num_states - 1
  // final_state has the least order, i.e., 0
  std::vector<int32_t> order;
  order.reserve(num_states);
  bool is_acyclic = true;  // order and is_acyclic are for topological sort

  while (!stack.empty()) {
    auto &current_state = stack.top();

    if (current_state.arc_begin == current_state.arc_end) {
      // we have finished visiting this state
      auto state = current_state.state;  // get a copy since we will destroy it
      stack.pop();
      state_status[state] = kVisited;

      order.push_back(state);

      if (dfnumber[state] == lowlink[state]) {
        // this is the root of the strongly connected component
        bool scc_coaccessible = false;  // if any node in scc is co-accessible,
                                        // it will be set to true
        auto k = scc_stack.size() - 1;
        auto num_nodes = 0;  // number of nodes in the scc

        auto tmp = 0;
        do {
          tmp = scc_stack[k--];
          if (coaccessible[tmp]) scc_coaccessible = true;
          ++num_nodes;
        } while (tmp != state);

        // if this cycle is not removed in the output fsa
        // set is_acyclic to false
        if (num_nodes > 1 && scc_coaccessible) is_acyclic = false;

        // now pop scc_stack and set co-accessible of each node
        do {
          tmp = scc_stack.back();
          if (scc_coaccessible) coaccessible[tmp] = true;
          scc_stack.pop_back();
          onstack[tmp] = false;
        } while (tmp != state);
      }

      if (!stack.empty()) {
        // if it has a parent, set the parent's co-accessible flag
        auto &parent = stack.top();
        if (coaccessible[state]) coaccessible[parent.state] = true;

        ++parent.arc_begin;  // process the next child

        lowlink[parent.state] = std::min(lowlink[parent.state], lowlink[state]);
      }
      continue;
    }

    const auto &arc = fsa.arcs[current_state.arc_begin];
    auto next_state = arc.dest_state;
    auto status = state_status[next_state];
    switch (status) {
      case kNotVisited: {
        // a new discovered node
        state_status[next_state] = kVisiting;
        auto arc_begin = fsa.arc_indexes[next_state];
        stack.push({next_state, arc_begin, fsa.arc_indexes[next_state + 1]});

        dfnumber[next_state] = df_count;
        lowlink[next_state] = df_count;
        ++df_count;
        scc_stack.push_back(next_state);
        onstack[next_state] = true;

        if (accessible[current_state.state]) accessible[next_state] = true;
        break;
      }
      case kVisiting:
        // this is a back arc, which means there is a loop in the fsa;
        // but this loop may be removed in the output fsa
        //
        // Refer to the above book for what the meaning of back arc is
        lowlink[current_state.state] =
            std::min(lowlink[current_state.state], dfnumber[next_state]);

        if (coaccessible[next_state]) coaccessible[current_state.state] = true;
        ++current_state.arc_begin;  // go to the next arc
        break;
      case kVisited:
        // this is a forward or cross arc;
        if (dfnumber[next_state] < dfnumber[current_state.state] &&
            onstack[next_state])
          lowlink[current_state.state] =
              std::min(lowlink[current_state.state], dfnumber[next_state]);

        // update the co-accessible flag
        if (coaccessible[next_state]) coaccessible[current_state.state] = true;
        ++current_state.arc_begin;  // go to the next arc
        break;
      default:
        LOG(FATAL) << "Unreachable code is executed!";
        break;
    }
  }

  state_map->reserve(num_states);
  if (!is_acyclic) {
    for (auto i = 0; i != num_states; ++i) {
      if (accessible[i] && coaccessible[i]) state_map->push_back(i);
    }
    return false;
  }

  // now for the acyclic case,
  // we return a state_map of a topologically sorted fsa
  const auto rend = order.rend();
  for (auto rbegin = order.rbegin(); rbegin != rend; ++rbegin) {
    auto s = *rbegin;
    if (accessible[s] && coaccessible[s]) state_map->push_back(s);
  }
  return true;
}

bool Connect(const Fsa &a, Fsa *b, std::vector<int32_t> *arc_map /*=nullptr*/) {
  CHECK_NOTNULL(b);
  if (arc_map != nullptr) arc_map->clear();

  std::vector<int32_t> state_b_to_a;
  bool is_acyclic = ConnectCore(a, &state_b_to_a);
  if (state_b_to_a.empty()) return true;

  b->arc_indexes.resize(state_b_to_a.size() + 1);
  b->arcs.clear();
  b->arcs.reserve(a.arcs.size());

  if (arc_map != nullptr) {
    arc_map->clear();
    arc_map->reserve(a.arcs.size());
  }

  std::vector<int32_t> state_a_to_b(a.NumStates(), -1);

  auto num_states_b = b->NumStates();
  for (auto i = 0; i != num_states_b; ++i) {
    auto state_a = state_b_to_a[i];
    state_a_to_b[state_a] = i;
  }

  auto arc_begin = 0;
  auto arc_end = 0;

  for (auto i = 0; i != num_states_b; ++i) {
    auto state_a = state_b_to_a[i];
    arc_begin = a.arc_indexes[state_a];
    arc_end = a.arc_indexes[state_a + 1];

    b->arc_indexes[i] = static_cast<int32_t>(b->arcs.size());
    for (; arc_begin != arc_end; ++arc_begin) {
      auto arc = a.arcs[arc_begin];
      auto dest_state = arc.dest_state;
      auto state_b = state_a_to_b[dest_state];

      if (state_b < 0) continue;  // dest_state is unreachable

      arc.src_state = i;
      arc.dest_state = state_b;
      b->arcs.push_back(arc);
      if (arc_map != nullptr) arc_map->push_back(arc_begin);
    }
  }
  b->arc_indexes[num_states_b] = b->arc_indexes[num_states_b - 1];
  return is_acyclic;
}

void RmEpsilonsPrunedMax(const WfsaWithFbWeights &a, float beam, Fsa *b,
                         std::vector<std::vector<int32_t>> *arc_derivs) {
  CHECK_EQ(a.weight_type, kMaxWeight);
  CHECK_GT(beam, 0);
  CHECK_NOTNULL(b);
  CHECK_NOTNULL(arc_derivs);
  b->arc_indexes.clear();
  b->arcs.clear();
  arc_derivs->clear();

  const auto &fsa = a.fsa;
  if (IsEmpty(fsa)) return;
  int32_t num_states_a = fsa.NumStates();
  int32_t final_state = fsa.FinalState();
  const auto &arcs_a = fsa.data;
  const float *arc_weights_a = a.arc_weights;

  // identify all states that should be kept
  std::vector<char> non_eps_in(num_states_a, 0);
  non_eps_in[0] = 1;
  for (const auto &arc : fsa) {
    // We suppose the input fsa `a` is top-sorted, but only check this in DEBUG
    // time.
    DCHECK_GE(arc.dest_state, arc.src_state);
    if (arc.label != kEpsilon) non_eps_in[arc.dest_state] = 1;
  }

  // remap state id
  std::vector<int32_t> state_map_a2b(num_states_a, -1);
  int32_t num_states_b = 0;
  for (int32_t i = 0; i != num_states_a; ++i) {
    if (non_eps_in[i] == 1) state_map_a2b[i] = num_states_b++;
  }
  b->arc_indexes.reserve(num_states_b + 1);
  int32_t arc_num_b = 0;

  const double *forward_state_weights = a.ForwardStateWeights();
  const double *backward_state_weights = a.BackwardStateWeights();
  const double best_weight = forward_state_weights[final_state] - beam;
  for (int32_t i = 0; i != num_states_a; ++i) {
    if (non_eps_in[i] != 1) continue;
    b->arc_indexes.push_back(arc_num_b);
    int32_t curr_state_b = state_map_a2b[i];
    // as the input FSA is top-sorted, we use a map here so we can process
    // states when they already have the best cost they are going to get
    std::map<int32_t, double>
        local_forward_weights;  // state -> local_forward_state_weights of this
                                // state
    // state -> (src_state, arc_index) entering this state which contributes to
    // `local_forward_weights` of this state.
    std::unordered_map<int32_t, std::pair<int32_t, int32_t>>
        local_backward_arcs;
    local_forward_weights.emplace(i, forward_state_weights[i]);
    // `-1` means we have traced back to current state `i`
    local_backward_arcs.emplace(i, std::make_pair(i, -1));
    while (!local_forward_weights.empty()) {
      std::pair<int32_t, double> curr_local_forward_weights =
          *(local_forward_weights.begin());
      local_forward_weights.erase(local_forward_weights.begin());
      int32_t state = curr_local_forward_weights.first;

      int32_t arc_end = fsa.indexes[state + 1];
      for (int32_t arc_index = fsa.indexes[state]; arc_index != arc_end;
           ++arc_index) {
        int32_t next_state = arcs_a[arc_index].dest_state;
        int32_t label = arcs_a[arc_index].label;
        double next_weight =
            curr_local_forward_weights.second + arc_weights_a[arc_index];
        if (next_weight + backward_state_weights[next_state] >= best_weight) {
          if (label == kEpsilon) {
            auto result =
                local_forward_weights.emplace(next_state, next_weight);
            if (result.second) {
              local_backward_arcs[next_state] =
                  std::make_pair(state, arc_index);
            } else {
              if (next_weight > result.first->second) {
                result.first->second = next_weight;
                local_backward_arcs[next_state] =
                    std::make_pair(state, arc_index);
              }
            }
          } else {
            b->arcs.emplace_back(curr_state_b, state_map_a2b[next_state],
                                 label);
            std::vector<int32_t> curr_arc_deriv;
            std::pair<int32_t, int32_t> curr_backward_arc{state, arc_index};
            auto *backward_arc = &curr_backward_arc;
            while (backward_arc->second != -1) {
              curr_arc_deriv.push_back(backward_arc->second);
              backward_arc = &(local_backward_arcs[backward_arc->first]);
            }
            std::reverse(curr_arc_deriv.begin(), curr_arc_deriv.end());
            arc_derivs->emplace_back(std::move(curr_arc_deriv));
            ++arc_num_b;
          }
        }
      }
    }
  }
  // duplicate of final state
  b->arc_indexes.push_back(b->arc_indexes.back());
}

void RmEpsilonsPrunedLogSum(
    const WfsaWithFbWeights &a, float beam, Fsa *b,
    std::vector<float> *b_arc_weights,
    std::vector<std::vector<std::pair<int32_t, float>>> *arc_derivs) {
  CHECK_GT(beam, 0);
  CHECK_NOTNULL(b);
  CHECK_NOTNULL(b_arc_weights);
  CHECK_NOTNULL(arc_derivs);
  b->arc_indexes.clear();
  b->arcs.clear();
  b_arc_weights->clear();
  arc_derivs->clear();

  const auto &fsa = a.fsa;
  if (IsEmpty(fsa)) return;
  int32_t num_states_a = fsa.NumStates();
  int32_t final_state = fsa.FinalState();
  const auto &arcs_a = fsa.data;
  const float *arc_weights_a = a.arc_weights;

  // identify all states that should be kept
  std::vector<char> non_eps_in(num_states_a, 0);
  non_eps_in[0] = 1;
  for (const auto &arc : fsa) {
    // We suppose the input fsa `a` is top-sorted, but only check this in DEBUG
    // time.
    DCHECK_GE(arc.dest_state, arc.src_state);
    if (arc.label != kEpsilon) non_eps_in[arc.dest_state] = 1;
  }

  // remap state id
  std::vector<int32_t> state_map_a2b(num_states_a, -1);
  int32_t num_states_b = 0;
  for (int32_t i = 0; i != num_states_a; ++i) {
    if (non_eps_in[i] == 1) state_map_a2b[i] = num_states_b++;
  }
  b->arc_indexes.reserve(num_states_b + 1);
  int32_t arc_num_b = 0;

  const double *forward_state_weights = a.ForwardStateWeights();
  const double *backward_state_weights = a.BackwardStateWeights();
  const double best_weight = forward_state_weights[final_state] - beam;
  for (int32_t i = 0; i != num_states_a; ++i) {
    if (non_eps_in[i] != 1) continue;
    b->arc_indexes.push_back(arc_num_b);
    int32_t curr_state_b = state_map_a2b[i];
    // as the input FSA is top-sorted, we use a set here so we can process
    // states when they already have costs over all paths they are going to get
    std::set<int32_t> qstates;
    std::unordered_map<int32_t, std::shared_ptr<LogSumTracebackState>>
        traceback_states;  // state -> LogSumTracebackState of this state
    std::shared_ptr<LogSumTracebackState> start_state(
        new LogSumTracebackState(i, forward_state_weights[i]));
    double start_forward_weights = start_state->forward_prob;
    traceback_states.emplace(i, start_state);
    qstates.insert(i);
    while (!qstates.empty()) {
      int32_t state = *(qstates.begin());
      qstates.erase(qstates.begin());

      const auto &curr_traceback_state = traceback_states[state];
      double curr_forward_weights = curr_traceback_state->forward_prob;
      int32_t arc_end = fsa.indexes[state + 1];
      for (int32_t arc_index = fsa.indexes[state]; arc_index != arc_end;
           ++arc_index) {
        int32_t next_state = arcs_a[arc_index].dest_state;
        int32_t label = arcs_a[arc_index].label;
        float curr_arc_weight = arc_weights_a[arc_index];
        double next_weight = curr_forward_weights + curr_arc_weight;
        if (next_weight + backward_state_weights[next_state] >= best_weight) {
          if (label == kEpsilon) {
            auto result = traceback_states.emplace(next_state, nullptr);
            if (result.second) {
              result.first->second = std::make_shared<LogSumTracebackState>(
                  next_state, curr_traceback_state, arc_index, curr_arc_weight);
              qstates.insert(next_state);
            } else {
              result.first->second->Accept(curr_traceback_state, arc_index,
                                           curr_arc_weight);
            }
          } else {
            b->arcs.emplace_back(curr_state_b, state_map_a2b[next_state],
                                 label);
            b_arc_weights->push_back(curr_forward_weights + curr_arc_weight -
                                     start_forward_weights);

            std::vector<std::pair<int32_t, float>> curr_arc_deriv;
            std::map<int32_t, LogSumTracebackState *> curr_states;
            curr_states.emplace(state, curr_traceback_state.get());
            TraceBackRmEpsilonsLogSum(&curr_states, arc_weights_a,
                                      &curr_arc_deriv);
            std::reverse(curr_arc_deriv.begin(), curr_arc_deriv.end());
            // push derivs info of current arc
            curr_arc_deriv.emplace_back(arc_index, 1);
            arc_derivs->emplace_back(std::move(curr_arc_deriv));
            ++arc_num_b;
          }
        }
      }
    }
  }
  // duplicate of final state
  b->arc_indexes.push_back(b->arc_indexes.back());
}

bool Intersect(const Fsa &a, const Fsa &b, Fsa *c,
               std::vector<int32_t> *arc_map_a /*= nullptr*/,
               std::vector<int32_t> *arc_map_b /*= nullptr*/) {
  CHECK_NOTNULL(c);
  c->arc_indexes.clear();
  c->arcs.clear();
  if (arc_map_a != nullptr) arc_map_a->clear();
  if (arc_map_b != nullptr) arc_map_b->clear();

  if (IsEmpty(a) || IsEmpty(b)) return true;
  if (!IsArcSorted(a) || !IsArcSorted(b)) return false;
  // either `a` or `b` must be epsilon-free
  if (!IsEpsilonFree(a) && !IsEpsilonFree(b)) return false;

  int32_t final_state_a = a.NumStates() - 1;
  int32_t final_state_b = b.NumStates() - 1;
  const auto arc_a_begin = a.arcs.begin();
  const auto arc_b_begin = b.arcs.begin();
  using ArcIterator = std::vector<Arc>::const_iterator;

  const int32_t final_state_c = -1;  // just as a placeholder
  // no corresponding arc mapping from `c` to `a` or `c` to `b`
  const int32_t arc_map_none = -1;
  auto &arc_indexes_c = c->arc_indexes;
  auto &arcs_c = c->arcs;

  // map state pair to unique id
  std::unordered_map<StatePair, int32_t, PairHash> state_pair_map;
  std::queue<StatePair> qstates;
  qstates.push({0, 0});
  state_pair_map.insert({{0, 0}, 0});
  state_pair_map.insert({{final_state_a, final_state_b}, final_state_c});
  int32_t state_index_c = 0;
  while (!qstates.empty()) {
    arc_indexes_c.push_back(static_cast<int32_t>(arcs_c.size()));

    auto curr_state_pair = qstates.front();
    qstates.pop();
    // as we have inserted `curr_state_pair` before.
    int32_t curr_state_index = state_pair_map[curr_state_pair];

    auto state_a = curr_state_pair.first;
    auto a_arc_iter_begin = arc_a_begin + a.arc_indexes[state_a];
    auto a_arc_iter_end = arc_a_begin + a.arc_indexes[state_a + 1];
    auto state_b = curr_state_pair.second;
    auto b_arc_iter_begin = arc_b_begin + b.arc_indexes[state_b];
    auto b_arc_iter_end = arc_b_begin + b.arc_indexes[state_b + 1];

    // As both `a` and `b` are arc-sorted, we first process epsilon arcs.
    // Noted that at most one for-loop below will really run as either `a` or
    // `b` is epsilon-free.
    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      if (kEpsilon != a_arc_iter_begin->label) break;

      StatePair new_state{a_arc_iter_begin->dest_state, state_b};
      int32_t new_state_index = InsertIntersectionState(
          new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_c.emplace_back(curr_state_index, new_state_index, kEpsilon);
      if (arc_map_a != nullptr)
        arc_map_a->push_back(
            static_cast<int32_t>(a_arc_iter_begin - arc_a_begin));
      if (arc_map_b != nullptr) arc_map_b->push_back(arc_map_none);
    }
    for (; b_arc_iter_begin != b_arc_iter_end; ++b_arc_iter_begin) {
      if (kEpsilon != b_arc_iter_begin->label) break;
      StatePair new_state{state_a, b_arc_iter_begin->dest_state};
      int32_t new_state_index = InsertIntersectionState(
          new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_c.emplace_back(curr_state_index, new_state_index, kEpsilon);
      if (arc_map_a != nullptr) arc_map_a->push_back(arc_map_none);
      if (arc_map_b != nullptr)
        arc_map_b->push_back(
            static_cast<int32_t>(b_arc_iter_begin - arc_b_begin));
    }

    // as both `a` and `b` are arc-sorted, we will iterate over the state with
    // less number of arcs.
    bool swapped = false;
    if ((a_arc_iter_end - a_arc_iter_begin) >
        (b_arc_iter_end - b_arc_iter_begin)) {
      std::swap(a_arc_iter_begin, b_arc_iter_begin);
      std::swap(a_arc_iter_end, b_arc_iter_end);
      swapped = true;
    }

    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      Arc curr_a_arc = *a_arc_iter_begin;  // copy here as we may swap later
      auto b_arc_range =
          std::equal_range(b_arc_iter_begin, b_arc_iter_end, curr_a_arc,
                           [](const Arc &left, const Arc &right) {
                             return left.label < right.label;
                           });
      for (auto it_b = b_arc_range.first; it_b != b_arc_range.second; ++it_b) {
        Arc curr_b_arc = *it_b;
        if (swapped) std::swap(curr_a_arc, curr_b_arc);
        StatePair new_state{curr_a_arc.dest_state, curr_b_arc.dest_state};
        int32_t new_state_index = InsertIntersectionState(
            new_state, &state_index_c, &qstates, &state_pair_map);
        arcs_c.emplace_back(curr_state_index, new_state_index,
                            curr_a_arc.label);

        auto curr_arc_index_a = static_cast<int32_t>(
            a_arc_iter_begin - (swapped ? arc_b_begin : arc_a_begin));
        auto curr_arc_index_b =
            static_cast<int32_t>(it_b - (swapped ? arc_a_begin : arc_b_begin));
        if (swapped) std::swap(curr_arc_index_a, curr_arc_index_b);
        if (arc_map_a != nullptr) arc_map_a->push_back(curr_arc_index_a);
        if (arc_map_b != nullptr) arc_map_b->push_back(curr_arc_index_b);
      }
    }
  }

  // push final state
  arc_indexes_c.push_back(static_cast<int32_t>(arcs_c.size()));
  ++state_index_c;
  // then replace `final_state_c` with the real index of final state of `c`
  for (auto &arc : arcs_c) {
    if (arc.dest_state == final_state_c) arc.dest_state = state_index_c;
  }
  // push a duplicate of final state, see the constructor of `Fsa` in
  // `k2/csrc/fsa.h`
  arc_indexes_c.emplace_back(arc_indexes_c.back());
  return true;
}

void ArcSort(const Fsa &a, Fsa *b,
             std::vector<int32_t> *arc_map /*= nullptr*/) {
  CHECK_NOTNULL(b);
  b->arc_indexes = a.arc_indexes;
  b->arcs.clear();
  b->arcs.reserve(a.arcs.size());
  if (arc_map != nullptr) arc_map->clear();

  using ArcWithIndex = std::pair<Arc, int32_t>;
  std::vector<int32_t> indexes(a.arcs.size());  // index mapping
  std::iota(indexes.begin(), indexes.end(), 0);
  const auto arc_begin_iter = a.arcs.begin();
  const auto index_begin_iter = indexes.begin();
  // we will not process the final state as it has no arcs leaving it.
  int32_t final_state = a.NumStates() - 1;
  for (int32_t state = 0; state < final_state; ++state) {
    int32_t begin = a.arc_indexes[state];
    // as non-empty fsa `a` contains at least two states,
    // we can always access `state + 1` validly.
    int32_t end = a.arc_indexes[state + 1];
    std::vector<ArcWithIndex> arc_range_to_be_sorted;
    arc_range_to_be_sorted.reserve(end - begin);
    std::transform(arc_begin_iter + begin, arc_begin_iter + end,
                   index_begin_iter + begin,
                   std::back_inserter(arc_range_to_be_sorted),
                   [](const Arc &arc, int32_t index) -> ArcWithIndex {
                     return std::make_pair(arc, index);
                   });
    std::sort(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
              [](const ArcWithIndex &left, const ArcWithIndex &right) {
                return left.first < right.first;  // sort on arc
              });
    // copy index mappings back to `indexes`
    std::transform(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
                   index_begin_iter + begin,
                   [](const ArcWithIndex &v) { return v.second; });
    // move-copy sorted arcs to `b`
    std::transform(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
                   std::back_inserter(b->arcs),
                   [](ArcWithIndex &v) { return v.first; });
  }
  if (arc_map != nullptr) arc_map->swap(indexes);
}

bool TopSort(const Fsa &a, Fsa *b,
             std::vector<int32_t> *state_map /*= nullptr*/) {
  CHECK_NOTNULL(b);
  b->arc_indexes.clear();
  b->arcs.clear();

  if (state_map != nullptr) state_map->clear();

  if (IsEmpty(a)) return true;
  if (!IsConnected(a)) return false;

  auto num_states = a.NumStates();
  auto final_state = num_states - 1;
  std::vector<int8_t> state_status(num_states, kNotVisited);

  // map order to state.
  // state 0 has the largest order, i.e., num_states - 1
  // final_state has the least order, i.e., 0
  std::vector<int32_t> order;
  order.reserve(num_states);

  bool is_acyclic = IsAcyclic(a, &order);

  if (!is_acyclic) return false;

  std::vector<int32_t> state_a_to_b(num_states);
  for (auto i = 0; i != num_states; ++i) {
    state_a_to_b[order[num_states - 1 - i]] = i;
  }

  // start state maps to start state
  CHECK_EQ(state_a_to_b.front(), 0);
  // final state maps to final state
  CHECK_EQ(state_a_to_b.back(), final_state);

  b->arcs.reserve(a.arc_indexes.size());
  b->arc_indexes.resize(num_states);

  int32_t arc_begin;
  int32_t arc_end;
  for (auto state_b = 0; state_b != num_states; ++state_b) {
    auto state_a = order[num_states - 1 - state_b];
    arc_begin = a.arc_indexes[state_a];
    arc_end = a.arc_indexes[state_a + 1];

    b->arc_indexes[state_b] = static_cast<int32_t>(b->arcs.size());
    for (; arc_begin != arc_end; ++arc_begin) {
      auto arc = a.arcs[arc_begin];
      arc.src_state = state_b;
      arc.dest_state = state_a_to_b[arc.dest_state];
      b->arcs.push_back(arc);
    }
  }
  if (state_map != nullptr) {
    std::reverse(order.begin(), order.end());
    state_map->swap(order);
  }
  b->arc_indexes.emplace_back(b->arc_indexes.back());
  return true;
}

void CreateFsa(const std::vector<Arc> &arcs, Fsa *fsa,
               std::vector<int32_t> *arc_map /*=null_ptr*/) {
  CHECK_NOTNULL(fsa);
  fsa->arc_indexes.clear();
  fsa->arcs.clear();

  if (arcs.empty()) return;

  using ArcWithIndex = std::pair<Arc, int32_t>;
  int arc_id = 0;
  std::vector<std::vector<ArcWithIndex>> vec;
  for (const auto &arc : arcs) {
    auto src_state = arc.src_state;
    auto dest_state = arc.dest_state;
    auto new_size = std::max(src_state, dest_state);
    if (new_size >= vec.size()) vec.resize(new_size + 1);
    vec[src_state].push_back({arc, arc_id++});
  }

  std::stack<DfsState> stack;
  std::vector<char> state_status(vec.size(), kNotVisited);
  std::vector<int32_t> order;

  auto num_states = static_cast<int32_t>(vec.size());
  for (auto i = 0; i != num_states; ++i) {
    if (state_status[i] == kVisited) continue;
    stack.push({i, 0, static_cast<int32_t>(vec[i].size())});
    state_status[i] = kVisiting;
    while (!stack.empty()) {
      auto &current_state = stack.top();
      auto state = current_state.state;

      if (current_state.arc_begin == current_state.arc_end) {
        state_status[state] = kVisited;
        order.push_back(state);
        stack.pop();
        continue;
      }

      const auto &arc = vec[state][current_state.arc_begin].first;
      auto next_state = arc.dest_state;
      auto status = state_status[next_state];
      switch (status) {
        case kNotVisited:
          state_status[next_state] = kVisiting;
          stack.push(
              {next_state, 0, static_cast<int32_t>(vec[next_state].size())});
          ++current_state.arc_begin;
          break;
        case kVisiting:
          LOG(FATAL) << "there is a cycle: " << state << " -> " << next_state;
          break;
        case kVisited:
          ++current_state.arc_begin;
          break;
        default:
          LOG(FATAL) << "Unreachable code is executed!";
          break;
      }
    }
  }

  CHECK_EQ(num_states, static_cast<int32_t>(order.size()));

  std::reverse(order.begin(), order.end());

  fsa->arc_indexes.resize(num_states + 1);
  fsa->arcs.reserve(arcs.size());
  std::vector<int32_t> arc_map_out;
  arc_map_out.reserve(arcs.size());

  std::vector<int32_t> old_to_new(num_states);
  for (auto i = 0; i != num_states; ++i) old_to_new[order[i]] = i;

  for (auto i = 0; i != num_states; ++i) {
    auto old_state = order[i];
    fsa->arc_indexes[i] = static_cast<int32_t>(fsa->arcs.size());
    for (auto arc_with_index : vec[old_state]) {
      auto &arc = arc_with_index.first;
      arc.src_state = i;
      arc.dest_state = old_to_new[arc.dest_state];
      fsa->arcs.push_back(arc);
      arc_map_out.push_back(arc_with_index.second);
    }
  }

  fsa->arc_indexes.back() = static_cast<int32_t>(fsa->arcs.size());
  if (arc_map != nullptr) arc_map->swap(arc_map_out);
}

float DeterminizePrunedLogSum(
    const WfsaWithFbWeights &a, float beam, int64_t max_step, Fsa *b,
    std::vector<float> *b_arc_weights,
    std::vector<std::vector<std::pair<int32_t, float>>> *arc_derivs) {
  CHECK_EQ(a.weight_type, kLogSumWeight);
  return DeterminizePrunedTpl<LogSumTracebackState>(a, beam, max_step, b,
                                                    b_arc_weights, arc_derivs);
}

float DeterminizePrunedMax(const WfsaWithFbWeights &a, float beam,
                           int64_t max_step, Fsa *b,
                           std::vector<float> *b_arc_weights,
                           std::vector<std::vector<int32_t>> *arc_derivs) {
  CHECK_EQ(a.weight_type, kMaxWeight);
  return DeterminizePrunedTpl<MaxTracebackState>(a, beam, max_step, b,
                                                 b_arc_weights, arc_derivs);
}

}  // namespace k2
