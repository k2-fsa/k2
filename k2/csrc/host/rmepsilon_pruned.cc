/**
 * @brief
 * rmepsilon
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/rmepsilon.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace {

/**
   This function identifies all states that should be kept in the input FSA
   (those kept states will be the states in the output FSA) and then maps
   input FSA's states to output FSA's states. A state should be kept if
   there is at least one labeled arc (arc.label != epsilon) entering this state.
     @param [in] fsa_in       The input FSA
     @param [out] non_eps_in  Indexed by state in `fsa_in`,
                              `non_eps_in[state] == 1` means this state should
                              be kept.
     @param [out] state_map   Indexed by state in `fsa_in`, maps states in
                              the input FSA to states in the output FSA.
                              `state_map[state] = -1` means this state will not
                              be kept and thus there's no corresponding state
                              in the output FSA.

   Returns the number of kept states which is num_states of the output FSA.
 */
static int32_t MapStates(const k2host::Fsa &fsa_in,
                         std::vector<char> *non_eps_in,
                         std::vector<int32_t> *state_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(non_eps_in, nullptr);
  K2_CHECK_NE(state_map, nullptr);

  int32_t num_states_in = fsa_in.NumStates();
  K2_CHECK_EQ(num_states_in, non_eps_in->size());
  K2_CHECK_EQ(num_states_in, state_map->size());

  // identify all states that should be kept
  auto &non_eps_in_values = *non_eps_in;
  auto &state_map_values = *state_map;
  // the start state and final state should always be kept.
  non_eps_in_values[0] = 1;
  non_eps_in_values[fsa_in.FinalState()] = 1;
  for (const auto &arc : fsa_in) {
    // We suppose the input fsa is top-sorted, but only check this in DEBUG
    // time.
    K2_DCHECK_GE(arc.dest_state, arc.src_state);
    if (arc.label != k2host::kEpsilon) non_eps_in_values[arc.dest_state] = 1;
  }

  // map state id
  int32_t num_states_out = 0;
  for (int32_t i = 0; i != num_states_in; ++i) {
    if (non_eps_in_values[i] == 1) state_map_values[i] = num_states_out++;
  }
  return num_states_out;
}

/**
   A TraceBack function exists for MaxTracebackState or LogSumTracebackState;
   it's used in EpsilonsRemover.  It finds derivative information for all arcs
   in a sub-graph. Generally, in EpsilonsRemover, we actually get a sub-graph
   when we find a non-epsilon arc starting from a particular state `s` (from
   which we are trying to remove epsilon arcs). All leaving arcs of all states
   in this sub-graph are epsilon arcs except the last one. Then, from the last
   state, we need to trace back to state `s` to find the derivative information
   for all epsilon arcs in this graph.
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
       @param [in] arcs_in  Array of arcs of the FSA, to look up weight
       @param [in] last_arc_index  The arc index of the last arc in the
                                   sub-graph above, it's a labeled arc,
                                   i.e. the arc's label is not epsilon.
       @param [out] deriv_out  Some derivative information at the output
                       will be written to here, which tells us how the weight
                       of the non-epsilon arc we created from the above
                       sub-graph varies as a function of the weights on the
                       arcs of the input FSA; it's a list
                       (input_arc_id, deriv) where, mathematically,
                       0 < deriv <= 1 (but we might still get exact zeros
                       due to limitations of floating point representation).
 */
static void TraceBackRmEpsilons(
    std::map<int32_t, k2host::LogSumTracebackState *> *curr_states,
    const k2host::Arc *arcs_in, int32_t last_arc_index,
    std::vector<std::pair<int32_t, float>> *deriv_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(curr_states->size(), 1);
  deriv_out->clear();
  // push derivative info of the last arc
  deriv_out->emplace_back(last_arc_index, 1);

  // as the input fsa is top-sorted, we traverse states in a reverse order so we
  // can process them when they already have correct backward_prob (all leaving
  // arcs have been processed).
  k2host::LogSumTracebackState *state_ptr = curr_states->rbegin()->second;
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
      k2host::LogSumTracebackState *prev_state = link.prev_state.get();
      double new_backward_prob = backward_prob + arcs_in[link.arc_index].weight;
      auto result = curr_states->emplace(prev_state->state_id, prev_state);
      if (result.second) {
        prev_state->backward_prob = new_backward_prob;
      } else {
        prev_state->backward_prob =
            k2host::LogAdd(new_backward_prob, prev_state->backward_prob);
      }
    }
    // we have processed all entering arcs of state curr_states->rbegin(),
    // we'll remove it now. As std::map.erase() does not support passing a
    // reverse iterator, we here pass --end();
    curr_states->erase(--curr_states->end());
    K2_CHECK(!curr_states->empty());
    state_ptr = curr_states->rbegin()->second;
  }
  // we have reached the state from which we are trying to remove epsilon arcs.
  K2_CHECK_EQ(curr_states->size(), 1);
}

/**
   The TraceBack function for MaxTracebackState. See documentation of Traceback
   for LogSumTracebackState above. This version is simpler as we only keep the
   best path among alternative epsilon paths when traversing paths in
   EmpsilonRemover for MaxTracebackState, so here we just trace back the best
   path to get the derivative information.
 */
static void TraceBackRmEpsilons(
    std::map<int32_t, k2host::MaxTracebackState *> *curr_states,
    const k2host::Arc *unused,  // arcs_in, unused
    int32_t last_arc_index, std::vector<int32_t> *deriv_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(curr_states->size(), 1);
  deriv_out->clear();
  // push derivative info of the last arc
  deriv_out->push_back(last_arc_index);

  k2host::MaxTracebackState *state_ptr = curr_states->begin()->second;
  while (state_ptr->prev_state != nullptr) {
    deriv_out->push_back(state_ptr->arc_id);
    state_ptr = state_ptr->prev_state.get();
  }
}
}  // namespace

namespace k2host {

template <typename TracebackState>
void EpsilonsRemoverPruned<TracebackState>::GetSizes(
    Array2Size<int32_t> *fsa_size, Array2Size<int32_t> *arc_derivs_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  K2_CHECK_NE(arc_derivs_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;
  arc_derivs_size->size1 = arc_derivs_size->size2 = 0;

  arc_indexes_.clear();
  arcs_.clear();
  arc_derivs_.clear();

  const auto &fsa = fsa_in_.fsa;
  if (IsEmpty(fsa)) return;
  int32_t num_states_in = fsa.NumStates();
  int32_t final_state_in = fsa.FinalState();
  const auto &arcs_in = fsa.data + fsa.indexes[0];

  // identify all states that should be kept
  std::vector<char> non_eps_in(num_states_in, 0);
  std::vector<int32_t> state_map(num_states_in, -1);
  int32_t num_states_out = MapStates(fsa, &non_eps_in, &state_map);

  arc_indexes_.reserve(num_states_out + 1);
  int32_t arc_num_out = 0;
  int32_t derivs_num_out = 0;
  const double *forward_state_weights = fsa_in_.ForwardStateWeights();
  const double *backward_state_weights = fsa_in_.BackwardStateWeights();
  const double best_weight = forward_state_weights[final_state_in] - beam_;
  for (int32_t i = 0; i != num_states_in; ++i) {
    if (non_eps_in[i] != 1) continue;
    arc_indexes_.push_back(arc_num_out);
    int32_t curr_state_out = state_map[i];

    // as the input FSA is top-sorted, we use a set here so we can process
    // states when they already have costs over all paths they are going to get
    std::set<int32_t> qstates;
    std::unordered_map<int32_t, std::shared_ptr<TracebackState>>
        traceback_states;  // state -> TracebackState of this state
    std::shared_ptr<TracebackState> start_state(
        new TracebackState(i, forward_state_weights[i]));
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
        const int32_t curr_arc_index = arc_index - fsa.indexes[0];
        int32_t next_state = arcs_in[curr_arc_index].dest_state;
        int32_t label = arcs_in[curr_arc_index].label;
        float curr_arc_weight = arcs_in[curr_arc_index].weight;
        double next_weight = curr_forward_weights + curr_arc_weight;
        if (next_weight + backward_state_weights[next_state] >= best_weight) {
          if (label == kEpsilon) {
            auto result = traceback_states.emplace(next_state, nullptr);
            if (result.second) {
              result.first->second = std::make_shared<TracebackState>(
                  next_state, curr_traceback_state, curr_arc_index,
                  curr_arc_weight);
              qstates.insert(next_state);
            } else {
              result.first->second->Accept(curr_traceback_state, curr_arc_index,
                                           curr_arc_weight);
            }
          } else {
            float arc_weight =
                curr_forward_weights + curr_arc_weight - start_forward_weights;
            arcs_.emplace_back(curr_state_out, state_map[next_state], label,
                               arc_weight);

            std::vector<typename TracebackState::DerivType> curr_arc_deriv;
            std::map<int32_t, TracebackState *> curr_states;
            curr_states.emplace(state, curr_traceback_state.get());
            TraceBackRmEpsilons(&curr_states, arcs_in, curr_arc_index,
                                &curr_arc_deriv);
            std::reverse(curr_arc_deriv.begin(), curr_arc_deriv.end());
            derivs_num_out += curr_arc_deriv.size();
            arc_derivs_.emplace_back(std::move(curr_arc_deriv));
            ++arc_num_out;
          }
        }
      }
    }
  }
  // duplicate of final state
  arc_indexes_.push_back(arc_indexes_.back());

  fsa_size->size1 = num_states_out;
  fsa_size->size2 = arcs_.size();
  arc_derivs_size->size1 = arcs_.size();
  arc_derivs_size->size2 = derivs_num_out;
}

template <typename TracebackState>
void EpsilonsRemoverPruned<TracebackState>::GetOutput(
    Fsa *fsa_out,
    Array2<typename TracebackState::DerivType *, int32_t> *arc_derivs) {
  NVTX_RANGE(K2_FUNC);
  if (IsEmpty(fsa_in_.fsa)) return;

  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_NE(arc_derivs, nullptr);

  // output FSA
  K2_CHECK_EQ(arc_indexes_.size(), fsa_out->size1 + 1);
  std::copy(arc_indexes_.begin(), arc_indexes_.end(), fsa_out->indexes);
  K2_CHECK_EQ(arcs_.size(), fsa_out->size2);
  std::copy(arcs_.begin(), arcs_.end(), fsa_out->data);

  // output arc derivative information
  K2_CHECK_EQ(arc_derivs_.size(), arc_derivs->size1);
  int32_t num_derivs = 0;
  for (int32_t i = 0; i != arc_derivs->size1; ++i) {
    arc_derivs->indexes[i] = num_derivs;
    const auto &curr_arc_deriv = arc_derivs_[i];
    std::copy(curr_arc_deriv.begin(), curr_arc_deriv.end(),
              arc_derivs->data + num_derivs);
    num_derivs += curr_arc_deriv.size();
  }
  arc_derivs->indexes[arc_derivs->size1] = num_derivs;
}

// explicit instantiation here
template class EpsilonsRemoverPruned<MaxTracebackState>;
template class EpsilonsRemoverPruned<LogSumTracebackState>;

}  // namespace k2host
