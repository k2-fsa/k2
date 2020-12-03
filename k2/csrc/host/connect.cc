/**
 * @brief
 * connect
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/connect.h"

#include <algorithm>
#include <limits>
#include <stack>
#include <unordered_map>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {

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
  NVTX_RANGE(K2_FUNC);
  using dfs::DfsState;
  using dfs::kNotVisited;
  using dfs::kVisited;
  using dfs::kVisiting;
  K2_CHECK_NE(state_map, nullptr);

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
  stack.push({0, fsa.indexes[0], fsa.indexes[1]});
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

    const auto &arc = fsa.data[current_state.arc_begin];
    auto next_state = arc.dest_state;
    auto status = state_status[next_state];
    switch (status) {
      case kNotVisited: {
        // a new discovered node
        state_status[next_state] = kVisiting;
        auto arc_begin = fsa.indexes[next_state];
        stack.push({next_state, arc_begin, fsa.indexes[next_state + 1]});

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
        K2_LOG(FATAL) << "Unreachable code is executed!";
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

void Connection::GetSizes(Array2Size<int32_t> *fsa_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;
  no_accessible_state_ = false;
  arc_indexes_.clear();
  arcs_.clear();
  arc_map_.clear();

  std::vector<int32_t> state_out_to_in;
  is_acyclic_ = ConnectCore(fsa_in_, &state_out_to_in);
  if (state_out_to_in.empty()) {
    no_accessible_state_ = true;
    return;
  }

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

bool Connection::GetOutput(Fsa *fsa_out, int32_t *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  if (no_accessible_state_) return true;

  // output FSA
  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_EQ(arc_indexes_.size(), fsa_out->size1 + 1);
  std::copy(arc_indexes_.begin(), arc_indexes_.end(), fsa_out->indexes);
  K2_CHECK_EQ(arcs_.size(), fsa_out->size2);
  std::copy(arcs_.begin(), arcs_.end(), fsa_out->data);

  // output arc map
  if (arc_map != nullptr) std::copy(arc_map_.begin(), arc_map_.end(), arc_map);

  return is_acyclic_;
}

}  // namespace k2host
