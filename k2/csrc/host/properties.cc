/**
 * @brief
 * properties
 *
 * @copyright
 * Copyright (c)  2020  Haowen Qiu
 *                      Daniel Povey
 *                      Mahsa Yarmohammadi
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/properties.h"

#include <algorithm>
#include <stack>
#include <unordered_set>
#include <vector>

#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {

bool IsValid(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  if (IsEmpty(fsa)) return true;
  int32_t num_states = fsa.NumStates();
  // Nonempty fsa contains at least two states,
  // the start state and the final state.
  if (num_states < 2) return false;
  int32_t state = 0;
  int32_t final_state = num_states - 1;
  // the number of arcs in one state
  int32_t num_arcs = 0;
  for (const auto &arc : fsa) {
    // only kFinalSymbol arcs enter the final state
    if (arc.dest_state == final_state && arc.label != kFinalSymbol)
      return false;
    if (arc.src_state == state) {
      ++num_arcs;
    } else {
      // `arc_indexes` and `arcs` in this state are not consistent.
      if ((fsa.indexes[state + 1] - fsa.indexes[state]) != num_arcs)
        return false;
      state = arc.src_state;
      num_arcs = 1;
    }
  }
  // check the last state
  if ((fsa.indexes[final_state] - fsa.indexes[state]) != num_arcs) return false;
  return true;
}

bool IsTopSorted(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  for (const auto &arc : fsa) {
    if (arc.dest_state < arc.src_state) return false;
  }
  return true;
}

bool IsArcSorted(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_states = fsa.NumStates();
  const auto begin = fsa.data;
  const auto &arc_indexes = fsa.indexes;
  for (int32_t state = 0; state != num_states; ++state) {
    if (!std::is_sorted(begin + arc_indexes[state],
                        begin + arc_indexes[state + 1]))
      return false;
  }
  return true;
}

bool HasSelfLoops(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  for (const auto &arc : fsa) {
    if (arc.dest_state == arc.src_state) return true;
  }
  return false;
}

// Detect cycles using DFS traversal
bool IsAcyclic(const Fsa &fsa, std::vector<int32_t> *order /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  using dfs::DfsState;
  using dfs::kNotVisited;
  using dfs::kVisited;
  using dfs::kVisiting;
  if (IsEmpty(fsa)) return true;

  auto num_states = fsa.NumStates();
  std::vector<int8_t> state_status(num_states, kNotVisited);

  std::stack<DfsState> stack;
  stack.push({0, fsa.indexes[0], fsa.indexes[1]});
  state_status[0] = kVisiting;
  bool is_acyclic = true;
  while (is_acyclic && !stack.empty()) {
    auto &current_state = stack.top();
    if (current_state.arc_begin == current_state.arc_end) {
      // we have finished visiting this state
      state_status[current_state.state] = kVisited;
      if (order != nullptr) order->push_back(current_state.state);
      stack.pop();
      continue;
    }
    const auto &arc = fsa.data[current_state.arc_begin];
    auto next_state = arc.dest_state;
    auto status = state_status[next_state];
    switch (status) {
      case kNotVisited: {
        // a new discovered node
        state_status[next_state] = kVisiting;
        stack.push(
            {next_state, fsa.indexes[next_state], fsa.indexes[next_state + 1]});
        ++current_state.arc_begin;
        break;
      }
      case kVisiting:
        // this is a back arc indicating a loop in the graph
        is_acyclic = false;
        break;
      case kVisited:
        // this is a forward cross arc, do nothing.
        ++current_state.arc_begin;
        break;
      default:
        K2_LOG(FATAL) << "Unreachable code is executed!";
        break;
    }
  }

  return is_acyclic;
}

bool IsDeterministic(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  std::unordered_set<int32_t> labels;
  int32_t state = 0;
  for (const auto &arc : fsa) {
    if (arc.src_state == state) {
      if (labels.find(arc.label) != labels.end()) return false;
      labels.insert(arc.label);
    } else {
      state = arc.src_state;
      labels.clear();
      labels.insert(arc.label);
    }
  }
  return true;
}

bool IsEpsilonFree(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  for (const auto &arc : fsa) {
    if (arc.label == kEpsilon) return false;
  }
  return true;
}

bool IsUnweighted(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  for (const auto &arc : fsa) {
    if (arc.weight != 0.0) return false;
  }
  return true;
}

bool IsConnected(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  std::vector<int32_t> state_map;
  ConnectCore(fsa, &state_map);
  return static_cast<int32_t>(state_map.size()) == fsa.NumStates();
}
}  // namespace k2host
