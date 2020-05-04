// k2/csrc/properties.cc

// Copyright (c)  2020 Haowen Qiu
//                     Daniel Povey
//                     Mahsa Yarmohammadi

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/properties.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"

namespace k2 {

bool IsValid(const Fsa &fsa) {
  if (IsEmpty(fsa)) return true;
  StateId num_states = fsa.NumStates();
  // Nonempty fsa contains at least two states,
  // the start state and the final state.
  if (num_states < 2) return false;
  StateId state = 0;
  StateId final_state = num_states - 1;
  // the number of arcs in one state
  int32_t num_arcs = 0;
  for (const auto &arc : fsa.arcs) {
    // only kFinalSymbol arcs enter the final state
    if (arc.dest_state == final_state && arc.label != kFinalSymbol)
      return false;
    if (arc.src_state == state) {
      ++num_arcs;
    } else {
      // every state contains at least one arc.
      if (arc.src_state != state + 1) return false;
      // `arc_indexes` and `arcs` in this state are not consistent.
      if ((fsa.arc_indexes[state + 1] - fsa.arc_indexes[state]) != num_arcs)
        return false;
      state = arc.src_state;
      num_arcs = 1;
    }
  }
  // check the last state
  if (final_state != state + 1) return false;
  if ((fsa.arc_indexes[final_state] - fsa.arc_indexes[state]) != num_arcs)
    return false;
  return true;
}

bool IsTopSorted(const Fsa &fsa) {
  for (const auto &arc : fsa.arcs) {
    if (arc.dest_state < arc.src_state) return false;
  }
  return true;
}

bool IsArcSorted(const Fsa &fsa) {
  StateId final_state = fsa.NumStates() - 1;
  const auto begin = fsa.arcs.begin();
  const auto &arc_indexes = fsa.arc_indexes;
  // we will not check the final state as it has no arcs leaving it.
  for (StateId state = 0; state < final_state; ++state) {
    // as non-empty `fsa` contains at least two states,
    // we can always access `state + 1` validly.
    if (!std::is_sorted(begin + arc_indexes[state],
                        begin + arc_indexes[state + 1]))
      return false;
  }
  return true;
}

bool HasSelfLoops(const Fsa &fsa) {
  for (const auto &arc : fsa.arcs) {
    if (arc.dest_state == arc.src_state) return true;
  }
  return false;
}

bool CheckCycles(StateId s, std::vector<bool> visited, std::vector<bool> back_arc,
                 std::unordered_set<StateId> *adj) {
  if(visited[s] == false) {
    visited[s] = true;
    back_arc[s] = true;

    for (auto i = adj[s].begin(); i != adj[s].end(); ++i) {
      if (!visited[*i] && CheckCycles(*i, visited, back_arc, adj) )
        return true;
      else if (back_arc[*i])
        return true;
    }
  }
  back_arc[s] = false;
  return false;
}

// Detect cycles using DFS traversal
bool IsAcyclic(const Fsa &fsa) {
  StateId num_states = fsa.NumStates();
  std::vector<bool> visited(num_states, false);
  std::vector<bool> back_arc(num_states, false);

  std::unordered_set<StateId> *adj = new std::unordered_set<StateId>[num_states];
  for (const auto &arc : fsa.arcs)
     adj[arc.src_state].insert(arc.dest_state);

  for (StateId i = 0; i < num_states; i++)
    if (CheckCycles(i, visited, back_arc, adj))
      return true;

    return false;
  return true;
}

bool IsDeterministic(const Fsa &fsa) {
  std::unordered_set<Label> labels;
  StateId state = 0;
  for (const auto &arc : fsa.arcs) {
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
  for (const auto &arc : fsa.arcs) {
    if (arc.label == kEpsilon) return false;
  }
  return true;
}

bool IsConnected(const Fsa &fsa) {
  std::vector<int32_t> state_map;
  ConnectCore(fsa, &state_map);
  return static_cast<int32_t>(state_map.size()) == fsa.NumStates();
}
}  // namespace k2
