// k2/csrc/properties.cc

// Copyright (c)  2020 Haowen Qiu
//                     Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/properties.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "k2/csrc/fsa.h"

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
  ArcLabelCompare comp;
  StateId final_state = fsa.NumStates() - 1;
  const auto begin = fsa.arcs.begin();
  const auto &arc_indexes = fsa.arc_indexes;
  // we will not check the final state as it has no arcs leaving it.
  for (StateId state = 0; state < final_state; ++state) {
    // as non-empty `fsa` contains at least two states,
    // we can always access `state + 1` validly.
    if (!std::is_sorted(begin + arc_indexes[state],
                        begin + arc_indexes[state + 1], comp))
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
  StateId num_states = fsa.NumStates();
  std::vector<bool> accessible(num_states, false);
  accessible[0] = true;  // the start state
  for (const auto &arc : fsa.arcs) {
    if (!accessible[arc.src_state]) return false;
    accessible[arc.dest_state] = true;
  }
  if (std::find(accessible.begin(), accessible.end(), false) !=
      accessible.end())
    return false;

  // reuse `accessible` to process co-accessible situation.
  std::fill(accessible.begin(), accessible.end(), false);
  accessible[num_states - 1] = true;  // the final state
  int32_t num_arcs = static_cast<int32_t>(fsa.arcs.size());
  for (int32_t i = num_arcs - 1; i >= 0; --i) {
    const auto &arc = fsa.arcs[i];
    if (!accessible[arc.dest_state]) return false;
    accessible[arc.src_state] = true;
  }
  return std::find(accessible.begin(), accessible.end(), false) ==
         accessible.end();
}
}  // namespace k2
