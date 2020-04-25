// k2/csrc/properties.cc

// Copyright (c)  2020 Haowen Qiu
//                     Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/properties.h"

#include <algorithm>
#include <iostream>
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
    // only epsilon arcs enter the final state
    if (arc.dest_state == final_state && arc.label != kEpsilon) return false;
    if (arc.src_state == state) {
      ++num_arcs;
    } else {
      // every state contains at least one arc.
      if (arc.src_state != state + 1) return false;
      // `leaving_arcs` and `arcs` in this state are not consistent.
      if ((fsa.leaving_arcs[state + 1].begin - fsa.leaving_arcs[state].begin) !=
          num_arcs)
        return false;
      state = arc.src_state;
      num_arcs = 1;
    }
  }
  // check the last state
  if (final_state != state + 1) return false;
  if ((fsa.leaving_arcs[final_state].begin - fsa.leaving_arcs[state].begin) !=
      num_arcs)
    return false;
  return true;
}

bool IsTopSorted(const Fsa &fsa) {
  for (const auto &arc : fsa.arcs) {
    if (arc.dest_state < arc.src_state) return false;
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

  std::cout << "accessible done........." << std::endl;
  // reuse `accessible` to process co-accessible situation.
  std::vector<bool> co_accessible(num_states, false);
  co_accessible[num_states - 1] = true;  // the final state
  int32_t num_arcs = fsa.arcs.size();
  for (int32_t i = num_arcs - 1; i >= 0; --i) {
    const auto &arc = fsa.arcs[i];
    if (!co_accessible[arc.dest_state]) return false;
    co_accessible[arc.src_state] = true;
  }
  std::cout << "co-accessible done........." << std::endl;
  // reuse `accessible` to process co-accessible situation.
  return std::find(co_accessible.begin(), co_accessible.end(), false) ==
         co_accessible.end();
}
}  // namespace k2
