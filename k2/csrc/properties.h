// k2/csrc/properties.h

// Copyright (c)  2020  Daniel Povey
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_PROPERTIES_H_
#define K2_CSRC_PROPERTIES_H_

#include "k2/csrc/fsa.h"

namespace k2 {

// TODO(Dan): we might not need this.
enum Properties {
  kTopSorted,            // topologically sorted
  kTopSortedAndAcyclic,  // topologically sorted and no self-loops (which
                         // implies acyclic)
  kArcSorted,            // arcs leaving each state are sorted on label
  kDeterministic,        // no state has two arcs leaving it with the same label
  kConnected,    // all states are both accessible (i.e. from start state) and
                 // coaccessible (i.e. can reach final-state)
  kEpsilonFree,  // there are no arcs with epsilon (kEpsilon == 0) as the label
  kNonempty      // the FST does not have zero states
};

/*
  `fsa` is valid if:
  1. it is empty, if not, it contains at least two states.
  2. only epsilon arcs enter the final state.
  3. every state contains at least one arc except the final state.
  4. `leaving_arcs` and `arcs` in this state are not consistent.
  TODO(haowen): add more rules?
 */
bool IsValid(const Fsa &fsa);

/*
  Returns true if the states in `fsa` are topologically sorted.
*/
bool IsTopSorted(const Fsa &fsa);

/*
  Returns true if `fsa` has any self-loops
 */
bool HasSelfLoops(const Fsa &fsa);

/*
  Returns true if `fsa` is both topologically sorted and free
  of self-loops (together, these imply that it is acyclic, i.e.
  free of all cycles).
*/
inline bool IsTopSortedAndAcyclic(const Fsa &fsa) {
  return IsTopSorted(fsa) && !HasSelfLoops(fsa);
}

/*
  Returns true if `fsa` is deterministic; an Fsa is deterministic if it has is
  no state that has multiple arcs leaving it with the same label on them.
*/
bool IsDeterministic(const Fsa &fsa);

/*
  Returns true if `fsa` is free of epsilons, i.e. if there are no arcs
  for which `label` is kEpsilon == 0.
*/
bool IsEpsilonFree(const Fsa &fsa);

/*
  Returns true if all states in `fsa` are both reachable from the start state
  (accessible) and can reach the final state (coaccessible).  Note: an FSA can
  be both connected and empty, because the empty FSA has no states (neither
  start state nor final state exist).  So you may sometimes want to check
  IsConnected() && IsNonempty().

  Requires that `fsa` be valid and top-sorted.
  TODO(haowen): implement another version for non-top-sorted `fsa`.
 */
bool IsConnected(const Fsa &fsa);

/*
  Returns true if `fsa` is empty. (Note: if `fsa` is not empty,
  it would contain at least two states, the start state and the final state).
 */
inline bool IsEmpty(const Fsa &fsa) {
  return fsa.leaving_arcs.empty() && fsa.arcs.empty();
}

/*
  Returns true if `fsa` is valid AND satisfies the list of properties
  provided in `properties`.

  TODO(Dan): create exhaustive list of what `valid` means, but it basically
  means that the data structures make sense, e.g. `leaving_arcs` and
  `arcs` are consistent, and only epsilon arcs enter the final state
  (note: the final state is special, and such arcs represent

 */
bool CheckProperties(const Fsa &fsa, const Properties &properties,
                     bool die_on_error = false);

}  // namespace k2

#endif  // K2_CSRC_PROPERTIES_H_
