/**
 * @brief
 * properties
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_PROPERTIES_H_
#define K2_CSRC_HOST_PROPERTIES_H_

#include <vector>

#include "k2/csrc/host/fsa.h"

namespace k2host {

// TODO(Dan): we might not need this.
enum Properties {
  kTopSorted,            // topologically sorted
  kTopSortedAndAcyclic,  // topologically sorted and no self-loops (which
                         // implies acyclic)
  kAcyclic,              // acyclic
  kArcSorted,            // arcs leaving each state are sorted on label and then
                         // destination state
  kDeterministic,        // no state has two arcs leaving it with the same label
  kConnected,    // all states are both accessible (i.e. from start state) and
                 // coaccessible (i.e. can reach final-state)
  kEpsilonFree,  // there are no arcs with epsilon (kEpsilon == 0) as the label
  kUnweighted,   // the scores are always zero.
  kNonempty      // the FST does not have zero states
};

/*
  `fsa` is valid if:
  1. it is empty, if not, it contains at least two states.
  2. only kFinalSymbol arcs enter the final state.
  3. `arc_indexes` and `arcs` in this state are consistent
  TODO(haowen): add more rules?
 */
bool IsValid(const Fsa &fsa);

/*
  Returns true if the states in `fsa` are topologically sorted.
*/
bool IsTopSorted(const Fsa &fsa);

/*
  Returns true if arcs leaving each state in `fsa` are sorted on label first and
  then on dest_state.
*/
bool IsArcSorted(const Fsa &fsa);

/*
  Returns true if `fsa` has any self-loops
 */
bool HasSelfLoops(const Fsa &fsa);

/*
  Returns true if `fsa` is acyclic. Cycles in parts of the FSA that are not
  accessible (i.e. from the start state) are not considered.
  The optional argument order, assigns the order in which visiting states is
  finished in DFS traversal. State 0 has the largest order (num_states - 1) and
  the final state has the smallest order (0).
 */
bool IsAcyclic(const Fsa &fsa, std::vector<int32_t> *order = nullptr);

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

  Requires that `fsa` be valid.
 */
bool IsConnected(const Fsa &fsa);

/*
  Returns true if all states in `fsa` have zero scores
 */
bool IsUnweighted(const Fsa &fsa);

/*
  Returns true if `fsa` is both acyclic and connected.
*/
inline bool IsAcyclicAndConnected(const Fsa &fsa) {
  return IsAcyclic(fsa) && IsConnected(fsa);
}

/*
  Returns true if `fsa` is both topologically sorted and connected.
*/
inline bool IsTopSortedAndConnected(const Fsa &fsa) {
  return IsTopSorted(fsa) && IsConnected(fsa);
}

/*
  Returns true if `fsa` is empty. (Note: if `fsa` is not empty,
  it would contain at least two states, the start state and the final state).

  Caution: this is not always very meaningful, as an FSA with no states is
  conceptually equivalent to an FSA with two states but no arcs.
 */
inline bool IsEmpty(const Fsa &fsa) { return fsa.size1 == 0; }

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

}  // namespace k2host

#endif  // K2_CSRC_HOST_PROPERTIES_H_
