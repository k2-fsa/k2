/**
 * @brief
 * remove epsilon
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RM_EPSILON_H_
#define K2_CSRC_RM_EPSILON_H_

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

/*
  Epsilon removal algorithm (for tropical semiring):

  Involves first separating epsilon and non-epsilon arcs, doing a closure on
  the epsilon part so we have direct epsilon arcs between pairs of states that
  can reach each other by epsilons; then re-combining the epsilon and
  non-epsilon parts of the FSA.  We decide, for each epsilon in the closure of
  the epsilon part, whether to combine it with following or preceding
  non-epsilon arcs based on which would produce fewer additional arcs.
*/

/*
  Extract just the epsilon arcs and the states with epsilons entering and
  leaving them (plus the start and final states).  For use inside
  epsilon-removal algorithm.
     @param [in] src   Source FsaVec; must have 3 axes.
     @param [in] dest  Output FsaVec; will contain all the states that had
                       epsilon arcs leaving them or entering them, plus any
                       initial and final final states in `src`.
                       Noted if src[i] has no arc (but has some states),
                       we would not keep the start state and final state of it
                       in the corresponding output Fsa dest[i], i.e. dest[i]
                       will be an empty Fsa.
    @param [out]       Will be set to a new Array1 mapping from the
                       state_idx01's in `dest` to the corresponding
                       state_idx01's in `src`.
    @param [out] arc_map  Will be set to a new Array1, mapping from the
                       arc_idx012's in `dest` to the corresponding arc_idx012's
                       in `src`.
*/
void ComputeEpsilonSubset(FsaVec &src, FsaVec *dest, Array1<int32_t> *state_map,
                          Array1<int32_t> *arc_map);

/*
  Extract just the non-epsilon arcs and the states that have non-epsilons
  leaving or entering them (plus the start and final states).  For use inside
  epsilon-removal algorithm.

     @param [in] src   Source FsaVec; must have 3 axes.
     @param [in] dest  Output FsaVec; will contain all the states that had
                       non-epsilon arcs leaving them or entering them, plus any
                       initial and final final states in `src`.
                       Noted if src[i] has no arc (but has some states),
                       we would not keep the start state and final state of it
                       in the corresponding output Fsa dest[i], i.e. dest[i]
                       will be an empty Fsa.
     @param [out]      Will be set to the renumbering object from the old to new
                       state indexes.
    @param [out] arc_map  Will be set to a new Array1, mapping from the
                       arc_idx012's in `dest` to the corresponding arc_idx012's
                       in `src`.
*/
void ComputeNonEpsilonSubset(FsaVec &src, FsaVec *dest, Renumbering *state_map,
                             Array1<int32_t> *arc_map);

/*
   Map some states of an FsaVec, with the arcs entering and leaving those states
      @param [in] src   Source FsaVec, to be mapped
      @param [in] state_row_splits   The row_splits vector for `dest`, which
                         determines the number of states for each output FSA
      @param [out] state_row_ids  The row_ids vector corresponding to
                        `state_row_splits`
      @param [in] map   Map from state_idx01's in `src` to state_idx01's in
                       `dest`, with -1 for states that are to be removed.  Note:
                        the number of states in `src` may be smaller or larger
                        than state_row_ids.Dim().
      @param [out] dest  Destination FsaVec; at exit, will contain all arcs in
                        `src` whose src_state and dest_state are both kept
                        (i.e. not mapped to -1).
      @param [out] arc_map Will be set to a new Array1 that maps from arc-index
                        in `dest` to original arc-index in `src`.

*/
void MapFsaVecStates(FsaVec &src, const Array1<int32_t> &state_row_splits,
                     const Array1<int32_t> &state_row_ids,
                     const Array1<int32_t> &state_map, FsaVec *dest,
                     Array1<int32_t> *arc_map);

/*
  Compute the closure of an FSA containing just epsilon arcs (as output
  by ComputeEpsilonSubset()).  This means adding epsilon arcs from
  each state s1 to each state s2 which is reachable indirectly by epsilons
  from s1 to s2.  Note: this implicitly assumes the tropical semiring, because
  we are taking only the best epsilon path from any state to any other
  state.

     @param [in] epsilon_fsa   FSA containing only epsilon arcs, as output
                           by ComputeEpsilonSubset()
     @param [out] closure_fsa  FSA containing the closure of the epsilon arcs.
                           Will be arc-sorted, and no state will have more than
                           one arc to any other state.

  Implementation notes from Dan: I suggest to repeatedly call
  ComputeEpsilonClosureOneIter() until there is no further change in the
  FsaVec (this can be by simple comparison on arcs vector, since thanks to
  sorting the order is deterministic).  Obviously the arc_maps from the
  individual iterations must be composed.
*/
void ComputeEpsilonClosure(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                           Ragged<int32_t> *arc_map);

/*
 One iteration of the algorithm in ComputeEpsilonClosure().
   @param [in] FSA containing only epsilon arcs (possibly already passed through
                   one or more iterations of closure).  Must have 3 axes.
   @param [out] closure_fsa   FSA that is the result of one iteration of
                    closure/ Will contain an arc from state s1 to s2 if there
                    was already such an arc in `epsilon_fsa` or if there was
                    a state s3 such that there was an arc from s1 to s2 and
                    one from s2 to s3. Will contain at most one arc from one
                    state to any other state.
    @param [out] arc_map   For each arc in closure_fsa, contains the sequence of
                  arc_idx012's in epsilon_fsa that was the source.

  Implementation notes from Dan: I suggest to over-generate arcs,
  (i.e. for each arc, generate n extra arcs if its dest-state had n arcs leaving
  it), then arc-sort with an operator that sorts on (dest-state then weight),
  then mark arcs to be (kept or not) according to whether the previous arc was
  to the same dest state, then renumber with a Renumbering class.
*/
void ComputeEpsilonClosureOneIter(FsaVec &epsilon_fsa, FsaVec *closure_fsa,
                                  Ragged<int32_t> *arc_map);

/*
  Remove epsilons from FsaVec in `src_fsa`, producing an FsaVec `dest_fsa` which
  is equivalent (in tropical semiring).  Uses an iterative algorithm which tries
  to minimize the number of arcs in the resulting FSA (epsilons are combined
  with either preceding or following arcs).
*/
void RemoveEpsilonsIterativeTropical(FsaVec &src_fsa, FsaVec *dest_fsa,
                                     Ragged<int32_t> *arc_map);
}  // namespace k2

#endif  // K2_CSRC_RM_EPSILON_H_
