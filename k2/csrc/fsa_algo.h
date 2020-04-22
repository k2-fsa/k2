// k2/csrc/fsa_algo.h

// Copyright     2020  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "k2/csrc/fsa.h"

#ifndef K2_CSRC_FSA_ALGO_H_
#define K2_CSRC_FSA_ALGO_H_


namespace k2 {

/*
  The core part of Connect().  Connect() removes states that are not accessible
  or are not coaccessible, i.e. not reachable from start state or cannot reach the
  final state.

  If the resulting Fsa is empty, `state_map` will be empty at exit.

     @param [in] fsa  The FSA to be connected.  Requires
     @param [out] state_map   Maps from state indexes in the output fsa to
                      state indexes in `fsa`.  Will retain the original order,
                      so the output will be topologically sorted if the input
                      was.
 */
void ConnectCore(const Fsa &fsa,
                 std::vector<int32> *state_map);


/*
  Removes states that are not accessible (from the start state) or are not
  coaccessible (i.e. that cannot reach the final state), and ensures that if the
  FSA admits a topological sorting (i.e. if contains no cycles except
  self-loops), the version that is output is topologically sorted.

     @param [in] a    Input FSA
     @param [out] b   Output FSA, that will be trim / connected (there are
                     two terminologies).

     @param [out] arc_map   If non-NULL, this function will
                     output a map from the arc-index in `b` to
                     the corresponding arc-index in `a`.

  Notes:
    - If `a` admitted a topological sorting, b will be topologically
      sorted.  TODO: maybe just leave in the same order as a??
    - If `a` was deterministic, `b` will be deterministic; same for
      epsilon free, obviously.
    - `b` will be arc-sorted (arcs sorted by label)
    - `b` will (obviously) be connected
 */
void Connect(const Fsa &a,
             Fsa *b,
             std::vector<int32> *arc_map = NULL);


/**
   Output an Fsa that is equivalent to the input but which has no epsilons.

    @param [in] a  The input FSA
    @param [out] b  The output FSA; will be epsilon-free, and the states
                    will be in the same order that they were in in `a`.
    @param [out] arc_map  If non-NULL: for each arc in `b`, a list of
                    the arc-indexes in `a` that contributed to that arc
                    (e.g. its cost would be a sum of their costs).
                    TODO: make it a VecOfVec, maybe?
 */
void RmEpsilons(const Fsa &a,
                Fsa *b,
                std::vector<std::vector> *arc_map = NULL);


/**
   Pruned version of RmEpsilons, which also uses a pruning beam.

   Output an Fsa that is equivalent to the input but which has no epsilons.

    @param [in] a  The input FSA
    @param [out] b  The output FSA; will be epsilon-free, and the states
                    will be in the same order that they were in in `a`.
    @param [out] arc_map  If non-NULL: for each arc in `b`, a list of
                    the arc-indexes in `a` that contributed to that arc
                    (e.g. its cost would be a sum of their costs).
                    TODO: make it a VecOfVec, maybe?
 */
void RmEpsilonsPruned(const Fsa &a,
                      const float *a_state_forward_costs,
                      const float *a_state_backward_costs,
                      const float *a_arc_costs,
                      float cutoff,
                      Fsa *b,
                      std::vector<std::vector> *arc_map = NULL);




/*
  Compute the intersection of two FSAs; this is the equivalent of composition
  for automata rather than transducers, and can be used as the core of
  composition.

  @param [in] a    One of the FSAs to be intersected.  Must satisfy
                   CheckProperties(a, kArcSorted)
  @param [in] b    The other FSA to be intersected  Must satisfy
                   CheckProperties(b, kArcSorted), and either a or b
                   must be epsilon-free (c.f. IsEpsilonFree); this
                   ensures that epsilons do not have to be treated
                   differently from any other symbol.
  @param [out] c   The composed FSA will be output to here.
  @param [out] arc_map_a   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `a` was.
  @param [out] arc_map_b   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `b` was.
 */
void Intersect(const Fsa &a,
               const Fsa &b,
               Fsa *c,
               std::vector<int32> *arc_map_a = NULL,
               std::vector<int32> *arc_map_b = NULL);


/*
  Version of Intersect where `a` is dense?
 */
void Intersect(const DenseFsa &a,
               const Fsa &b,
               Fsa *c,
               std::vector<int32> *arc_map_a = NULL,
               std::vector<int32> *arc_map_b = NULL);


/*
  Version of Intersect where `a` is dense, pruned with pruning beam `beam`.
  Suppose states in the output correspond to pairs (s_a, s_b), and have
  forward-weights w(s_a, s_b), i.e. best-path from the start state...
  then if a state has a forward-weight w(s_a, s_b) that is less than
  (the largest w(s_a, x) for any x) minus the beam, we don't expand it.

  This is the same as time-synchronous Viterbi beam pruning.
*/
void IntersectPruned(const DenseFsa &a,
                     const Fsa &b,
                     float beam,
                     Fsa *c,
                     std::vector<int32> *arc_map_a = NULL,
                     std::vector<int32> *arc_map_b = NULL);




/**
   Intersection of two weighted FSA's: the same as Intersect(), but it prunes
   based on the sum of two costs.  Note: although these costs are provided per
   arc, they would usually be a sum of forward and backward costs, that is
   0 if this arc is on a best path and otherwise is the distance between
   the cost of this arc and the best-path cost.

  @param [in] a    One of the FSAs to be intersected.  Must satisfy
                   CheckProperties(a, kArcSorted)
  @param [in] a_cost  Pointer to array containing a cost per arc of a
  @param [in] b    The other FSA to be intersected  Must satisfy
                   CheckProperties(b, kArcSorted), and either a or b
                   must be epsilon-free (c.f. IsEpsilonFree).
  @param [in] b_cost  Pointer to array containing a cost per arc of b
  @param [in] cutoff  Cutoff, such that we keep an arc in the output
                   if its cost_a + cost_b is less than this cutoff,
                   where cost_a and cost_b are elements of
                   `a_cost` and `b_cost`.
  @param [out] c   The output FSA will be written to here.
  @param [out] state_map_a  Maps from arc-index in c to the corresponding
                   arc-index in a
  @param [out] state_map_b  Maps from arc-index in c to the corresponding
                   arc-index in b
 */
void IntersectPruned2(const Fsa &a, const float *a_cost,
                      const Fsa &b, const float *b_cost,
                      float cutoff,
                      Fsa *c,
                      std::vector<int32> *state_map_a,
                      std::vector<int32> *state_map_b);




void RandomPath(const Fsa &a,
                const float *a_cost,
                Fsa *b,
                std::vector<int32> *state_map = NULL);


}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
