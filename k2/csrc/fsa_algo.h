// k2/csrc/fsa_algo.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_ALGO_H_
#define K2_CSRC_FSA_ALGO_H_

#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/weights.h"

namespace k2 {

/*
  The core part of Connect().  Connect() removes states that are not accessible
  or are not coaccessible, i.e. not reachable from start state or cannot reach
  the final state.

  If the resulting Fsa is empty, `state_map` will be empty at exit.

     @param [in]  fsa         The FSA to be connected.  Requires
     @param [out] state_map   Maps from state indexes in the output fsa to
                              state indexes in `fsa`. If the input fsa is
                              acyclic, the output fsa is topologically sorted.
 */
void ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map);

/*
  Removes states that are not accessible (from the start state) or are not
  co-accessible (i.e. that cannot reach the final state), and ensures that if the
  FSA admits a topological sorting (i.e. if contains no cycles except
  self-loops), the version that is output is topologically sorted.

  If the input fsa is not topologically sorted but is acyclic, then the output
  fsa is topologically sorted.

     @param [in] a    Input FSA
     @param [out] b   Output FSA, that will be trim / connected (there are
                      two terminologies).

     @param [out] arc_map   If non-NULL, this function will
                            output a map from the arc-index in `b` to
                            the corresponding arc-index in `a`.

  Notes:
    - If `a` admitted a topological sorting, b will be topologically
      sorted. If `a` is not topologically sorted but is acyclic, b will
      also be topologically sorted. TODO(Dan): maybe just leave in the
      same order as a??
    - If `a` was deterministic, `b` will be deterministic; same for
      epsilon free, obviously.
    - `b` will be arc-sorted (arcs sorted by label). TODO(fangjun): this
       has not be implemented.
    - `b` will (obviously) be connected
 */
void Connect(const Fsa &a, Fsa *b, std::vector<int32_t> *arc_map = nullptr);

/**
   Output an Fsa that is equivalent to the input (in the tropical semiring,
   which here means taking the max of the weights along paths) but which has no
   epsilons.  The input needs to have associated weights, because they will be
   used to choose the best among alternative epsilon paths between states.

    @param [in]  a  The input, with weights and forward-backward weights
                    as required by this computation.  For now we assume
                    that `a` is topologically sorted, as required by
                    the current constructor of WfsaWithFbWeights.
    @param [in] beam  beam > 0 that affects pruning; this algorithm will
                    keep paths that are within `beam` of the best path.
                    Just make this very large if you don't want pruning.
    @param [out] b  The output FSA; will be epsilon-free, and the states
                    will be in the same order that they were in in `a`.
    @param [out] arc_map  If non-NULL: for each arc in `b`, a list of
                    the arc-indexes in `a`, in order, that contributed
                    to that arc (e.g. its cost would be a sum of their costs).

   Notes on algorithm (please rework all this when it's complete, i.e. just
   make sure the code is clear and remove this).

     The states in the output FSA will correspond to the subset of states in the
     input FSA which are within `beam` of the best path and which have at least
     one non-epsilon arc entering them, plus the start state.  (Note: this
     automatically includes the final state, assuming `a` has at least one
     successful path; if it does not, the output will be empty).

     If we ever need the associated state map from calling code, we'll add an
     extra output argument to this function.

     The basic algorithm is to (1) identify the kept states, (2) from each kept
     input-state ki, we'll iterate over all states that are reachable via zero
     or more epsilons from this state and process the non-epsilon outgoing arcs
     from those states, which will become the arcs in the output.  We'll also
     store a back-pointer array that will allow us to figure out the best path
     back to ki, in order to produce the output `arc_map`.    Assume we have
     arrays

     local_forward_weights (float) and local_backpointers (int) indexed by
     state-id, and that the local_forward_weights are initialized with
     -infinity's each time we process a new ki. (we have to figure out how to do
     this efficiently).


      Processing input-state ki:
         local_forward_state_weights[ki] = forward_state_weights[ki] // from WfsaWithFbWeights.
                                                                     // Caution: we should probably use
                                                                     // double here; these kinds of algorithms
                                                                     // are extremely sensitive to roundoff for
                                                                     // very long FSAs.
         local_backpointers[ki] = -1  // will terminate a sequence..
         queue.push_back(ki)
         while (!queue.empty()) {
            ji = queue.front()  // we have to be a bit careful about order here, to make sure
                                // we always process states when they already have the
                                // best cost they are going to get.  If
                                // FSA was top-sorted at the start, which we assume, we could perhaps
                                // process them in numerical order, e.g. using a heap.
            queue.pop_front() 
            for each arc leaving state ji:
                next_weight = local_forward_state_weights[ji] + arc_weights[this_arc_index]
                if next_weight + backward_state_weights[arc_dest_state] < best_path_weight - beam:
                    if arc label is epsilon: 
                        if next_weight < local_forward_state_weight[next_state]:
                           local_forward_state_weight[next_state] = next_weight
                           local_backpointers[next_state] = ji
                else:
                    add an arc to the output FSA, and create the appropriate
                    arc_map entry by following backpointers (hopefully you can
                    figure out the details).  Note: the output FSA's weights can be
                    computed later on, by calling code, using the info in arc_map.
 */
void RmEpsilonsPruned(const WfsaWithFbWeights &a, float beam, Fsa *b,
                      std::vector<std::vector<int32_t>> *arc_map);

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
                   `c` what the source arc in `a` was, `-1` represents
                   there is no corresponding source arc in `a`.
  @param [out] arc_map_b   If non-NULL, at exit will be a vector of
                   size c->arcs.size(), saying for each arc in
                   `c` what the source arc in `b` was, `-1` represents
                   there is no corresponding source arc in `b`.
 */
bool Intersect(const Fsa &a, const Fsa &b, Fsa *c,
               std::vector<int32_t> *arc_map_a = nullptr,
               std::vector<int32_t> *arc_map_b = nullptr);

/*
  Version of Intersect where `a` is dense?
 */
void Intersect(const DenseFsa &a, const Fsa &b, Fsa *c,
               std::vector<int32_t> *arc_map_a = nullptr,
               std::vector<int32_t> *arc_map_b = nullptr);

/*
  Version of Intersect where `a` is dense, pruned with pruning beam `beam`.
  Suppose states in the output correspond to pairs (s_a, s_b), and have
  forward-weights w(s_a, s_b), i.e. best-path from the start state...
  then if a state has a forward-weight w(s_a, s_b) that is less than
  (the largest w(s_a, x) for any x) minus the beam, we don't expand it.

  This is the same as time-synchronous Viterbi beam pruning.
*/
void IntersectPruned(const DenseFsa &a, const Fsa &b, float beam, Fsa *c,
                     std::vector<int32_t> *arc_map_a = nullptr,
                     std::vector<int32_t> *arc_map_b = nullptr);

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
void IntersectPruned2(const Fsa &a, const float *a_cost, const Fsa &b,
                      const float *b_cost, float cutoff, Fsa *c,
                      std::vector<int32_t> *state_map_a,
                      std::vector<int32_t> *state_map_b);

void RandomPath(const Fsa &a, const float *a_cost, Fsa *b,
                std::vector<int32_t> *state_map = nullptr);

/**
    Sort arcs leaving each state in `fsa` on label first and then on dest_state
    @param [in]   a   Input fsa to be arc sorted.
    @param [out]  b   Output fsa which is an arc sorted fsa.
    @param [out]  arc_map   Maps from arc indexes in the output fsa to
                              arc indexes in input fsa.
 */
void ArcSort(const Fsa &a, Fsa *b, std::vector<int32_t> *arc_map = nullptr);
/**
    Sort the input fsa topologically.

    It returns an empty fsa when the input fsa is not acyclic,
    is not connected, or is empty; otherwise it returns the topologically
    sorted fsa in `b`.

    @param [in]   a   Input fsa to be topo sorted.
    @param [out]  b   Output fsa. It is set to empty if the input fsa is not
                      acyclic or is not connected; otherwise it contains the
                      topo sorted fsa.
    @param [out]  state_map   Maps from state indexes in the output fsa to
                              state indexes in input fsa. It is empty if
                              the output fsa is empty.
    @return true if the input fsa is acyclic and connected,
            or if the input is empty; return false otherwise.
 */
bool TopSort(const Fsa& a, Fsa* b, std::vector<int32_t>* state_map = nullptr);

/**

 */
void Determinize(const Fsa &a, Fsa *b,
                 std::vector<std::vector<StateId>> *state_map);

}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
