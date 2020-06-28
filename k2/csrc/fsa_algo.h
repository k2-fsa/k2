// k2/csrc/fsa_algo.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_ALGO_H_
#define K2_CSRC_FSA_ALGO_H_

#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/weights.h"

namespace k2 {

/*
  The core part of Connect().  Connect() removes states that are not accessible
  or are not coaccessible, i.e. not reachable from start state or cannot reach
  the final state.

  If the resulting Fsa is empty, `state_map` will be empty at exit and
  it returns true.

     @param [in]  fsa         The FSA to be connected.  Requires
     @param [out] state_map   Maps from state indexes in the output fsa to
                              state indexes in `fsa`. If the input fsa is
                              acyclic, the output fsa is topologically sorted.

      Returns true on success (i.e. the output will be topsorted).
      The only failure condition is when the input had cycles that were not self
  loops.

      Caution: true return status does not imply that the returned FSA is
  nonempty.

 */
bool ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map);

/*
  Removes states that are not accessible (from the start state) or are not
  co-accessible (i.e. that cannot reach the final state), and ensures that
  if the FSA admits a topological sorting (i.e. it contains no cycles except
  self-loops), the version that is output is topologically sorted.  This
  is not a stable sort, i.e. states may be renumbered even for top-sorted
  input.

     @param [in] a    Input FSA
     @param [out] b   Output FSA, that will be trim / connected (there are
                      two terminologies).

     @param [out] arc_map   If non-NULL, this function will
                            output a map from the arc-index in `b` to
                            the corresponding arc-index in `a`.

     @return   The return status indicates whether topological sorting
        was successful; if true, the result is top-sorted.  The only situation
        it might return false is when the input had cycles that were not self
        loops; such FSAs do not admit a topological sorting.

        Caution: true return status does not imply that the returned FSA is
        nonempty.

  Notes:
    - If `a` admitted a topological sorting, b will be topologically
      sorted. If `a` is not topologically sorted but is acyclic, b will
      also be topologically sorted. TODO(Dan): maybe just leave in the
      same order as a?? (Current implementation may **renumber** the state)
    - If `a` was deterministic, `b` will be deterministic; same for
      epsilon free, obviously.
    - `b` will be arc-sorted (arcs sorted by label). TODO(fangjun): this
       has not be implemented.
    - `b` will (obviously) be connected
 */
bool Connect(const Fsa &a, Fsa *b, std::vector<int32_t> *arc_map = nullptr);

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
bool TopSort(const Fsa &a, Fsa *b, std::vector<int32_t> *state_map = nullptr);

/* Create an acyclic FSA from a list of arcs.

   Arcs do not need to be pre-sorted by src_state.
   If there is a cycle, it aborts.

   The start state MUST be 0. The final state will be automatically determined
   by topological sort.

   @param [in] arcs  A list of arcs.
   @param [out] fsa  Output fsa. Must be initialized; search for 'initialized
                     definition' in class Array2 in array.h for meaning.
   @param [out] arc_map   If non-NULL, this function will
                            output a map from the arc-index in `fsa` to
                            the corresponding arc-index in input `arcs`.
*/
void CreateFsa(const std::vector<Arc> &arcs, Fsa *fsa,
               std::vector<int32_t> *arc_map = nullptr);

}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
