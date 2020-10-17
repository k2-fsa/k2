/**
 * @brief
 * compose
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_FSA_ALGO_H_
#define K2_CSRC_FSA_ALGO_H_

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"

namespace k2 {

/*
  This version of Connect() works for a single FSA.
    @param [in] src  Source FSA
    @param [out] dest   Destination; at exit will be equivalent to `src`
                     but will have no states that are unreachable or which
                     can't reach the final-state, i.e. its Properties() will
                     contain kFsaPropertiesMaybeCoaccessible and
                     kFsaPropertiesMaybeAccessible
    @param [out,optional] arc_map   For each arc in `dest`, gives the index of
                         the corresponding arc in `src` that it corresponds to.
    @return  Returns true on success (which basically means the input did not
            have cycles, so the algorithm could not succeed).  Success
            does not imply that `dest` is nonempty.

   CAUTION: for now this only works for CPU.

   This works for both Fsa and FsaVec!

 */
bool Connect(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map = nullptr);

/*
  Sort arcs of an Fsa or FsaVec in-place (this version of the function does not
  output derivatives).

          @param[in,out] fsa  FSA of which to sort the arcs.  Does not have
                         to be non-empty.
*/
void ArcSort(Fsa *fsa);

void ArcSort(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map = nullptr);

/*
  Topologically sort an Fsa or FsaVec.

      @param [in] src  Input Fsa or FsaVec
      @param [out] dest  Output Fsa or FsaVec.  At exit, its states will be
                      top-sorted.  (However, if `src` contained cycles other
                      than self-loops, it won't contain all of the states
                      in the input; this can be detected by the user directly
                      by looking at the number of states.
      @param [out] arc_map  If not nullptr, at exit a map from arc-indexes in
                      `dest` to their source arc-indexes in `src` will have
                       been assigned to this location.

  Implementation nots: from wikipedia
  https://en.wikipedia.org/wiki/Topological_sorting#Parallel_algorithms

 "An algorithm for parallel topological sorting on distributed memory machines
  parallelizes the algorithm of Kahn for a DAG {\displaystyle
  G=(V,E)}G=(V,E)[1]. On a high level, the algorithm of Kahn repeatedly removes
  the vertices of indegree 0 and adds them to the topological sorting in the
  order in which they were removed. Since the outgoing edges of the removed
  vertices are also removed, there will be a new set of vertices of indegree 0,
  where the procedure is repeated until no vertices are left."
*/
void TopSort(FsaVec &src, FsaVec *dest, Array1<int32_t> *arc_map = nullptr);

/*
  compose/intersect array of FSAs (multiple streams decoding or training in
  parallel, in a batch)... basically composition with frame-synchronous beam
  pruning, like in speech recognition.

  This code is intended to run on GPU (but should also work on CPU).

         @param[in] a_fsas   Input FSAs, `decoding graphs`.  There should
                         either be one FSA (a_fsas.Dim0() == 1) or a vector of
                         FSAs with the same size as b_fsas (a_fsas.Dim0() ==
                         b_fsas.Dim0()).
         @param[in] b_fsas   Input FSAs that correspond to neural network
                         outputs (see documentation in fsa.h).
         @param[in] beam   Decoding beam, e.g. 10.  Smaller is faster,
                         larger is more exact (less pruning).  This is the
                         default value; it may be modified by {min,max}_active.
         @param[in] max_active  Maximum active states allowed per frame.
                         (i.e. at each time-step in the sequences).  Sequence-
                         specific beam will be reduced if more than this number
                         of states are active.
         @param[in] min_active  Minimum active states allowed per frame; beam
                         will be decreased if the number of active states falls
                         below this
         @param[out] out Output vector of composed, pruned FSAs, with same
                         Dim0() as b_fsas.  Elements of it may be empty if the
                         composition was empty, either intrinsically or due to
                         failure of pruned search.
         @param[out] arc_map_a  Vector of

*/
void IntersectDensePruned(FsaVec &a_fsas, DenseFsaVec &b_fsas, float beam,
                          int32_t max_active_states, int32_t min_active_states,
                          FsaVec *out, Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b);

/*
  This is 'normal' intersection (we would call this Compose() for FSTs, but
  you can do that using Intersect(), by calling this and then Invert()
  (or just attaching the other aux_labels).  NOTE: epsilons are not treated
  specially, so this will only give the conventionally-correct answer
  if either a_fsas or b_fsas is epsilon free.

        @param [in] a_fsas  Fsa or FsaVec that is one of the arguments
                           for composition (i.e. 2 or 3 axes)
        @param [in] b_fsas  Fsa or FsaVec that is one of the arguments
                           for composition (i.e. 2 or 3 axes)
        @param [out] out    Output Fsa, will be an FsaVec (NumAxes() == 3)
                           regardless of the num_axes of the arguments.
        @param [out] arc_map_a  If not nullptr, this function will write to
                           here a map from (arc in `out`) to (arc in a_fsas).
        @param [out] arc_map_b  If not nullptr, this function will write to
                           here a map from (arc in `out`) to (arc in b_fsas).
 */
void Intersect(FsaOrVec &a_fsas, FsaOrVec &b_fsas, FsaVec *out,
               Array1<int32_t> *arc_map_a, Array1<int32_t> *arc_map_b);

/*
  Create a linear FSA from a sequence of symbols

    @param [in] symbols  Input symbol sequence (must not contain
                kFinalSymbol == -1).

    @return     Returns an FSA that accepts only this symbol
                sequence, with zero score.  Note: if
                `symbols.size() == n`, the returned FSA
                will have n+1 arcs (including the final-arc) and
                n+2 states.
*/
Fsa LinearFsa(const Array1<int32_t> &symbols);

/*
  Create an FsaVec containing linear FSAs, given a list of sequences of
  symbols

    @param [in] symbols  Input symbol sequences (must not contain
                kFinalSymbol == -1). Its num_axes is 2.

    @return     Returns an FsaVec with `ans.Dim0() == symbols.Dim0()`.  Note: if
                the i'th row of `symbols` has n elements, the i'th returned FSA
                will have n+1 arcs (including the final-arc) and n+2 states.
 */
FsaVec LinearFsas(Ragged<int32_t> &symbols);



}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
