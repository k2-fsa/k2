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
  contain kFsaPropertiesMaybeCoaccessible and kFsaPropertiesMaybeAccessible
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
  Topologically sort an Fsa or FsaVec (where possible).  Note: if the FSA had
  cycles the result will not be topologically sorted (it will have cycles) but
  the states will be ordered by the number of arcs it takes to reach them from
  the start state.

      @param [in] src  Input Fsa or FsaVec
      @param [out] dest  Output Fsa or FsaVec.  At exit, its states will be
                        top-sorted if the FSA was acyclic.  If it had cycles,
                        they will be sorted by distance from the start-state
                        (distance in terms of how many arcs must be traversed,
                        not the weights).
      @param [out] arc_map  If not nullptr, a map from arc-indexes in `dest` to
                        arc-indexes in `src` will be output to here
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
  Dim0() as b_fsas.  Elements of it may be empty if the composition was empty,
  either intrinsically or due to failure of pruned search.
         @param[out] arc_map_a  Vector of

*/
void IntersectDensePruned(FsaVec &a_fsas, DenseFsaVec &b_fsas, float beam,
                          int32_t max_active_states, int32_t min_active_states,
                          FsaVec *out, Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b);

}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
