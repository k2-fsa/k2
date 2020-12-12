/**
 * @brief
 * compose
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
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
  This version of Connect() works for both Fsa and FsaVec.
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
 */
bool Connect(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map = nullptr);

/*
  Sort arcs of an Fsa or FsaVec in-place (this version of the function does not
  output derivatives).

          @param[in,out] fsa  FSA of which to sort the arcs.  Does not have
                         to be non-empty.
*/
void ArcSort(FsaOrVec *fsa);

void ArcSort(FsaOrVec &src, FsaOrVec *dest, Array1<int32_t> *arc_map = nullptr);

/*
  Topologically sort an Fsa or FsaVec.

      @param [in] src  Input Fsa or FsaVec
      @param [out] dest  Output Fsa or FsaVec.  At exit, its states will be
                      top-sorted.
                      Caution: our current implementation requires that the
                      input Fsa is acyclic (self-loop is OK) and there's no
                      arc entering the start state. So if `src` contained
                      cycles other than self-loops or there are arcs (except
                      self-loops) entering start state, the program will
                      abort with an error.
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
  Same with `TopSort` above, but only works for CPU. It's just a wrapper of
  `TopSorter` in host/topsort.h. We use it for test purpose, users should never
  call this function in production code. Instead, you should call the version
  above.
      @param [in] src  Input Fsa or FsaVec
      @param [out] dest  Output Fsa or FsaVec.  At exit, its states will be
                      top-sorted.  (However, if `src` contained cycles other
                      than self-loops, the function will return false and the
                      output Fsa will be empty).
      @param [out] arc_map  If not nullptr, at exit a map from arc-indexes in
                      `dest` to their source arc-indexes in `src` will have
                       been assigned to this location.
      @return Returns true on success (i.e. the output will be topsorted).
            The only failure condition is when the input had cycles that were
            not self loops. Noted we may remove those states in the
            input Fsa which are not accessible or co-accessible.
            Caution: true return status does not imply that the returned FSA
            is nonempty.
 */
bool HostTopSort(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map = nullptr);

/*
  Add epsilon self-loops to an Fsa or FsaVec (this is required when composing
  using a composition method that does not treat epsilons specially, if the
  other FSA has epsilons in it).

     @param [in] src  Input Fsa or FsaVec.  Does not have to have any
                    special properties (although if you want correct
                    composition in the log semiring it should probably
                    be epsilon free).
     @param [out] dest  Output Fsa or FsaVec.  At exit it will be like 'src'
                    but with epsilon self-loops with zero scores on all
                    non-final states.  These self-loops will be the first arcs.
                    This will ensure that the output satisfies the property
                    kFsaPropertiesArcSorted, although for non-top-sorted inputs
                    with epsilon arcs it may not be considered as arc-sorted by
                    the `host` code because that takes into account the
                    destination-states.
     @param [out] arc_map  If not nullptr, will be set to a new Array1<int32_t>
                    containing the input arc-index corresponding
                    to each output arc (or -1 for newly added self-loops).
 */
void AddEpsilonSelfLoops(FsaOrVec &src, FsaOrVec *dest,
                         Array1<int32_t> *arc_map = nullptr);

/*
  compose/intersect array of FSAs (multiple streams decoding or training in
  parallel, in a batch)... basically composition with frame-synchronous beam
  pruning, like in speech recognition.

  This code is intended to run on GPU (but should also work on CPU).

         @param[in] a_fsas   Input FSAs, `decoding graphs`.   There should
                         either be one FSA (3 axes and a_fsas.Dim0() == 1; or
                         2 axes) or a vector of FSAs with the same size as
                         b_fsas (a_fsas.Dim0() == b_fsas.Dim0()).  We don't
                         currently support having a_fsas.Dim0() > 1 and
                         b_fsas.Dim0() == 1, which is not a fundamental
                         limitation of the algorithm but it would require
                         code changes to support.
         @param[in] b_fsas   Input FSAs that correspond to neural network
                         outputs (see documentation in fsa.h).
         @param[in] search_beam   Beam for frame-synchronous beam pruning,
                    e.g. 20. Smaller is faster, larger is more exact
                    (less pruning). This is the default value; it may be
                    modified by {min,max}_active which dictate the minimum
                    or maximum allowed number of active states per frame.
         @param[in] output_beam   Beam with which we prune the output (analogous
                         to lattice-beam in Kaldi), e.g. 8.  We discard arcs in
                         the output that are not on a path that's within
                         `output_beam` of the best path of the composed output.
         @param[in] min_active  Minimum active states allowed per frame; beam
                         will be decreased if the number of active states falls
                         below this
         @param[in] max_active  Maximum active states allowed per frame.
                         (i.e. at each time-step in the sequences).  Sequence-
                         specific beam will be reduced if more than this number
                         of states are active.  The hash size used per FSA is 4
                         times (this rounded up to a power of 2), so this affects
                         memory consumption.
         @param[out] out Output vector of composed, pruned FSAs, with same
                         Dim0() as b_fsas.  Elements of it may be empty if the
                         composition was empty, either intrinsically or due to
                         failure of pruned search.
         @param[out] arc_map_a  Vector of

*/
void IntersectDensePruned(FsaVec &a_fsas, DenseFsaVec &b_fsas,
                          float search_beam, float output_beam,
                          int32_t min_active_states, int32_t max_active_states,
                          FsaVec *out, Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b);

/* IntersectDense is a version of IntersectDensePruned that does not
   do pruning in the 1st pass.
 */
void IntersectDense(FsaVec &a_fsas, DenseFsaVec &b_fsas,
                    float output_beam, FsaVec *out,
                    Array1<int32_t> *arc_map_a,
                    Array1<int32_t> *arc_map_b);


/*
  This is 'normal' intersection (we would call this Compose() for FSTs, but
  you can do that using Intersect(), by calling this and then Invert()
  (or just attaching the other aux_labels).


        @param [in] a_fsas  Fsa or FsaVec that is one of the arguments
                           for composition (i.e. 2 or 3 axes)
        @param [in] properties_a  properties for a_fsas. Will be computed
                           internally if its value is -1.
        @param [in] properties_b  properties for b_fsas. Will be computed
                           internally if its value is -1.
        @param [in] b_fsas  Fsa or FsaVec that is one of the arguments
                           for composition (i.e. 2 or 3 axes)
        @param [in] treat_epsilons_specially   If true, epsilons will
                          be treated as epsilon, meaning epsilon arcs
                          can match with an implicit epsilon self-loop.
                          If false, epsilons will be treated as real,
                          normal symbols (to have them treated as epsilons
                          in this case you may have to add epsilon self-loops
                          to whichever of the inputs is naturally epsilon-free).

         If treat_epsilons_specially is false, you'll get the correct results
         if one of the following conditions hold:

        - Any zero symbols in your graphs were never intended to be treated as
          epsilons, but have some other meaning (e.g. they are just regular
          symbols, or blank or something like that).
        - a_fsas or b_fsas are both epsilon free
        - One of a_fsas or b_fsas was originally epsilon-free, and you
          modified it with AddEpsilonSelfLoops()
        - Neither a_fsas or b_fsas was originally epsilon-free, you modified
          both of them by calling AddEpsilonSelfLoops(), and you don't care
          about duplicate paths with the same weight (search online
          for "epsilon-sequencing problem").

          If treat_epsilons_specially is true, you'll get the correct results
          except that our algorithm does not have an epsilon-sequencing
          filter, so you may get multiple equivalent paths corresponding
          to different orders of processing epsilons; for many
          applications this does not matter.

        @param [out] out    Output Fsa, will be an FsaVec (NumAxes() == 3)
                           regardless of the num_axes of the arguments.
        @param [out] arc_map_a  If not nullptr, this function will write to
                           here a map from (arc in `out`) to (arc in a_fsas).
        @param [out] arc_map_b  If not nullptr, this function will write to
                           here a map from (arc in `out`) to (arc in b_fsas).
        @return   Returns true if intersection was successful for all inputs
                  (requires input FSAs to be arc-sorted and at least one of
                  them to be epsilon free).
 */
bool Intersect(FsaOrVec &a_fsas, int32_t properties_a, FsaOrVec &b_fsas,
               int32_t properties_b, bool treat_epsilons_specially, FsaVec *out,
               Array1<int32_t> *arc_map_a, Array1<int32_t> *arc_map_b);

/*
    Remove epsilons (symbol zero) in the input Fsas, it works for both
    Fsa and FsaVec.
    @param [in] src   Source Fsa or FsaVec. Must be top-sorted as we will
                      compute forward and backward scores on it.
    @param [out] dest Destination; at exit will be equivalent to `src` under
                      tropical semiring but will be epsilon-free, i.e.
                      there's no arc in dest with arc.label==0, its
                      Properties() will contain kFsaPropertiesEpsilonFree.
    @param [out] arc_derivs  If not nullptr, at exit arc_dervis.NumAxes() == 2,
                      its rows are indexed by arc-indexes in `dest`. Row i in
                      arc_derivs is the sequence of arcs in `src` that arc
                      `i` in `dest` corresponds to; the weight of the arc in
                      `dest` will equal the sum of those input arcs' weights.
    Note we don't support pruning here.

    CAUTION: It only works for CPU;
 */
void RemoveEpsilon(FsaOrVec &src, FsaOrVec *dest,
                   Ragged<int32_t> *arc_derivs = nullptr);

/*
    Determinize the input Fsas, it works for both Fsa and FsaVec.
    @param [in] src   Source Fsa or FsaVec.
                      Expected to be epsilon free, but this is not checked;
                      in any case, epsilon will be treated as a normal
                      symbol. We also assume src is connected.
    @param [out] dest Destination; at exit will be equivalent to `src` under
                      tropical semiring but will be deterministic, i.e.
                      there are no duplicate labels for arcs leaving a given
                      state; if you call ArcSort on `dest`, then the
                      ans's Properties() will contain
                      kFsaPropertiesArcSortedAndDeterministic.
    @param [out] arc_derivs  If not nullptr, at exit arc_dervis.NumAxes() == 2,
                      its rows are indexed by arc-indexes in `dest`. Row i in
                      arc_derivs is the sequence of arcs in `src` that arc
                      `i` in `dest` corresponds to; the weight of the arc in
                      `dest` will equal the sum of those input arcs' weights.
    Note we don't support pruning here.

    CAUTION: It only works for CPU;
 */
void Determinize(FsaOrVec &src, FsaOrVec *dest,
                 Ragged<int32_t> *arc_derivs = nullptr);

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

/* Compute the forward shortest path in the tropical semiring.

   @param [in] fsas  Input FsaVec (must have 3 axes).  Must be
                 top-sorted and without self loops, i.e. would have the
                 property kFsaPropertiesTopSortedAndAcyclic if you were
                 to compute properties.

   @param [in] entering_arcs   An array indexed by state_idx01 into `fsas`,
                saying which arc_idx012 is the best arc entering it,
                or -1 if there is no such arc.

   @return returns a tensor with 2 axes indexed by [fsa_idx0][arc_idx012]
           containing the best arc indexes of each fsa.
 */
Ragged<int32_t> ShortestPath(FsaVec &fsas,
                             const Array1<int32_t> &entering_arcs);

/* Compute the union of all fsas in a FsaVec.

   @param [in]  fsas       Input FsaVec (must have 3 axes). Every fsa must
                           be non-empty.
   @param [out] arc_map    It maps the arc indexes of the output fsa
                           to the input fsas. That is,
                           arc_map[out_arc_index] = fsas_arc_index.
                           Its entry may be -1 if there is no
                           corresponding arc in fsas.
   @return  returns an FSA that is the union of the input FSAs.
 */
Fsa Union(FsaVec &fsas, Array1<int32_t> *arc_map = nullptr);

/* Compute the closure of an FSA.

   The closure is implemented in the following way:

   - For all arcs entering the final-state, change the destination-state from
   the final-state to state 0 (the start state), and the label from -1 to 0.

   - Add an arc from the start-state to the final-state (with label -1, and
   score 0, of course); this can be the last numbered arc of the start state of
   the output. The implementation wouldn't be in-place, we'd create a new FSA.
   Assume that the original start state has two arcs, then the arc-map would be
   [ 0 1 -1 2 3 ... ].

   Note: The output will not be epsilon-free though, so if an epsilon-free
   output is desired, which it generally will be, the caller will need to
   RemoveEpsilon afterward.

   Caution: The caller will have to modify any extra labels (like aux_labels) to
   deal with -1's correctly.

   @param in]   fsa        The input FSA. Must have NumAxes() == 2, i.e., it
                           must be a single FSA.
   @param [out] arc_map    It maps the arc indexes of the output fsa
                           to the input fsa. That is,
                           arc_map[out_arc_index] = fsa_arc_index.
                           Its entry may be -1 if there is no
                           corresponding arc in fsa.
   @return It returns a closure of the input fsa.
 */
Fsa Closure(Fsa &fsa, Array1<int32_t> *arc_map = nullptr);

}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
