/**
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *                      Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_FSA_UTILS_H_
#define K2_CSRC_FSA_UTILS_H_

#include <string>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"

namespace k2 {

/*
  Create an Fsa from string.

  The string `s` consists of lines. Every line, except the line for
  the final state, has one of the following two formats:

  (1)
      src_state dest_state label cost
  which means the string represents an acceptor.

  (2)
      src_state dest_state label aux_label cost
  which indicates the string is a transducer.

  The line for the final state has the following format when `openfst` is false:

      final_state

  This is because final state in k2 does not bear a cost. Instead, we put the
  cost on the arc that connects to the final state, and set its label to -1.
  When `openfst` is true, we expect the more generic OpenFst sytle final state
  format :

      final_state cost

  And we allow more than one final states when `openfst` is true.

  Note that fields are separated by spaces and tabs. There can exist
  multiple tabs and spaces.

  CAUTION: The first column has to be in non-decreasing order.

  @param [in]   s   The input string. See the above description for its format.
  @param [in]   openfst
                    If true, the string form has the weights as costs, not
                    scores, so we negate them as we read. We will also allow
                    multiple final states with weights associated with them.
  @param [out]  aux_labels
                    If NULL, we treat the input as an acceptor; otherwise we
                    treat the input as an transducer, and store the
                    corresponding output labels to it. It is allocated inside
                    the function and will contain aux_label of each arc.
                    Note that it is allocated on CPU if needed.

  @return It returns an Fsa on CPU.
 */
Fsa FsaFromString(const std::string &s, bool openfst = false,
                  Array1<int32_t> *aux_labels = nullptr);

/* Convert an FSA to a string.

   If the FSA is an acceptor, i.e., aux_labels == nullptr,  every arc
   is converted to a line with the following form:

      src_state dest_state label score

   If the FSA is a transducer, i.e., aux_labels != nullptr, every arc
   is converted to a lien with the following form:

      src_state dest_state label aux_label score

   The last line of the resulting string contains:

      final_state

   NOTE: Fields are separated by only ONE space.
   There are no leading or trailing spaces.

   NOTE: If `openfst` is true, scores are first negated and then printed.

   @param [in]  fsa   The input FSA

   @param [in]  openfst
                      If true, the scores will first be negated and
                      then printed.
   @param in]   aux_labels
                      If not NULL, the FSA is a transducer and it contains the
                      aux labels of each arc.
 */
std::string FsaToString(const Fsa &fsa, bool openfst = false,
                        const Array1<int32_t> *aux_labels = nullptr);

/*  Returns a renumbered version of the FsaVec `src`.
      @param [in] src    An FsaVec, assumed to be valid, with NumAxes() == 3
      @param [in] order  An ordering of states in `src` (contains idx01's in
                    `src`).  Does not need to contain all states in `src`.  The
                    order of FSAs in `src` must not be changed by this: i.e., if
                    we get the old fsa-index for each element of `order`, they
                    must be non-decreasing.
                    Caution: the function will abort with an error if the
                    dest_state of an arc in the original Fsa is not kept in
                    the output Fsa (i.e.  the dest_state is not in `order`).
                    Noted if the number of states in `order` for an input Fsa
                    is not zero, then it must be at least 2 (and those states
                    must contain the start state and final state in the input
                    FSA), but we never check this in the function.
      @param [out] arc_map  If non-NULL, this will be set to a new Array1 that
                   maps from arcs in the returned FsaVec to the original
                   arc-index in `fsas`.
      @return  Returns renumbered FSA.
*/
FsaVec RenumberFsaVec(FsaVec &src, const Array1<int32_t> &order,
                      Array1<int32_t> *arc_map);

/*
  Returns a ragged tensor representing batches of states in top-sorted FSAs
  `fsas` which can be processed sequentially with each batch of states only
  having arcs to later-numbered batches.

      @param [in] fsas  Input FSAs, with NumAxes() == 3.  Must have property
                kFsaPropertiesTopSortedAndAcyclic, although this is not
                checked.  (Note: this really just means top-sorted, as a truly
                top-sorted FSA cannot have cycles).
      @param [in] transpose    If true, the result will be indexed
                [batch][fsa_idx][state]; if false, it will be
                indexed [fsa_idx][batch][state].

      @return  Returns batches of states (contains idx01s into `fsas`).
             Normally (transpose==true) these will be indexed
             [batch][fsa][state_list]
 */
Ragged<int32_t> GetStateBatches(FsaVec &fsas, bool transpose = true);

/*
  Returns a ragged tensor of arc-indexes of arcs leaving states in `fsas`, in
  batches that can be processed sequentially in top-sorted FSAs.

     @param [in] fsas   Input FSAs, with `fsas.NumAxes() == 3`.
     @param [in] state_batches  Batches of states as returned from
               `GetStateBatches(fsas, true)`, indexed
               [batch][fsa][state_list].
     @return   Returns a tensor with `ans.NumAxes() == 4`, containing
               arc_idx012's into `fsas`.  Axes 0,1,2 correspond to
               those of `state_batches`; the last axis is a list of
               arcs, i.e. the indexing is [batch][fsa][state_list][arc_list]
 */
Ragged<int32_t> GetLeavingArcIndexBatches(FsaVec &fsas,
                                          Ragged<int32_t> &state_batches);

/*
 Returns a ragged tensor of the arc-indexes of arcs entering states in `fsas`,
 in batches that can be processed sequentially in top-sorted FSAs.

     @param [in] fsas  Input FSAs, with `fsas.NumAxes() == 3`.
     @param [in] incoming_arcs  A tensor containing the arc-indexes of the
                       arcs entering each state in `fsas`, indexed
                       [fsa][state][arc_list].
     @param [in] state_batches  Batches of states as returned from
                  `GetStateBatches(fsas, true)`, indexed
                  [batch][fsa][state_list].
     @return   Returns a tensor with `ans.NumAxes() == 4`, containing
               arc_idx012's into `fsas`.  Axes 0,1,2 correspond to
               those of `state_batches`; the last axis is a list of
               arcs, i.e. the indexing is [batch][fsa][state_list][arc_list]

 */
Ragged<int32_t> GetEnteringArcIndexBatches(FsaVec &fsas,
                                           Ragged<int32_t> &incoming_arcs,
                                           Ragged<int32_t> &state_batches);

/*
  Returns a ragged tensor of incoming arc-indexes for the states in `fsas`.

       @param [in] fsas          Input FsaVec (must have 3 axes)
       @param [in] dest_states   Array of destination-states of each arc, in the
                    idx01 format (i.e. idx01's of dest states), as returned by
                   `GetDestStates(fsas, true)`.
       @return     Returns a tensor with 3 axes, indexed
                   [fsa][state][list_of_arcs], containing the idx012's of arcs
                   entering states in `fsas`  Its values will be a permutation
                   of the numbers 0 through fsas.NumElements() - 1.
 */
Ragged<int32_t> GetIncomingArcs(FsaVec &fsas,
                                const Array1<int32_t> &dest_states);

/*
  Rearrange an FsaVec into a different arrangement of arcs which will actually
  not be a valid FsaVec but will contain the same information in a different
  form.. the arcs are rearranged so they are listed by the dest_state, not
  src_state.  Implementation is 2 lines, using GetIncomingArcs().
 */
FsaVec GetIncomingFsaVec(FsaVec &fsas);

/*
   Compute and return forward scores per state (like alphas in Baum-Welch),
   or forward best-path scores if log_semiring == false.

      @param [in] fsas  Input FsaVec (must have 3 axes).  Must be
                 top-sorted and without self loops, i.e. would have the
                 property kFsaPropertiesTopSortedAndAcyclic if you were
                 to compute properties.
      @param [in] state_batches  Batches of states, as returned by
                 GetStateBatches(fsas, true) (must have 3 axes:
                 [iter][fsa][state_list]).
      @param [in] entering_arc_batches  Arcs-indexes (idx012's in fsas) of arcs
                 entering states in `state_batches`, indexed
                 [iter][fsa][state_list][arc_list], as returned by
                 GetEnteringArcIndexBatches().
      @param [in] log_semiring   If true, combine path with LogAdd
                  (i.e., mathematically, `log(exp(a)+exp(b))`); if false,
                   combine as `max(a,b)`.
      @param [out,optional] entering_arcs   If non-NULL and if
                log_semiring == false, will be set to
                a new Array1, indexed by state_idx01 into `fsas`,
                saying which arc_idx012 is the best arc entering it,
                or -1 if there is no such arc.  It is an error if this
                is non-NULL and log_semiring == true.
      @return   Returns vector indexed by state-index (idx01 into fsas), i.e.
               `ans.Dim()==fsas.TotSize(1)`, containing forward scores.
                (these will be zero for the start-states).

    CAUTION: there is another version of GetForwardScores() for CPU only,
    declared in host_shim.h.
*/
template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, Ragged<int32_t> &state_batches,
                                   Ragged<int32_t> &entering_arc_batches,
                                   bool log_semiring,
                                   Array1<int32_t> *entering_arcs = nullptr);

/*
  Does the back-propagation for GetForwardScores().
     @param [in] fsas           Same object given to GetForwardScores()
     @param [in] state_batches   Same object given to GetForwardScores()
     @param [in] leaving_arc_batches  Arc-indexes (idx012's in fsas) of arcs
                leaving states in `state_batches`, indexed
                [iter][fsa][state_list][arc_list], as returned by
                GetLeavingArcIndexBatches().  CAUTION: not the same
                as entering_arc_batches as used in GetForwardScores().
     @param [in] log_semiring    Same option as given to GetForwardScores()
     @param [in] entering_arcs   Only if log_semiring is false, we require this
                                 to be supplied (i.e. not nullptr).  It must be
                                 the array that was output by
                                 GetForwardScores().
     @param [in] forward_scores  The return value of GetForwardScores().
     @param [in] forward_scores_deriv  The derivative of the loss function
                                 w.r.t. `forward_scores` (i.e. the return
                                 value of GetForwardScores()).
     @return  Returns the derivative of the loss function w.r.t. the scores
                               of the `fsas` argument to GetForwardScores().
 */
template <typename FloatType>
Array1<FloatType> BackpropGetForwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches,
    bool log_semiring,
    const Array1<int32_t> *entering_arcs,
    const Array1<FloatType> &forward_scores,
    const Array1<FloatType> &forward_scores_deriv);

/*
  Return array of total scores (one per FSA), e.g. could be interpreted as
  the data probability or partition function.
         @param [in] fsas   Input FsaVec (must have 3 axes)
         @param [in] forward_scores  Array of forward scores, as returned
                          by GetForwardScores with the same FsaVec `fsas`.
         @return  Returns array of total scores, of dimension fsas.Dim0(),
                   which will contain the scores in the final-states of
                   `forward_scores`, or -infinity for FSAs that had no
                   states.
*/
template <typename FloatType>
Array1<FloatType> GetTotScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores);

/*
   Compute and return backward scores per state (like betas in Baum-Welch),
   or backward best-path scores if log_semiring == false.
      @param [in] fsas  Input FsaVec (must have 3 axes).  Must be
                 top-sorted and without self loops, i.e. would have the
                 property kFsaPropertiesTopSortedAndAcyclic if you were
                 to compute properties.
       @param [in] state_batches  Batches of states, as returned by
                   GetStateBatches(fsas, true) (must have 3 axes:
                   [iter][fsa][state_list]).
       @param [in] leaving_arc_batches  Arcs-indexes (idx012's in fsas) of arcs
                 leaving states in `state_batches`, indexed
                 [iter][fsa][state_list][arc_list], as returned by
                 GetLeavingArcIndexBatches().
       @param [in] log_semiring  If true, use LogAdd to combine
                 scores; if false, use max.
       @return  Returns a vector indexed by state-index (idx01 in fsas), with
                `ans.Dim() == fsas.TotSize(1)`, containing backward
                scores.

     CAUTION: there is another version of GetBackwardScores() for CPU only,
     declared in host_shim.h.
 */
template <typename FloatType>
Array1<FloatType> GetBackwardScores(FsaVec &fsas,
                                    Ragged<int32_t> &state_batches,
                                    Ragged<int32_t> &leaving_arc_batches,
                                    bool log_semiring = true);

/*
   Back-propagates through GetBackwardScores().

      @param [in] fsas  Input FsaVec, as given to GetBackwardScores()
      @param [in] state_batches  Batches of states, as given to
                  GetBackwardScores() and GetForwardScores()
      @param [in] entering_arc_batches  Arcs-indexes (idx012's in fsas) of arcs
                 entering states in `state_batches`, indexed
                 [iter][fsa][state_list][arc_list], as returned by
                 GetEnteringArcIndexBatches().
      @param [in] log_semiring  The same option as given to GetBackwardScors()
      @param [in] backward_scores   The return value of GetBackwardScores()
      @param [in] backward_scores_deriv  The derivative of the loss function
                            w.r.t. the return value of `GetBackwardScores()`
      @return  Returns the derivative of the loss function w.r.t. the scores
               of the input arg `fsas` to this function.
 */
template <typename FloatType>
Array1<FloatType> BackpropGetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &entering_arc_batches, bool log_semiring,
    const Array1<FloatType> &backward_scores,
    const Array1<FloatType> &backward_scores_deriv);

/*
  Compute and return arc-level posterior scores which are:
  `forward_score[src_state] + arc.score + backward_score[dest_state] -
  tot_score[fsa]`, where tot_score[fsa] is computed as the average of the
  forward score for the final state of that FSA and the backward score of the
  initial state.

   You can think of the result as the log probability that you go through that
   arc, which would be log(1.0) = 0.0 for an FSA with only one path.

       @param [in] fsas   The FSAs that we want the arc-level probabilities
                         from
       @param [in] forward_scores  The state-level forward scores, which
                        should have been computed by GetForwardScores() with
                        the same `fsas` and log_semiring that
                        GetBackwardScores used to compute `backward_scores`.
       @param [in] backward_scores  The state-level backward scores, which
                        should have been computed using GetBackwardScores()
                        with the same `fsas` and log_semiring that
                        GetForwardScores used to compute `forward_scores`.
       @return    returns scores for arcs, indexed by arc_idx012 in `fsas`,
                  with ans.Dim() == fsas.NumElements().
*/
template <typename FloatType>
Array1<FloatType> GetArcPost(FsaVec &fsas,
                             const Array1<FloatType> &forward_scores,
                             const Array1<FloatType> &backward_scores);

/*
  Does the backprop for GetArcPost(), outputting the deriv of the loss
  function w.r.t the `forward_scores` and `backward_scores` args to
  GetArcPost().
       @param [in] fsas  The FSAs we're getting scores from, the same as the
                        original arg to GetArcPost().
       @param [in] incoming_arcs   The result of calling
                       `GetIncomingArcs(fsas, GetDestStates(fsas, true))`
       @param [in] arc_post_deriv  The derivative of the loss function
                       w.r.t. the return value of `GetArcPost()`
       @param [out] forward_scores_deriv  The derivative of the loss function
                       w.r.t. the input `forward_scores` to GetArcPost()
                       will be written to here.
       @param [out] backward_deriv  The derivative of the loss function
                       w.r.t. the input `backward_scores` to GetArcPost()
                      will be written to here.
 */
template <typename FloatType>
void BackpropGetArcPost(FsaVec &fsas, Ragged<int32_t> &incoming_arcs,
                        const Array1<FloatType> &arc_post_deriv,
                        Array1<FloatType> *forward_scores_deriv,
                        Array1<FloatType> *backward_scores_deriv);

/*
  Returns an array of the destination-states for all arcs in an FsaVec

     @param [in] fsas       Source FsaVec; must have NumAxes() == 3.
     @param [in] as_idx01   If true, return dest-states in the idx01 format
                            instead of idx1. (See "Index naming scheme"
                            in utils.h).
     @return                Returns a vector with dimension equal to the
                            number of arcs in `fsas` (i.e. fsas.NumElements()),
                            containing idx01's of dest-states if as_
                            idx01 == true, else idx1's of dest-states.
                            Note: if you want this as a ragged tensor you can
                            use the constructor:
                            Ragged<int32>(fsas.shape,
                                          GetDestStates(fsas, as_idx01))
*/
Array1<int32_t> GetDestStates(FsaVec &fsas, bool as_idx01);

/*
  Convert a DenseFsaVec to an FsaVec.  Intended for use in testing code.

     @param [in] src  DenseFsaVec to convert
     @return          Returns the DenseFsaVec converted to FsaVec format.

  TODO(Dan): maybe at some point add an arc_map argument which will enable
  testing of arc_map-related things.
 */
FsaVec ConvertDenseToFsaVec(DenseFsaVec &src);

/*
  Return a random Fsa, with a CPU context. Intended for testing.

     @param [in] acyclic     If true, generated Fsa will be acyclic.
     @param [in] max_symbol  Maximum symbol on arcs. Generated arcs' symbols
                             will be in range [-1,max_symbol], note -1 is
                             kFinalSymbol; must be at least 0;
     @param [in] min_num_arcs Minimum number of arcs; must be at least 0.
     @param [in] max_num_arcs Maximum number of arcs; must be >= min_num_arcs.
 */
Fsa RandomFsa(bool acyclic = true, int32_t max_symbol = 50,
              int32_t min_num_arcs = 0, int32_t max_num_arcs = 1000);
/*
  Return a random FsaVec, with a CPU context. Intended for testing.

     @param [in] min_num_fsas Minimum number of fsas we'll generated in the
                              returned FsaVec;  must be at least 1.
     @param [in] max_num_fsas Maximum number of fsas we'll generated in the
                              returned FsaVec; must be >= min_num_fsas.
     @param [in] acyclic     If true, generated Fsas will be acyclic.
     @param [in] max_symbol  Maximum symbol on arcs. Generated arcs' symbols
                             will be in range [-1,max_symbol], note -1 is
                             kFinalSymbol; must be at least 0;
     @param [in] min_num_arcs Minimum number of arcs in each Fsa;
                              must be at least 0.
     @param [in] max_num_arcs Maximum number of arcs in each Fsa;
                              must be >= min_num_arcs.
 */
FsaVec RandomFsaVec(int32_t min_num_fsas = 1, int32_t max_num_fsas = 1000,
                    bool acyclic = true, int32_t max_symbol = 50,
                    int32_t min_num_arcs = 0, int32_t max_num_arcs = 1000);

/*
  Return a randomly generated DenseFsaVec.  It will have -infinities in the
  expected places (see documentation of DenseFsaVec).

           @param [in] min_num_fsas  Minimum value of ans.shape.Dim0()
           @param [in] max_num_fsas  Minimum value of ans.shape.Dim0()
           @param [in] min_frames    Minimum number of frames *per sequence*,
                                     not counting the frame corresponding to the
                                     final-symbol -1.
           @param [in] max_frames    Maximum number of frames *per sequence*.
           @param [in] min_nsymbols  Minimum number of symbols, including
                                     epsilon but not the final-symbol -1,
                                     so the scores.Dim1()
                                     will be >= min_symbols + 1.
           @param [in] max_nsymbols  Maximum number of symbols; scores.Dim1()
                                     will be <= max_symbols + 1.
           @param [in] scores_stddev  Scaling factor used on the scores
           @returns   Returns a random FsaVec, on CPU.
 */
DenseFsaVec RandomDenseFsaVec(int32_t min_num_fsas = 1,
                              int32_t max_num_fsas = 10, int32_t min_frames = 0,
                              int32_t max_frames = 20, int32_t min_nsymbols = 1,
                              int32_t max_nsymbols = 50,
                              float scores_scale = 1.0);

/*
  Create and return tensor containing the start-states of `src`, indexed by
  FSA (so ans.Dim0() == src.Dim0()) and then a list of 0 or 1 states.
  (There will be no element present for any source FSA that has no states.)

    @param [in]  src   Source FsaVec; must have 3 axes
    @return            Returns ragged array containing indexes of
                       start-states in `src` (i.e. of type state_idx01).
                       It will satisfy ans.NumAxes() == 2,
                       ans.Dim0() == src.Dim0(),
                       ans.NumElements() <= ans.Dim0().
 */
Ragged<int32_t> GetStartStates(FsaVec &src);

/* Create a FsaVec from a tensor of best arc indexes returned by `ShortestPath`.

   @param [in] fsas   Input FsaVec. It must be the same FsaVec for getting
                     `best_arc_indexes`.
   @param [in] best_arc_index
                      As returned by `ShortestPath`; has 2 axes, indexed
                      [fsa_idx][list of arcs]


   @return returns a linear FsaVec that contains the best path of `fsas`.
 */
FsaVec FsaVecFromArcIndexes(FsaVec &fsas, Ragged<int32_t> &best_arc_indexes);

/*
  Compose arc maps from two successive FSA operations which give arc maps as
  type ragged tensor.

   @param [in] step1_arc_map   Arc map from the first Fsa operation.
                               Must have NumAxes() == 2.
   @param [in] step2_arc_map   Arc map from the second Fsa operation.
                         The elements in it are indexes into `step1_arc_map`,
                         so we require that
                         `0 <= step2_arc_map.values[i] < step1_arc_map.Dim0()`
                         for `0 <= i < step2_arc_map.NumElements()`.
                         Must have NumAxes() == 2 and have the same device
                         type as `step1_arc_map`.

   @return Returns the composed arc map. ans.NumAxes() == 2 and ans.Dim0()
           equals to `step2_arc_map.Dim0()`. Suppose elements in row i of
           `step2_arc_map` are [2, 3, 7], then the elements in row i of ans
           will be the concatenation of elements in row 2, 3, 7 of
           `step1_arc_map`.
 */
Ragged<int32_t> ComposeArcMaps(Ragged<int32_t> &step1_arc_map,
                               Ragged<int32_t> &step2_arc_map);


/*
  Return a ragged array that represents the cumulative distribution function
  (cdf) of the probability of arcs leaving each state of `fsas`.
  This is according to the distribution implied by the arc posteriors
  in `arc_post`.  It's intended so that given a distribution over
  arc probabilities you can prepare to call RandomPaths() to select
  arcs according to that probability distribution.


    @param [in] fsas Fsa or FsaVec for which we want the cdf.
    @param [in] arc_post  Arc-level posteriors for this FsaVec, probably
                   from GetArcPost().  (Probably only makes sense if you had
                   log_semiring=true when getting the forward and backward
                   scores, but would still give you something otherwise.)

    @return   Returns an Array<FloatType> with ans.Dim() == fsas.NumElements();
                  the element corresponding the 1st arc leaving any state
                  will always be 0.0, and the rest will be non-decreasing,
                  representing the exclusive-sum of prior members of
                  `arc_post` leaving that state, all divided by the sum
                  of `arc_post` leaving that state; you can imagine
                  that there is an implicit "last element" for each state
                  that is equal to 1.0.  We are careful to eliminate
                  roundoff errors of a type that would cause fatal
                  errors in sampling.
 */
template <typename FloatType>
Array1<FloatType> GetArcCdf(FsaOrVec &fsas,
                            Array1<FloatType> &arc_post);

/*
  Return pseudo-randomly chosen paths through acyclic FSAs.  (Actually the paths
  are deterministic, taken at fixed intervals through a certain cdf).

    @param [in] fsas  An FsaVec (3 axes) that we are sampling from.
    @param [in] arc_cdf  The result of calling GetArcCdf() with `fsas`.
    @param [in] num_paths  An array giving the number of paths that is
                        requested for each Fsa, with num_paths.Dim() ==
                        fsas.Dim0().  Must be zero for any Fsa that
                        is equivalent to the empty Fsa (e.g. its
                        GetTotScores() entry is -infinity).
   @param [in] state_batches  The result of calling GetStateBatches(fsas, true).
                        Is needed so we can know the maximum
                        possible length of each path, to know how much memory to
                        allocate.

   @return  Returns a ragged tensor with 3 axes: [fsa][path][arc],
            containing arc-indexes (idx012) into `fsas`,
             with `ans.Dim0() == fsas.Dim0()`,
            `ans.TotSize(1) == Sum(num_paths)`.  Each bottom-level
            sub-list is a list of consecutive arcs from the start-state
            to the final-state.

  See also the other form of RandomPaths(), which allows you to provide
  `num_paths` as a scalar.
 */
template <typename FloatType>
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<FloatType> &arc_cdf,
                            const Array1<int32_t> &num_paths,
                            Ragged<int32_t> &state_batches);


/*
  Return pseudo-randomly chosen paths through acyclic FSAs.  (Actually the paths
  are deterministic, taken at fixed intervals through a certain cdf).

    @param [in] fsas  An FsaVec (3 axes) that we are sampling from.
    @param [in] arc_cdf  The result of calling GetArcCdf() with `fsas`.
    @param [in] num_paths  The number of paths requested for those FSAs
                       that have successful paths through them.  (For other
                       FSAs, no paths will be returned).
    @param [in] tot_scores  Total score of each FSA in `fsas`, as returned
                      by GetTotScores (semiring of forward_scores does not
                      matter).  Is needed so we can know which FSAs
                      had successful paths, i.e. had tot_score not equal to
                      -infinity.
    @param [in] state_batches  The result of calling GetStateBatches(fsas, true)
                      on `fsas`.  Is needed so we can know the maximum
                      possible length of each path, to know how much memory to
                      allocate.

   @return  Returns a ragged tensor with 3 axes: [fsa][path][arc],
            containing arc-indexes (idx012) into `fsas`,
             with `ans.Dim0() == fsas.Dim0()`.  Each bottom-level
            sub-list is a list of consecutive arcs from the start-state
            to the final-state.


  See also the other form of RandomPaths(), which allows you to provide
  `num_paths` separately for each Fsa.
 */
template <typename FloatType>
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<FloatType> &arc_cdf,
                            int32_t num_paths,
                            const Array1<FloatType> &tot_scores,
                            Ragged<int32_t> &state_batches);



/*
  This function detects if there are any FSAs in an FsaVec that have exactly one
  state (which is not allowed; the empty FSA may have either 0 or 2 states); and
  it removes those states.  These states cannot have any arcs leaving them; if
  they do, it is an error and this function may crash or give undefined output.

    @param [in,out] fsas  FsaVec to possibly modify; must have 3 axes.

  CAUTION: this is not used right now and I'm not sure if there are any
  situations where it really should be used; think carefully before using it.
 */
void FixNumStates(FsaVec *fsas);

}  // namespace k2

#endif  //  K2_CSRC_FSA_UTILS_H_
