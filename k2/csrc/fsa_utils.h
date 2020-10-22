/**
 * @brief Utilities for creating FSAs.
 *
 * Note that serializations are done in Python.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *
 * @copyright
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
   Compute and return forward scores per state (like alphas in Baum-Welch),
   or forward best-path scores if log_semiring == false.

      @param [in] fsas  Input Fsa or FsaVec (must have 3 axes).  Must be
                 top-sorted and without self loops, i.e. would have the
                 property kFsaPropertiesTopSortedAndAcyclic if you were
                 to compute properties.
      @param [in] state_batches  Batches of states, as returned by
                 GetStateBatches(fsas, true) (must have 3 axes:
                 [iter][fsa][state_list]).
      @param [in] entering_arc_batches  Arcs-indexes (idx012's in fsas) of arcs
                 entering states in `state_batches`, indexed
                 [iter][fsa][state_list][arc_list], as returned by
                 EnteringArcIndexBatches().
      @param [in] log_semiring   If true, combine path with LogAdd
                  (i.e., mathematically, `log(exp(a)+exp(b))`); if false,
                   combine as `max(a,b)`.
      @return   Returns vector indexed by state-index (idx01 into fsas), i.e.
               `ans.Dim()==fsas.TotSize(1)`, containing forward scores.
                (these will be zero for the start-states).
*/
template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, Ragged<int32_t> &state_batches,
                                   Ragged<int32_t> &entering_arc_batches,
                                   bool log_semiring);

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
       @param [in] fsas  Input FsaVec (must have 3 axes)
       @param [in] state_batches  Batches of states, as returned by
                  GetBatches (must have 3 axes: [iter][fsa][state_list]).
       @param [in] leaving_arc_batches  Arcs-indexes (idx012's in fsas) of arcs
                 leaving states in `state_batches`, indexed
                 [iter][fsa][state_list][arc_list], as returned by
                 LeavingArcIndexBatches().
       @param [in] tot_scores  If provided, we'll treat the backward
                  scores of final-states as the negative of these
                  tot_scores (which must have
                  `tot_scores->Dim() == fsas.Dim0())`; otherwise
                  as zero.
       @param [in] log_semiring  If true, use LogAdd to combine
                  scores; if false, use max.
       @return  Returns a vector indexed by state-index (idx01 in fsas), with
               `ans.Dim() == fsas.TotSize(1)`, containing backward
               scores.
 */
template <typename FloatType>
Array1<FloatType> GetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches,
    const Array1<FloatType> *tot_scores = nullptr, bool log_semiring = true);

/*
  Compute and return arc-level forward-backward scores, which are:
   `forward_score[src_state] + arc.score + backward_score[dest_state]`.

   If you provided the `tot_scores` argument to GetBackwardScores, and if
   log_semiring == true, then you can think of these as the log probability that
   you go through that arc, which would be log(1.0) = 0.0 for an FSA with only
   one path.

       @param [in] fsas   The FSAs that we want the arc-level probabilities
                         from
       @param [in] forward_scores  The state-level forward scores, which
                        should have been computed by GetForwardScores() with
                        the same `fsas` and `log_semiring` that
                        GetBackwardScores used to compute `backward_scores`.
       @param [in] backward_scores  The state-level backward scores, which
                        should have been computed using GetBackwardScores()
                        with the same `fsas` and `log_semiring` that
                        GetForwardScores used to compute `forward_scores`.
       @return    returns scores for arcs, indexed by arc_idx012 in `fsas`,
                  with ans.Dim() == fsas.NumElements().
*/
template <typename FloatType>
Array1<FloatType> GetArcScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores,
                               const Array1<FloatType> &backward_scores);

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

}  // namespace k2

#define IS_IN_K2_CSRC_FSA_UTILS_H_
#include "k2/csrc/fsa_utils_inl.h"
#undef IS_IN_K2_CSRC_FSA_UTILS_H_

#endif  //  K2_CSRC_FSA_UTILS_H_
