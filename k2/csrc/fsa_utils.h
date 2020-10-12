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
                   maps from arcs in the returned FsaVec to the original arc-index
                   in `fsas`.
      @return  Returns renumbered FSA.
*/
FsaVec RenumberFsaVec(FsaVec &fsas, const Array1<int32_t> &order,
                      Array1<int32_t> *arc_map);


/*
  Returns a ragged array representing batches of states in top-sorted FSAs
  `fsas` which can be processed sequentially with each batch of states only
  having arcs to later-numbered batches.

      @param [in] fsas  Input FSAs.  Must have property
                kFsaPropertiesTopSortedAndAcyclic, although this is not
                checked.  (Note: this really just means top-sorted, as a truly
                top-sorted FSA cannot have cycles).
      @param [in] transpose    If true, the result will be indexed [batch][fsa_idx][state];
                if false, it will be indexed [fsa_idx][batch][state].

      @return  Returns batches
 */
Ragged<int32_t> GetBatches(FsaVec &fsas, bool transpose = true);


/*
  Returns an array of the destination-states for all arcs in an FsaVec

     @param [in] fsas       Source FsaVec; must have NumAxes() == 3.
     @param [in] as_idx01   If true, return dest-states in the idx01 format instead of idx1.
                            (See "Index naming scheme" in utils.h).
     @return                Returns a vector with dimension equal to the
                            number of arcs in `fsas` (i.e. fsas.NumElements()),
                            containing idx01's of dest-states if as_idx01 == true,
                            else idx1's of dest-states.
     Note: if you want this as a ragged tensor you can use the constructor:
     Ragged<int32>(fsas.shape, GetDestStates(fsas, as_idx01))
*/
Array1<int32_t> GetDestStates(FsaVec &fsas, bool as_idx01);




}  // namespace k2

#endif  //  K2_CSRC_FSA_UTILS_H_
