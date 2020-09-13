/**
 * @brief
 * aux_labels
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_AUX_LABELS_H_
#define K2_CSRC_HOST_AUX_LABELS_H_

#include <vector>

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"

namespace k2host {

/*
  This header contains utilities for dealing with auxiliary labels on FSAs.

  These auxiliary labels can be used where you really have a transducer (i.e. to
  store the ilabels or olabels, whichever of the two is not participating
  directly in the operation you are doing).

  We deal with two formats of labels: a vector of int32_t, one per arc,
  for cases where we know we have at most one label per arc; and
  struct AuxLabels for cases where there may in general be a sequence
  of labels (one per arc).

*/

/*
  This allows you to store auxiliary labels (e.g. olabels or ilabels)
  on each arc of an Fsa.

  auto &start_pos = AuxLabels::Indexes;

     Suppose this is associated with an Fsa f.  start_pos will be of
     size f.arcs.size() + 1; start_pos[i] is the start position in
     `labels` of the label sequence on arc i.  start_pos.end()
     equals labels.size().

  auto &labels = AuxLabels::data;

     For arc i, (labels[start_pos[i] ], labels[start_pos[i]+1], ...
     labels[start_pos[i+1]-1]) are the list of labels on that arc.
     We treat epsilon the same as other symbols here, so there are no
     requirements on elements of `labels`.
 */
using AuxLabels = Array2<int32_t *, int32_t>;

/*
  Maps auxiliary labels after an FSA operation where each arc in the output
  FSA corresponds to exactly one arc in the input FSA.
 */
class AuxLabels1Mapper {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] labels_in  Labels on the arcs of the input FSA
     @param [in] arc_map    At entry `arc_map.size` equals to num-arcs of
                            the output Fsa, `arc_map.data[i]` gives which arc of
                            the input FSA that arc i in the output FSA
                            corresponds to.
  */
  AuxLabels1Mapper(const AuxLabels &labels_in, const Array1<int32_t *> &arc_map)
      : labels_in_(labels_in), arc_map_(arc_map) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] aux_size   The number of lists in the output AuxLabels
                                (equals num-arcs in the output FSA) and
                                the number of elements (equals num-aux-labels
                                on the arcs in the output FSA) will be written
                                to here.
  */
  void GetSizes(Array2Size<int32_t> *aux_size) const;

  /*
    Finish the operation and output auxiliary labels to `labels_out`.
       @param [out]  labels_out  Auxiliary labels on the arcs of the output FSA.
                                 Must be initialized; search for 'initialized
                                 definition' in class Array2 in array.h for
                                meaning.
   */
  void GetOutput(AuxLabels *labels_out);

 private:
  const AuxLabels &labels_in_;
  const Array1<int32_t *> &arc_map_;
};

/*
  Maps auxiliary labels after an FSA operation where each arc in the output
  FSA can correspond to a sequence of arcs in the input FSA.
 */
class AuxLabels2Mapper {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] labels_in  Labels on the arcs of the input FSA
     @param [in] arc_map    At entry `arc_map.size1` equals to num-arcs of
                            the output FSA. `arc_map.data[arc_map.indexes[i]]`
                            through `arc_map.data[arc_map.indexes[i+1] - 1]`
                            gives the sequence of arc-indexes in the input
                            FSA that arc i in the output FSA corresponds to.
  */
  AuxLabels2Mapper(const AuxLabels &labels_in, const Array2<int32_t *> &arc_map)
      : labels_in_(labels_in), arc_map_(arc_map) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] aux_size   The number of lists in the output AuxLabels
                                (equals num-arcs in the output FSA) and
                                the number of elements (equals num-aux-labels
                                on the arcs in the output FSA) will be written
                                to here.
  */
  void GetSizes(Array2Size<int32_t> *aux_size) const;

  /*
    Finish the operation and output auxiliary labels to `labels_out`.
       @param [out]  labels_out  Auxiliary labels on the arcs of the output FSA.
                                 Must be initialized; search for 'initialized
                                 definition' in class Array2 in array.h for
                                meaning.
   */
  void GetOutput(AuxLabels *labels_out);

 private:
  const AuxLabels &labels_in_;
  const Array2<int32_t *> &arc_map_;
};

/*
  Invert an FST, swapping the symbols in the FSA with the auxiliary labels.
  (e.g. swap input and output symbols in FST, but you decide which is which).
  Because each arc may have more than one auxiliary label, in general
  the output FSA may have more states than the input FSA.
 */
class FstInverter {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in     Input FSA
     @param [in] labels_in  Input aux-label sequences, one for each arc in
                            fsa_in
  */
  FstInverter(const Fsa &fsa_in, const AuxLabels &labels_in)
      : fsa_in_(fsa_in), labels_in_(labels_in) {}

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the FSA
                                will be written to here
        @param [out] aux_size   The number of lists in the output AuxLabels
                                (equals num-arcs in the output FSA) and
                                the number of elements (equals the number of
                                labels on `fsa_in`, although epsilons
                                will be removed) will be written to here.
  */
  void GetSizes(Array2Size<int32_t> *fsa_size,
                Array2Size<int32_t> *aux_size) const;

  /*
    Finish the operation and output inverted FSA to `fsa_out` and
    auxiliary labels to `labels_out`.
      @param [out]  fsa_out  The inverted FSA will be written to here.
                             Must be initialized; search for 'initialized
                             definition' in class Array2 in array.h for meaning.

                             Will have a number of states >= that in fsa_in.
                             If fsa_in was top-sorted it will be top-sorted.
                             Labels in the FSA will correspond to those in
                             `labels_in`.
      @param [out]  labels_out  The auxiliary labels will be written to here.
                                Must be initialized; search for 'initialized
                                definition' in class Array2 in array.h for
                                meaning.

                                Will be the same as the labels on `fsa_in`,
                                although epsilons (kEpsilon, zeros) will be
                                removed.
   */
  void GetOutput(Fsa *fsa_out, AuxLabels *labels_out);

 private:
  const Fsa &fsa_in_;
  const AuxLabels &labels_in_;
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_AUX_LABELS_H_
