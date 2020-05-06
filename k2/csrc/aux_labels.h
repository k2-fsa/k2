// k2/csrc/aux_labels.h

// Copyright (c)  2020  Xiaomi Corporation (author: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_AUX_LABELS_H_
#define K2_CSRC_AUX_LABELS_H_

#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {

/*
  This header contains utilities for dealing with auxiliary labels on FSAs.

  These auxiliary labels can be used where you really have a transducer (i.e. to
  store the ilabels or olabels, whichever of the two is not participating
  directly in the operation you are doing).

  We deal with two formats of labels: a vector of int32_t, one per arc,
  for cases where we know we have at most one label per arc; and
  struct Auxint32_ts for cases where there may in general be a sequence
  of labels (one per arc).

*/


/*
  This allows you to store auxiliary labels (e.g. olabels or ilabels)
  on each arc of an Fsa.
 */
struct Auxint32_ts {
  /* Suppose this is associated with an Fsa f.  start_pos will be of
     size f.arcs.size() + 1; start_pos[i] is the start position in
     `labels` of the label sequence on arc i.  start_pos.end()
     equals labels.size(). */
  std::vector<int32_t> start_pos;
  /* For arc i, (labels[start_pos[i] ], labels[start_pos[i]+1], ... labels[start_pos[i+1]-1])
     are the list of labels on that arc.  None of the elements of `labels` are
     expected to be zero (epsilon). */
  std::vector<int32_t> labels;
};


/*
  Maps auxiliary labels after an FSA operation where each arc in the output
  FSA corresponds to exactly one arc in the input FSA.
     @param [in] labels_in   int32_ts on the arcs of the input FSA
     @param [in] arc_map    Vector of size (output_fsa.arcs.size()),
                            saying which arc of the input FSA it
                            corresponds to.
     @param [in] labels_out  int32_ts on the arcs of the output FSA
 */
void MapAuxint32_ts1(const Auxint32_ts &labels_in,
                   const std::vector<int32_t> &arc_map,
                   Auxint32_ts *labels_out);

/*
  Maps auxiliary labels after an FSA operation where each arc in the output
  FSA can correspond to a sequence of arcs in the input FSA.
     @param [in] labels_in   int32_ts on the arcs of the input FSA
     @param [in] arc_map    Vector of size (output_fsa.arcs.size()),
                            giving the sequence of arc-indexes in the input
                            FSA that it corresponds to.
     @param [in] labels_out  int32_ts on the arcs of the output FSA
 */
void MapAuxint32_ts2(const Auxint32_ts &labels_in,
                   const std::vector<std::vector<int32_t> > &arc_map,
                   Auxint32_ts *labels_out);


/*
  Invert an FST, swapping the symbols in the FSA with the auxiliary labels.
  (e.g. swap input and output symbols in FST, but you decide which is which).
  Because each arc may have more than one auxiliary label, in general
  the output FSA may have more states than the input FSA.

     @param [in] fsa_in  Input FSA
     @param [in] labels_in  Input aux-label sequences, one for each arc in
                         fsa_in
     @param [out] fsa_out   Output FSA.  Will have a number of states
                        >= that in fsa_in.  If fsa_in was top-sorted it
                        will be top-sorted.  int32_ts in the FSA will
                        correspond to those in `labels_in`.
     @param [out] aux_labels_out  Auxiliary labels on the arcs of
                        fsa_out.  Will be the same as the labels on
                        `fsa_in`, although epsilons (kEpsilon, zeros) will be
                        removed.
 */
void InvertFst(const Fsa &fsa_in,
               const Auxint32_ts &labels_in,
               Fsa *fsa_out,
               Auxint32_ts *aux_labels_out);



}  // namespace k2

#endif  // K2_CSRC_AUX_LABELS_H_
