// k2/csrc/fsa.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_H_
#define K2_CSRC_FSA_H_

#include <cstdint>
#include <vector>

namespace k2 {

using Label = int32_t;
using StateId = int32_t;
using Weight = float;

enum {
  kFinalSymbol = -1,  // final-costs are represented as arcs with
                      // kFinalSymbol as their label, to the final
                      // state (see Fst::final_state).
  kEpsilon = 0        // Epsilon, which means "no symbol", is numbered zero,
                      // like in OpenFst.
};

/* Range is what we use when we want (begin,end) indexes into some array.
   `end` is the last element plus one.
   TODO(Dan): may store just begin and have the next element be the end.
*/
struct Range {
  int32_t begin;
  int32_t end;
};

struct Arc {
  StateId src_state;
  StateId dest_state;
  Label label;  // 'label' as in a finite state acceptor.
                // For FSTs, the other label will be present in the
                // aux_label array.  Which of the two represents the input
                // vs. the output can be decided by the user; in general,
                // the one that appears on the arc will be the one that
                // participates in whatever operation you are doing

  /* Note: the costs are not stored here but outside the Fst object, in some
     kind of array indexed by arc-index.  */
};

struct ArcLabelCompare {
  bool operator()(const Arc& a, const Arc& b) const {
    return a.label < b.label;
  }
};

/*
  struct Fsa is an unweighted finite-state acceptor (FSA) and is at the core of
  operations on weighted FSA's and finite state transducers (FSTs).  Note: being
  a final-state is represented by an arc with label == kEpsilon to final_state.

  The start-state is always numbered zero and the final-state is always the
  last-numbered state.  However, we represent the empty FSA (the one that
  accepts no strings) by having no states at all, so `arcs` would be empty.
 */
struct Fsa {
  // `leaving_arcs` is indexed by state-index, is of length num-states,
  // contains the first arc-index leaving this state (index into `arcs`).
  // The next element of this array gives the end of that range.
  // Note: the final-state is numbered last, and implicitly has no
  // arcs leaving it.
  std::vector<Range> leaving_arcs;

  // Note: an index into the `arcs` array is called an arc-index.
  std::vector<Arc> arcs;

  StateId NumStates() const {
    return static_cast<StateId>(leaving_arcs.size());
  }
};

/*
  DenseFsa represents an FSA stored as a matrix, representing something
  like CTC output from a neural net.  We view `weights` as a T by N
  matrix, where N is the number of symbols (including blank/zero).

  Physically, we would access weights[t,n] as weights[t * t_stride + n].

  This FSA has T + 2 states, with state 0 the start state and state T + 2
  the final state.  (Caution: if we formulated our FSAs more normally we
  would have T + 1 states, but because we represent final-probs via an
  arc with symbol kFinalSymbol on it to the last state, we need one
  more state).   For 0 <= t < T, we have an arc with symbol n on it for
  each 0 <= n < N, from state t to state t+1, with weight equal to
  weights[t,n].


 */
struct DenseFsa {
  Weight* weights;  // Would typically be a log-prob or unnormalized log-prob
  int32_t T;        // The number of time steps == rows in the matrix `weights`;
                    // this FSA has T + 2 states, see explanation above.
  int32_t num_symbols;  // The number of symbols == columns in the matrix
                        // `weights`.
  int32_t t_stride;     // The stride of the matrix `weights`

  /* Constructor
     @param [in] data   Pointer to the raw data, which is a T by num_symbols
     matrix with stride `stride`, containing logprobs

      CAUTION: we may later enforce that stride == num_symbols, in order to
      be able to know the layout of a phantom matrix of arcs.  (?)
   */
  DenseFsa(Weight* data, int32_t T, int32_t num_symbols, int32_t stride);
};

/*
  this general-purpose structure conceptually the same as
  std::vector<std::vector>; elements of `ranges` are (begin, end) indexes into
  `values`.
 */
struct VecOfVec {
  std::vector<Range> ranges;
  std::vector<int32_t> values;
};

struct Fst {
  Fsa core;
  std::vector<int32_t> aux_label;
};

using FsaVec = std::vector<Fsa>;
using FstVec = std::vector<Fst>;
using DenseFsaVec = std::vector<DenseFsa>;

}  // namespace k2

#endif  // K2_CSRC_FSA_H_
