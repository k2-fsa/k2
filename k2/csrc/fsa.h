// k2/csrc/fsa.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_H_
#define K2_CSRC_FSA_H_

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/util.h"

namespace k2 {

enum {
  kFinalSymbol = -1,  // final-costs are represented as arcs with
                      // kFinalSymbol as their label, to the final
                      // state (see Fst::final_state).
  kEpsilon = 0,       // Epsilon, which means "no symbol", is numbered zero,
                      // like in OpenFst.
};

struct Arc {
  int32_t src_state;
  int32_t dest_state;
  int32_t label;  // 'label' as in a finite state acceptor.
                // For FSTs, the other label will be present in the
                // aux_label array.  Which of the two represents the input
                // vs. the output can be decided by the user; in general,
                // the one that appears on the arc will be the one that
                // participates in whatever operation you are doing

  /* Note: the costs are not stored here but outside the Fst object, in some
     kind of array indexed by arc-index.  */

  bool operator==(const Arc &other) const {
    return std::tie(src_state, dest_state, label) ==
           std::tie(src_state, other.dest_state, other.label);
  }

  bool operator<(const Arc &other) const {
    // compares `label` first, then `dest_state`
    return std::tie(label, dest_state) <
           std::tie(other.label, other.dest_state);
  }
};

struct ArcHash {
  std::size_t operator()(const Arc &arc) const noexcept {
    std::size_t result = 0;
    hash_combine(&result, arc.src_state);
    hash_combine(&result, arc.dest_state);
    hash_combine(&result, arc.label);
    return result;
  }
};

/*
  struct Fsa is an unweighted finite-state acceptor (FSA) and is at the core of
  operations on weighted FSA's and finite state transducers (FSTs).  Note: being
  a final-state is represented by an arc with label == kFinalSymbol to
  final_state.

  The start-state is always numbered zero and the final-state is always the
  last-numbered state.  However, we represent the empty FSA (the one that
  accepts no strings) by having no states at all, so `arcs` would be empty.
 */
struct Fsa {
  // `arc_indexes` is indexed by state-index, is of length num-states,
  // contains the first arc-index leaving this state (index into `arcs`).
  // The next element of this array gives the end of that range.
  // Note: the final-state is numbered last, and implicitly has no
  // arcs leaving it. For non-empty FSA, we put a duplicate of the final state
  // at the end of `arc_indexes` to avoid boundary check for some FSA
  // operations. Caution: users should never call `arc_indexes.size()` to get
  // the number of states, they should call `NumStates()` to get the number.
  std::vector<int32_t> arc_indexes;

  // Note: an index into the `arcs` array is called an arc-index.
  std::vector<Arc> arcs;

  Fsa() = default;
  // just for creating testing FSA examples for now.
  Fsa(std::vector<Arc> fsa_arcs, int32_t final_state)
      : arcs(std::move(fsa_arcs)) {
    if (arcs.empty()) return;

    int32_t curr_state = -1;
    int32_t index = 0;
    for (const auto &arc : arcs) {
      CHECK_LE(arc.src_state, final_state);
      CHECK_LE(arc.dest_state, final_state);
      CHECK_LE(curr_state, arc.src_state);
      while (curr_state < arc.src_state) {
        arc_indexes.push_back(index);
        ++curr_state;
      }
      ++index;
    }
    // noted that here we push two `final_state` at the end, the last element is
    // just to avoid boundary check for some FSA operations.
    for (; curr_state <= final_state; ++curr_state)
      arc_indexes.push_back(index);
  }

  int32_t NumStates() const {
    return !arc_indexes.empty() ? (static_cast<int32_t>(arc_indexes.size()) - 1)
                                : 0;
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
  float *weights;  // Would typically be a log-prob or unnormalized log-prob
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
  DenseFsa(float *data, int32_t T, int32_t num_symbols, int32_t stride);
};

struct Fst {
  Fsa core;
  std::vector<int32_t> aux_label;
};

/*
  This demonstrates an interface for a deterministic FSA or FST; it's similar
  to Kaldi's DeterministicOnDemandFst class.  It can be used for things like
  language models.  Actually we'll template on types like this.  There is no
  need to actually inherit from this class.  */
class DeterministicGenericFsa {

  int32_t Start();


  bool LookupArc(int32_t cur_state,
                 int32_t label,
                 int32_t *arc_index);


  float GetWeightForArc(int32_t arc_index);

  int32_t Getint32_tForArc(int32_t arc_index);

  int32_t GetPrevStateForArc(int32_t arc_index);

  int32_t GetNextStateForArc(int32_t arc_index);

  // Specific subclasses of this may have additional functions, e.g.
  int32_t GetOlabelForArc(int32_t arc_index);

};


using FsaVec = std::vector<Fsa>;
using FstVec = std::vector<Fst>;
using DenseFsaVec = std::vector<DenseFsa>;

}  // namespace k2

#endif  // K2_CSRC_FSA_H_
