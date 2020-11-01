// k2/csrc/fsa.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_HOST_FSA_H_
#define K2_CSRC_HOST_FSA_H_

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/util.h"

namespace k2host {

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
  int32_t label;  // 'label' as in a finite state acceptor.  For FSTs, the
                   // other label will be present in the aux_label array.  Which
                   // of the two represents the input vs. the output can be
                   // decided by the user; in general, the one that appears on
                   // the arc will be the one that participates in whatever
                   // operation you are doing.  For comparison purposes in
                   // sorting, we will treat this as uint32_t so that
                   // arc-sorting puts epsilon (0) before final-symbol (-1)

  float weight;  // log-prob of this arc, or the negative of a cost.
                 // Note: in some contexts, the scores/weights are
                 // taken to be zero (?)
  Arc() = default;

  Arc(int32_t src_state, int32_t dest_state, int32_t label, float weight)
      : src_state(src_state),
        dest_state(dest_state),
        label(label),
        weight(weight) {}

  bool operator==(const Arc &other) const {
    return std::tie(src_state, dest_state, label, weight) ==
           std::tie(src_state, other.dest_state, other.label, weight);
  }

  bool operator!=(const Arc &other) const { return !(*this == other); }

  bool operator<(const Arc &other) const {
    // compares `label` first, then `dest_state`.
    return std::tie(reinterpret_cast<const uint32_t&>(label),
                    dest_state) <
      std::tie(reinterpret_cast<const uint32_t&>(other.label),
               other.dest_state);
  }
};

std::ostream &operator<<(std::ostream &os, const Arc &arc);

// Caution: this does not include the weight in the hash.  You have to decide
// whether this makes sense in your use of this.
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
  last-numbered state. We represent an empty FSA(the one that accepts no
  strings) by having no states at all, so `size1` would be 0 (As an empty FSA is
  an initialized Array2 object, and `indexes` would be allocated and has at
  least one element, but we don't care about it here).
 */
struct Fsa : public Array2<Arc *, int32_t> {
  // `size1` is equal to num-states of the FSA.
  //
  // `size2` is equal to num-arcs of the FSA.
  //
  // `data` stores the arcs of the Fsa and is indexed by arc-index (an index
  // into the `data` array is called an arc-index). We may use `arcs` as an
  // alias of `data` in the context of FSA.
  //
  // `indexes` is indexed by state-index, is of length num-states + 1; it
  // contains the first arc-index leaving this state (index into `arcs`).
  // The next element of this array gives the end of that range.  Note: the
  // final-state is numbered last, and implicitly has no arcs leaving it.
  // We may use `arc-indexes` as an alias of `indexes`.

  // inherits constructors in Array2
  using Array2::Array2;

  int32_t NumStates() const {
    K2_CHECK_GE(size1, 0);
    return size1;
  }

  int32_t FinalState() const {
    // It's not valid to call FinalState if the FSA is empty.
    K2_CHECK_GE(size1, 2);
    return size1 - 1;
  }
};

std::ostream &operator<<(std::ostream &os, const Fsa &fsa);

using Cfsa = Fsa;
using CfsaVec = Array3<Arc *, int32_t>;

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
 public:
  int32_t Start();

  bool LookupArc(int32_t cur_state, int32_t label, int32_t *arc_index);

  float GetWeightForArc(int32_t arc_index);

  int32_t GetLabelForArc(int32_t arc_index);

  int32_t GetPrevStateForArc(int32_t arc_index);

  int32_t GetNextStateForArc(int32_t arc_index);

  // Specific subclasses of this may have additional functions, e.g.
  int32_t GetOlabelForArc(int32_t arc_index);
};

using FsaVec = std::vector<Fsa>;
using FstVec = std::vector<Fst>;

}  // namespace k2host

#endif  // K2_CSRC_HOST_FSA_H_
