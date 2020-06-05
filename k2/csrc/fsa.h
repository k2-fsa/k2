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
#include "k2/csrc/array.h"
#include "k2/csrc/util.h"

namespace k2 {

enum {
  kFinalSymbol = -1,  // final-costs are represented as arcs with
                      // kFinalSymbol as their label, to the final
                      // state (see Fst::final_state).
  kEpsilon = 0,       // Epsilon, which means "no symbol", is numbered zero,
                      // like in OpenFst.
};

// CAUTION: the sizeof() this is probably 128, not 96.  This could be a
// waste of space.  We may later either use the extra field for something, or
// find a way to reduce the size.
struct Arc {
  int32_t src_state;
  int32_t dest_state;
  int32_t label;  // 'label' as in a finite state acceptor.
                  // For FSTs, the other label will be present in the
                  // aux_label array.  Which of the two represents the input
                  // vs. the output can be decided by the user; in general,
                  // the one that appears on the arc will be the one that
                  // participates in whatever operation you are doing
  Arc() = default;
  Arc(int32_t src_state, int32_t dest_state, int32_t label)
      : src_state(src_state), dest_state(dest_state), label(label) {}

  /* Note: the costs are not stored here but outside the Fst object, in some
     kind of array indexed by arc-index.  */

  bool operator==(const Arc &other) const {
    return std::tie(src_state, dest_state, label) ==
           std::tie(src_state, other.dest_state, other.label);
  }

  bool operator!=(const Arc &other) const { return !(*this == other); }

  bool operator<(const Arc &other) const {
    // compares `label` first, then `dest_state`
    return std::tie(label, dest_state) <
           std::tie(other.label, other.dest_state);
  }
};

std::ostream &operator<<(std::ostream &os, const Arc &arc);

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
  // `arc_indexes` is indexed by state-index, is of length num-states + 1; it
  // contains the first arc-index leaving this state (index into `arcs`).  The
  // next element of this array gives the end of that range.  Note: the
  // final-state is numbered last, and implicitly has no arcs leaving it. For
  // non-empty FSA, we put a duplicate of the final state at the end of
  // `arc_indexes` to avoid boundary check for some FSA operations. Caution:
  // users should never call `arc_indexes.size()` to get the number of states,
  // they should call `NumStates()` to get the number.
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
  int32_t FinalState() const {
    // It's not valid to call this if the FSA is empty.
    CHECK(!arc_indexes.empty());
    return static_cast<int32_t>(arc_indexes.size()) - 2;
  }
};

// TODO(haowen): replace Cfsa and CfsaVec with below definitions
using Cfsa_ = Array2<int32_t, Arc>;
using CfsaVec_ = Array3<int32_t, Arc>;

/*
  Cfsa is a 'const' FSA, which we'll use as the input to operations.  It is
  designed in such a way that the storage underlying it may either be an Fsa
  (i.e. with std::vectors) or may be some kind of tensor (probably CfsaVec).
  Note: the pointers it holds aren't const for now, because there may be
  situations where it makes sense to change them (even though the number of
  states and arcs can't be changed).
 */
struct Cfsa {
  int32_t num_states;  // number of states including final state.  States are
                       // numbered `0 ... num_states - 1`.  Start state is 0,
                       // final state is state `num_states - 1`.  We store a
                       // redundant representation here out of a belief that it
                       // might reduce the number of instructions in code.
  int32_t begin_arc;   // a copy of arc_indexes[0]; gives the first index in
                       // `arcs` for the arcs in this FSA.  Will be >= 0.
  int32_t end_arc;     // a copy of arc_indexes[num_states]; gives the
                       // one-past-the-last index in `arcs` for the arcs in this
                       // FSA.  Will be >= begin_arc.

  const int32_t *arc_indexes;  // an array, indexed by state index, giving the
                               // first arc index of each state.  The last one
                               // is repeated, so for any valid state 0 <= s <
                               // num_states we can use arc_indexes[s+1].  That
                               // is: elements 0 through num_states (inclusive)
                               // are valid.  CAUTION: arc_indexes[0] may be
                               // greater than zero.

  Arc *arcs;  // Note: arcs[begin_arc] through arcs[end_arc - 1]
              // are valid.

  Cfsa();
  // Constructor from Fsa. The passed `fsa` should be kept alive
  // as long as this cfsa is alive.
  explicit Cfsa(const Fsa &fsa);

  Cfsa &operator=(const Cfsa &cfsa) = default;
  Cfsa(const Cfsa &cfsa) = default;

  int32_t NumStates() const { return num_states; }
  int32_t NumArcs() const { return end_arc - begin_arc; }
  int32_t FinalState() const {
    CHECK_GE(num_states, 2) << "It's an error to invoke this method for "
                            << "an empty cfsa";
    return num_states - 1;
  }

  // for test only
  bool operator==(const Cfsa &other) const {
    if (other.num_states != num_states) return false;

    if (other.NumArcs() != NumArcs()) return false;

    for (int32_t i = 0; i != NumArcs(); ++i) {
      const auto &this_arc = arcs[begin_arc + i];
      const auto &other_arc = other.arcs[other.begin_arc + i];

      if (this_arc != other_arc) return false;
    }

    return true;
  }
};

std::ostream &operator<<(std::ostream &os, const Cfsa &cfsa);

constexpr int32_t kCfsaVecVersion = 0x01;

struct CfsaVecHeader {
  int32_t version;
  int32_t num_fsas;
  int32_t state_offsets_start;
  int32_t arc_indexes_start;
  int32_t arcs_start;
};

class CfsaVec {
 public:
  /*
      Constructor from linear data, e.g. from data stored in a torch.Tensor.
      This would previously have been created using CreateCfsaVec().

         @param [in] size    size in int32_t elements of `data`, only
                             needed for checking purposes.
         @param [in] data    The underlying data.   Format of data is
                             described below (all elements are of type
                             int32_t unless stated otherwise).  Would have
                             been created by CreateCfsaVec().

             - version       Format version number, currently always 1.
             - num_fsas      The number of FSAs
             - state_offsets_start  The offset from the start of `data` of
                             where the `state_offsets` array is, in int32_t
                             (4-byte) elements.
             - arc_indexes_start   The offset from the start of `data` of
                             where the `arc_indexes` array is, in int32_t
                             (4-byte) elements.
             - arcs_start    The offset from the start of `data` of where
                             the first Arc is, in sizeof(Arc) multiples, i.e.
                             Arc *arcs = ((Arc*)data) + arcs_start

            [possibly some padding here]
             - state_offsets[num_fsas + 1]   state_offsets[f] is the sum of
                             the num-states of all the FSAs preceding f.  It is
                             also the offset from the beginning of the
                             `arc_indexes` array of where the part corresponding
                             to FSA f starts.  The number of states in FSA f
                             is given by
                             `state_offsets[f+1] - state_offsets[f] - 1`.
                             Caution: one is subtracted above because the last
                             entry in the arc_indexes array is repeated.
                             This is >= 0; it will be zero if the
                             FSA f is empty, and >= 2 otherwise.
            [possibly some padding here]

             - arc_indexes[tot_states + num_fsas]   This gives the indexes
                             into the `arcs` array of where we can find the
                             first of each state's arcs. `num_fsas` is needed
                             since the final state of every fsa is repeated in
                             `arc_indexes`.

             [pad as needed for memory-alignment purposes then...]

             - arcs[tot_arcs]
  */
  CfsaVec(std::size_t size, void *data);

  int32_t NumFsas() const { return num_fsas_; }

  Cfsa operator[](int32_t i) const;

  CfsaVec &operator=(const CfsaVec &) = delete;
  CfsaVec(const CfsaVec &) = delete;

  ~CfsaVec() {
    if (opaque_deleter_) (*opaque_deleter_)(opaque_ptr_);
  }

  void SetDeleter(void (*deleter)(void *), void *p) {
    opaque_deleter_ = deleter;
    opaque_ptr_ = p;
  }

 private:
  int32_t num_fsas_;

  // The raw underlying data;
  // CAUTION: we do NOT own the memory here.
  int32_t *data_;
  // The size of the underlying data;
  // Caution: it is the number of `int32_t` in data_, NOT the number of bytes.
  std::size_t size_;

  // the following two fields are for DLPack, which enables us to
  // share memory with `torch::Tensor`.
  //
  // C++ code will in generate not touch them.
  void (*opaque_deleter_)(void *) = nullptr;
  void *opaque_ptr_ = nullptr;
};

/*
  Return the number of bytes we'd need to represent this vector of Cfsas
  linearly as a CfsaVec. */
std::size_t GetCfsaVecSize(const std::vector<Cfsa> &cfsas);

// Return the number of bytes we'd need to represent this Cfsa
// linearly as a CfsaVec with one element
std::size_t GetCfsaVecSize(const Cfsa &cfsa);

/*
  Create a CfsaVec from a vector of Cfsas (this involves representing
  the vector of Fsas in one big linear memory region).

     @param [in] cfsas   The vector of Cfsas to be linearized;
                      must be nonempty
     @param [in] data    The allocated data of size `size` bytes
     @param [in] size    The size of the memory block in bytes passed;
                         must equal the return value of
                         GetCfsaVecSize(cfsas).
 */
void CreateCfsaVec(const std::vector<Cfsa> &cfsas, void *data,
                   std::size_t size);

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

}  // namespace k2

#endif  // K2_CSRC_FSA_H_
