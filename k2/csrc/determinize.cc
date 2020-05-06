// k2/csrc/fsa_algo.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey dpove@gmail.com, Haowen Qiu qindazhu@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <utility>
#include <vector>

namespace k2 {


struct DetStateElement {
  // Element of the doubly linked list whose start/end are
  // members 'head' and 'tail' of DetState.
  // We can trace back the `parent` links, which will take
  // us backward along a path in the original FSA.
  DetStateElement *parent = nullptr;
  int32_t arc_index;  // Index of most recent arc in path to the dest-state.
                      // This data-structure represents a path through the FSA,
                      // with this arc being the most recent arc on that path.
  int32_t symbol;     // Symbol on the arc numbered `arc_index` of the input FSA
                      // (copied here for convenience).

  double weight;      // Weight from reference state to this state, along
                      // the path taken by following the 'parent' links
                      // (the path would have `seq_len` arcs in it).
                      // Note: by "this state" we mean the destination-state of
                      // the arc at `arc_index`.

  // `prev` and `next` form the doubly linked list of DetStateElement
  DetStateElement *prev = nullptr;
  DetStateElement *next = nullptr;

  // This comparator function compares the weights, but is careful in case of
  // ties to ensure deterministic behavior.
  bool operator < (const DetStateElement &other) const {
    if (weight < other.weight) return true;
    else if (weight > other.weight) return false;


  }

};





/*
  Conceptually a determinized state in weighted FSA determinization would normally
  be a weighted subset of states in the input FSA, with the weights normalized
  somehow (e.g. subtracting the sum of the weights).

  Two determinized states are equal if the states and weights are the same.  To
  ensure differentiability, our assumption is that in general no two arcs in the
  input FSA have identical weights.  We argue that two determinized states can
  always be represented as a base-state and a symbol sequence.  Imagine that we
  follow arcs with that symbol sequence from the base-state, and then in case we
  reach the same states in the different ways we always select the best path
  from the base-state.  That process gives us a set of states and weights.  We
  argue that this representation is unique.  (If not, it won't matter actually;
  it will just give us an output that's less minimal than it could be).


 */
struct DetState {
  // `base_state` is a state in the input FSA.
  int32_t base_state;
  // seq_len is the length of symbol sequence that we follow from state `base_state`.
  // The sequence of symbols can be found by tracing back one of the DetStateElements
  // in the doubly linked list (it doesn't matter which you pick, the result will be the
  // same.
  int32_t seq_len;

  bool normalized { false };

  DetState *parent; // Maybe not needed!

  DetStateElement *head;
  DetStateElement *tail;

  double forward_backward_weight;

  /*
    Normalizes this DetState and sets forward_backward_weight.

    By 'normalize' what we mean is the following:

       - Remove duplicates.

         If the DLL of DetStateElements contains duplicate elements (i.e.
         elements whose paths end in the same state) it removes whichever has the
         smallest weight.  (Remember, a determinized state is, conceptually, a
         weighted subset of elements; we are implementing determinization in a
         tropical-like semiring where we take the best weight.

         In case of ties on the weights, we carefully re-examine the paths to
         make sure that the tie was not due to numerical roundoffi; and if it
         was still a tie, we disambiguate using a lexical order on state
         sequences.  The reason it's important to have deterministic behavior in
         case of ties on weights, is that a failure here could lead to
         situations where we didn't advance the base state where we could,
         leading the number of determinized states to be larger than it could
         be.

       - Advance the base state if possible.  Each DetState can be represented
         as a base state and a sequence of symbols from that base state, but
         if some initial subsequence of that symbol sequence takes us to
         a unique state then we say the DetState is not normalized.  In that
         case we need to advance the base state and reduced `seq_len`.
         If this happens, then the arc sequence which takes us to the new
         base state will be output to `leftover_arcs`.  When this is done,
         the 'weight' components of the DetStateElement members also need
         to be adjusted to remove the weight contribution from those arcs.

     The forward_backward_weight is the weight on the best path through the
     output determinized FSA that will include this DetState.  It will determine
     the order of expansion of DetStates and also whether the states are
     expanded at all (if the pruning beam `beam` is finite).
     forward_backward_weight is the sum of the forward weight of the base state,
     plus (the greatest over the DetStateElements, of its `weight` element,
     plus the backward weight in the input FSA of the state that corresponds
     to it).


     worked outobtained from

   */
  void Normalize(std::vector<int32_t> *leftover_arcs);
};


void DetState::Normalize(std::vector<int32_t> *input_arcs) {

}


class DetStateMap {
 public:
  /*
    Outputs the output state-id corresponding to a specific DetState structure.
    This does not store any pointers to the DetState or its contents, so
    you can delete the DetState without affecting this object's ability to map
    an equivalent DetState to the same state-id.

       @param [in] a  The DetState that we're looking up
       @param [out] state_id  The state-index in the output FSA
                      corresponding to this DetState (will
                      be freshly allocated if an equivalent of
                      this DetState did not already exist.
        @return  Returns true if this was a NEWLY CREATED state,
              false otherwise.
   */
  bool GetOutputState(const DetState &a, int32_t *state_id) {
    std::pair<uint64_t, uint64_t> compact;
    DetStateToCompact(a, &compact);
    auto p = map_.insert({compact, cur_output_state));
    bool inserted = p.second;
    if (inserted) {
      *state_id = cur_output_state_++;
      return true;
    } else {
      *state_id = p.first->second;
      return false;
    }
  }

  size_t size() const { return cur_output_state_; }

 private:

  int32_t cur_output_state_ { 0 };
  std::unordered_map<std::pair<uint64_t, uint64_t>, int32_t, DetStateVectorHasher> map_;

  /* Turns DetState into a compact form of 128 bits.  Technically there
     could be collisions, which would be fatal for the algorithm, but this
     is one of those lifetime-of-the-universe type of things (kind of like
     the theoretical potential for git hash collision) that we ignore. */
  void DetStateToCompact(const DetState &d,
                         std::pair<uint64_t, uint64_t> *vec) {
    assert(d.normalized);

    uint64_t a = d.base_state + 17489 * d.seq_len,
        b = d.base_state * 103979  + d.seq_len;

    // We choose an arbitrary DetStateElement (the first one in the list) to
    // read the symbol sequence from; the symbol sequence will be the same no
    // matter which element we choose to trace back.
    DetStateElement *elem = d.head;
    int32_t seq_len = d.seq_len;
    for (int32_t i = 0; i < seq_len; i++) {
      a = elem->symbol + 102299 * a;
      b = elem->symbol + 102983 * b;
    }
    vec->first = a;
    vec->second = b;
  }

  struct DetStateHasher {
    size_t operator () (const std::pair<uint64_t, uint64_t> &p) const {
      return p.first;
    }
  };



};



void DeterminizeMax(const WfsaWithFbWeights &a,
                    float beam,
                    Fsa *b,
                    std::vector<std::vector<int32_t> > *arc_map) {
  // TODO: use glog stuff.
  assert(IsValid(a) && IsEpsilonFree(a) && IsTopSortedAndAcyclic(a));
  if (a.arc_indexes.empty()) {
    b->Clear();
    return;
  }
  float cutoff = a.backward_state_weights[0] - beam;


}


}  // namespace k2
