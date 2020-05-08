// k2/csrc/determinize.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
// dpove@gmail.com, Haowen Qiu qindazhu@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include <utility>
#include <vector>
#include <algorithm>

#include "k2/csrc/fsa_algo.h"

namespace k2 {

using std::shared_ptr;
using std::vector;
using std::priority_queue;
using std::pair


struct MaxTracebackState {
  // Element of a path from the start state to some state in an FSA
  // We can trace back the `parent` links, which will take
  // us backward along a path in the original FSA.
  std::shared_ptr<MaxTracebackState> prev;

  int32_t arc_index;  // Index of most recent arc in path from start-state to
                      // the dest-state, or -1 if the path is empty (only
                      // possible if this element belongs to the start-state).

  int32_t symbol;     // Symbol on the arc numbered `arc_index` of the input FSA
                      // (copied here for convenience), or 0 if arc_index == -1.

  MaxTracebackState(std::shared_ptr<MaxTracebackState> prev,
              int32_t arc_index, int32_t symbol):
      prev(prev), arc_index(arc_index), symbol(symbol) { }

};


class LogSumTracebackState;

// This struct is used inside LogSumTracebackState; it represents an
// arc that traces back to a previous LogSumTracebackState.
// A LogSumTracebackState represents a weighted colletion of paths
// terminating in a specific state.
struct LogSumTracebackLink {

  int32_t arc_index;  // Index of most recent arc in path from start-state to
                      // the dest-state, or -1 if the path is empty (only
                      // possible if this element belongs to the start-state).

  int32_t symbol;     // Symbol on the arc numbered `arc_index` of the input FSA
                      // (copied here for convenience), or 0 if arc_index == -1.

  double prob;        // The probability mass associated with this incoming
                      // arc in the LogSumTracebackState to which this belongs.

  std::shared_ptr<LogSumTracebackState> prev_state;
};

struct LogSumTracebackState {
  // LogSumTracebackState can be thought of as as a weighted set of paths from the
  // start state to a particular state.  (It will be limited to the subset of
  // paths that have a specific symbol sequence).

  // `prev_elements` is, conceptually, a list of pairs (incoming arc-index,
  // traceback link); we will keep it free of duplicates of the same incoming
  // arc.
  vector<LogSumTracebackLink> prev_elements;


  int32_t arc_index;  // Index of most recent arc in path from start-state to
                      // the dest-state, or -1 if the path is empty (only
                      // possible if this element belongs to the start-state).

  int32_t symbol;     // Symbol on the arc numbered `arc_index` of the input FSA
                      // (copied here for convenience), or 0 if arc_index == -1.

  MaxTracebackState(std::shared_ptr<MaxTracebackState> prev,
              int32_t arc_index, int32_t symbol):
      prev(prev), arc_index(arc_index), symbol(symbol) { }

};


struct DetStateElement {

  double weight;      // Weight from reference state to this state, along
                      // the path taken by following the 'prev' links
                      // (the path would have `seq_len` arcs in it).
                      // Note: by "this state" we mean the destination-state of
                      // the arc at `arc_index`.
                      // Interpret this with caution, because the
                      // base state, and the length of the sequence arcs from the
                      // base state to here, are known only in the DetState
                      // that owns this DetStateElement.

  std::shared_ptr<PathElement> path;
                      // The path from the start state to here (actually we will
                      // only follow back `seq_len` links.  Will be nullptr if
                      // seq_len == 0 and this belongs to the initial determinized
                      // state.

  DetStateElement &&Advance(float arc_weight, int32_t arc_index, int32_t arc_symbol) {
    return DetStateElement(weight + arc_weight,
                           std::make_shared<PathElement>(path, arc_index, arc_symbol));
  }

  DetStateElement(double weight, std::shared_ptr<PathElement> &&path):
      weight(weight), path(path) { }

};

class DetState;


struct DetStateCompare {
  // Comparator for priority queue.  Less-than operator that compares
  // forward_backward_weight for best-first processing.
  bool operator()(const shared_ptr<DetState> &a,
                  const shared_ptr<DetState> &b);
};



class Determinizer {
 public:
 private:

  using DetStatePriorityQueue = priority_queue<shared_ptr<DetState>,
                                               vector<shared_ptr<DetState> >,
                                               DetStateCompare>;


};


/*
  Conceptually a determinized state in weighted FSA determinization would
  normally
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


  Not really following the Google guidelines by not having _ at the end of class
  members, but this is more struct-like (members are public).

 */
class DetState {
 public:
  // `output_state` is the state in the output FSA that this determinized
  // state corresponds to.
  int32_t output_state;

  // `base_state` is the state in the input FSA from which the sequence of
  // `seq_len` symbols starts.  The weighted set of states that this DetState
  // represents is the set of states reachable by following that symbol sequence
  // from state `base_state`, with the best weights (per reachable state) along
  // those paths.  When Normalize() is called we may advance
  int32_t base_state;


  // seq_len is the length of symbol sequence that we follow from state
  // `base_state`.  The sequence of symbols can be found by tracing back one of
  // the DetStateElements in the doubly linked list (it doesn't matter which you
  // pick, the result will be the same.
  int32_t seq_len;

  bool normalized{false};


  std::list<DetStateElement> elements;

  // This is the weight on the best path that includes this determinized state.
  // It's needed to form a priority queue on DetStates, so we can process them
  // best-first.  It is computed as: the forward-weight on `base_state`,
  // plus the best/most-positive of: (the weight in a DetStateElement plus
  // the backward-weight of the state associated with that DetStateElement).
  double forward_backward_weight;


  /*
    Process arcs leaving this determinized state, possibly creating new determinized
    states in the process.
              @param [in] wfsa_in  The input FSA that we are determinizing, along
                                 with forward-backward weights.
                                 The input FSA should normally be epsilon-free as
                                 epsilons are treated as a normal symbol; and require
                                 wfsa_in.weight_tpe == kMaxWeight, for
                                 now (might later create a version of this code
                                 that works
              @param [in] prune_cutoff   Cutoff on forward-backward likelihood
                                 that we use for pruning; will equal
                                 wfsa_in.backward_state_weights[0] - prune_beam.
                                 Will be -infinity if we're not doing pruning.
              @param [in,out] state_map  Map from DetState to state-index in



  */
  void ProcessArcs(const WfsaWithFbWeights &wfsa_in,
                   Fsa *wfsa_out,
                   float prune_cutoff,
                   DetStateMap *state_map,
                   DetStatePriorityQueue *queue);


  /*
    Normalizes this DetState and sets forward_backward_weight.

    By 'normalize' what we mean is the following:

       - Remove duplicates.

         If the DLL of DetStateElements contains duplicate elements (i.e.
         elements whose paths end in the same state) it removes whichever has
    the
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
   */
  void Normalize(const Fsa &input_fsa,
                 const float *input_fsa_weights,
                 float *removed_weight,
                 std::vector<int32_t> *leftover_arcs) {
#ifndef NDEBUG
    CheckElementOrder();
#endif
    RemoveDuplicatesOfStates(input_fsa);
    RemoveCommonPrefix(input_fsa, input_fsa_weights, removed_weight, leftover_arcs);
  }
 private:

  /*
    Called from Normalize(), this function removes duplicates in
    `elements`: that is, if two elements represent paths that terminate at
    the same state in `input_fsa`, we choose the one with the better
    weight (or the first one in case of a tie).
   */
  void RemoveDuplicatesOfStates(const Fsa &input_fsa,
                                const float *input_fsa_weights);

  /*
    Called from Normalize(), this function removes any common prefix that the
    paths in `elements` possess.  If there is a common prefix it will reduce
    `seq_len`, subtract the weights associated with the removed arcs from the
    weights in `elements`, and set `input_arcs` to the sequence of arcs that
    were removed from
   */
  RemoveCommonPrefix(const Fsa &input_fsa,
                     const float *input_fsa_weights,
                     std::vector<int32_t> *input_arcs);
  /*
    This function just does some checking on the `elements` list that
    they are in the correct order, which is a lexicographical
    order (by state-id) on the paths of length `seq_len` starting from
    `base_state`.  The label sequences don't come into it because
    they are all the same.
   */
  void CheckElementOrder() const;

};

bool DetStateCompare::operator()(const shared_ptr<DetState> &a,
                                 const shared_ptr<DetState> &b) {
  return a->forward_backward_weight < b->forward_backward_weight;
}



void DetState::RemoveDuplicatesOfStates(const Fsa &input_fsa) {

  /*
    `state_to_elem` maps from int32_t state-id to the DetStateElement
    associated with it (there can be only one, we choose the one with
    the best weight).
   */
  std::unordered_map<int32_t, typename std::list<DetStateElement>::iterator> state_to_elem;



  for (auto iter = elements.begin(); iter != elements.end(); ++iter) {
    int32_t state =  input_fsa.arcs[elem.arc_index].nextstate;
    auto p = state_to_elem.insert({state, elem});
    bool inserted = p.second;
    if (!inserted) {
      DetStateElement *old_elem = p.first->second;
      if (old_elem->weight > elem->weight) {  // old weight is better
        this->RemoveElement(elem);
      } else {
        p.first->second = elem;
        this->RemoveElement(old_elem);
      }
    }
  }
}

void DetState::RemoveCommonPrefix(const Fsa &input_fsa,
                                  const float *input_fsa_weights,
                                  float *removed_weight_out,
                                  std::vector<int32_t> *input_arcs) {

  CHECK_GE(seq_len, 0);
  int32_t len;
  auto first_path = elements.front().path,
      last_path = elements.back().path;

  for (len = 1; len < seq_len; ++len) {
    first_path = first_path->prev;
    last_path = last_path->prev;
    if (first_path == last_path) {
      // Note: we are comparing pointers here.  We reached the same PathElement,
      // which means we reached the same state.
      break;
    }
  }
  input_arcs->clear();
  if (len < seq_len) {
    /* We reach a common state after traversing fewer than `seq_len` arcs,
       so we can remove a shared prefix. */
    double removed_weight = 0.0;
    int32_t new_seq_len = len,
        removed_seq_len = seq_len - len;
    input_arcs->resize(removed_seq_len);
    // Advance base_state
    int32_t new_base_state = input_fsa.arcs[first_path->arc_index].src_state;
    for (; len < seq_len; ++len) {
      auto arc = input_fsa.arcs[first_path->arc_index];
      input_arcs[seq_len - 1 - len] = first_path->arc_index;
      removed_weight += input_fsa_weights[first_path->arc_index];
      first_path = first_path->prev;
    }
    // Check that we got to base_state.
    CHECK((self->base_state == 0 && first_path == nullptr) ||
          fsa.arcs[first_path->arc_index].dest_state == this->base_state);
    this->base_state = new_base_state;
    if (removed_weight != 0) {
      for (DetStateElement &det_state_elem: elements) {
        det_state_elem.weight -= removed_weight;
      }
    }
    *removed_weight_out = removed_weight;
  } else {
    *removed_weight_out = 0;
    input_arcs->clear();
  }
}

void DetState::CheckElementOrder(const Fsa &input_fsa) const {
  // Checks that the DetStateElements are in a lexicographical order on the
  // lists of states in their paths.  This will be true becase of how we
  // construct them (it requires on the IsArcSorted() property, whereby arcs
  // leaving each state in the FSA are sorted first on label and then on
  // dest_state.
  if (seq_len == 0) {
    CHECK(elements.size() == 1);
    CHECK(elements.front().weight == 0.0);
  }

  std::vector<int32> prev_seq;
  for (auto iter = elements.begin(); iter != elements.end(); ++iter) {
    auto path = iter->path;
    std::vector<int32> cur_seq;
    for (int32_t i = 0; i < seq_len; i++) {
      cur_seq.push_back(input_fsa.arcs[path->arc_index].prev_state);
      path = path->prev;
    }
    std::reverse(cur_seq.begin(), cur_seq.end());
    if (iter != elements.begin()) {
      CHECK(cur_seq > prev_seq);
    }
    prev_seq.swap(cur_seq);
  }
}


/*
  This class maps from determinized states (DetState) to integer state-id
  in the determinized output.
 */
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
    auto p = map_.insert({compact, cur_output_state});
    bool inserted = p.second;
    if (inserted) {
      *state_id = cur_output_state_++;
      return true;
    } else {
      *state_id = p.first->second;
      return false;
    }
  }

  int32_t size() const { return cur_output_state_; }

 private:
  int32_t cur_output_state_{0};
  std::unordered_map<std::pair<uint64_t, uint64_t>, int32_t,
                     DetStateVectorHasher> map_;

  /* Turns DetState into a compact form of 128 bits.  Technically there
     could be collisions, which would be fatal for the algorithm, but this
     is one of those lifetime-of-the-universe type of things (kind of like
     the theoretical potential for git hash collision) that we ignore.

     The normalized form

  */
  void DetStateToCompact(const DetState &d,
                         std::pair<uint64_t, uint64_t> *vec) {
    assert(d.normalized);

    uint64_t a = d.base_state + 17489 * d.seq_len,
             b = d.base_state * 103979 + d.seq_len;

    // We choose an arbitrary DetStateElement (the first one in the list) to
    // read the symbol sequence from; the symbol sequence will be the same no
    // matter which element we choose to trace back.
    DetStateElement *elem = d.head;
    int32_t seq_len = d.seq_len;
    for (int32_t i = 0; i < seq_len; ++i) {
      a = elem->symbol + 102299 * a;
      b = elem->symbol + 102983 * b;
      elem = elem->parent
    }
    vec->first = a;
    vec->second = b;
  }

  struct DetStateHasher {
    size_t operator()(const std::pair<uint64_t, uint64_t> &p) const {
      return p.first;
    }
  };
};

void DeterminizeMax(const WfsaWithFbWeights &a, float beam, Fsa *b,
                    std::vector<std::vector<int32_t> > *arc_map) {
  // TODO(dpovey): use glog stuff.
  assert(IsValid(a) && IsEpsilonFree(a) && IsTopSortedAndAcyclic(a));
  if (a.arc_indexes.empty()) {
    b->Clear();
    return;
  }
  float cutoff = a.backward_state_weights[0] - beam;
  // TODO(dpovey)
}
}  // namespace k2
