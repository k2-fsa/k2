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
using std::weak_ptr;
using std::vector;
using std::priority_queue;
using std::pair;



// setting this to true will reduce memory consumption, but could increase
// compute time in cases where we finish before the beam.
constexpr bool PROCESS_ARCS_IMMEDIATELY = false;



struct MaxTracebackState {
  using DerivOutputType = int32_t;


  int32_t state_id;  // state-id in the input FSA

  int32_t arc_id;    // arc-id in input FSA of the arc that enters
                     // state `state_id` (or -1 if this is the start state).

  // prev_state is the state we trace back to (the previous state),
  // which is the src_state of the arc numbered arc_id.
  // It will be nullptr if state_id == 0 (start state).
  shared_ptr<MaxTracebackState> prev_state;

  double forward_prob;    // The total forward log-probability from the start
                          // state to this state (along whichever specific
                          // sequence of symbols we took to get here; it's not
                          // necessarily the best forward in the lattice).

  // This constructor is for the start-state of the input FSA.
  MaxTracebackState(): state_id(0), arc_id(-1), arc_symbol(-1),
                       prev_state(nullptr), forward_prob(0.0) { }

  /**
     @param [in] state_id  State in input FSA that this corresponds to
     @param [in] src   Previous LogSumTracebackState that we'll point back
                      to, or NULL
     @param [in] incoming_arc_index.  Its src_state will equal src->state_id,
                      its dest_state will equal state_id.
     @param [in] src_symbol   Symbol on the input arc
     @param [in] arc_weight   Weight on the input arc
   */
  MaxTracebackState(int32_t state_id,
                    const std::shared_ptr<MaxTracebackState> &src,
                    int32_t incoming_arc_index,
                    int32_t arc_weight):
      state_id(state_id),
      arc_id(incoming_arc_index),
      prev_state(src),
      forward_prob(element->forward_prob + arc_weight) { }

  void Accept(const std::shared_ptr<MaxTracebackState> &src,
              int32_t arc_index, int32_t _symbol, float arc_weight) {
    double new_forward_prob = src->forward_prob + arc_weight;
    if (new_forward_prob > forward_prob) {
      forward_prob = new_forward_prob;
      arc_id = arc_index;
      prev_state = src;
      // state_id doesn't change, nor does _symbol.
    }
  }
};


class LogSumTracebackState;

// This struct is used inside LogSumTracebackState; it represents an
// arc that traces back to a previous LogSumTracebackState.
// A LogSumTracebackState represents a weighted colletion of paths
// terminating in a specific state.
struct LogSumTracebackLink {

  shared_ptr<LogSumTracebackState> prev_state;
                      // `prev_state` is the state that this points back to.


  int32_t arc_index;  // Index (in input FSA) of this arc from prev_state to the
                      // destination state (in whose LogSumTracebackState this
                      // LogSumTracebackLink will be located).

  double forward_prob;  // The total forward log-probability from the start
                        // state to the end of this arc just before it joins the
                        // node (conceptually).  Note: this is only the total
                        // forward log-prob limited to whatever symbol-sequence
                        // got us to this point...  there may be other
                        // symbol-sequences terminating in this state.
                        // (That symbol-sequence can be obtained by tracing back
                        // this data structure till you hit a LogSumTracebackState
                        // with prev_state == nullptr (and state_id == 0).


  LogSumTracebackLink(const std::shared_ptr<LogSumTracebackState> &src,
                      int32_t arc_index, float arc_weight):
      src(src), arc_index(arc_index),
      forward_prob(arc_weight + src->forward_prob) { }


};


struct LogSumTracebackState {
  using DerivOutputType = pair<int32_t, float>;

  // LogSumTracebackState can be thought of as as a weighted set of paths from
  // the start state to a particular state.  (It will be limited to the subset
  // of paths that have a specific symbol sequence).

  // `prev_elements` is, conceptually, a list of incoming arcs with associated
  // weights.
  vector<LogSumTracebackLink> prev_elements;

  int32_t state_id;       // The state-id in the input FSA that this
                          // LogSumTracebackState corresponds to.  (Unique to
                          // this determinized state; the same state-id may
                          // appear in multiple determinized states, in general.

  double forward_prob;    // The total forward log-probability from the start
                          // state to this state (along whichever specific
                          // sequence of symbols we took to get here; it's not
                          // necessarily the best forward-prob in the lattice
                          // that would take us to this state).  Will equal the
                          // log-sum of the forward_probs of the prev_elements.

  double backward_prob;   // Used temporarily in algorithms as a backward prob.
                          // Undefined most of the time.

  // This constructor is to be used only for the start-state (of both the
  // input FSA and the determinized FSA).
  LogSumTracebackState(): state_id(0), forward_prob(0.0) { }

  /**
     @param [in] state_id  State in input FSA that this corresponds to
     @param [in] src   Previous LogSumTracebackState that we'll point back
                      to, or nullptr if this belongs to the initial
                      determinized-state.
     @param [in] incoming_arc_index.  Arc-index in input FSA.
                      Its src_state will equal src->state_id, its dest_state
                      will equal state_id.
     @param [in] arc_weight   Weight on the arc
   */
  LogSumTracebackState(int32_t state_id,
                       const std::shared_ptr<LogSumTracebackState> &src,
                       int32_t incoming_arc_index,
                       int32_t arc_weight):
      state_id(state_id),
      forward_prob(element->forward_prob + arc_weight) {
    prev_elements.emplace_back(src, incoming_arc_index, forward_prob);
  }

  /*
     Accept a new incoming link.  The args are the same as for
     the constructor just above; see documentation there.
   */
  void Accept(const std::shared_ptr<LogSumTracebackState> &src,
              int32_t arc_index, float arc_weight) {
    double link_forward_prob = src.forward_prob + arc_weight;
    prev_elements.emplace_back(src, arc_index, link_forward_prob);
    this->forward_prob = LogAdd(this->forward_prob, link_forward_prob);
  }
};


/*
  Find the most recent common ancestor LogSumTracebackState of a set of
  LogSumTracebackStates, and return the number of links we had to follow to get
  there (i.e. the length of symbol sequence we had to remove from
  each path).

    @param [in,out] cur_states   A set of TracebackStates that we'll
                  trace back from.  Must be nonempty.  Equality
                  here is simply pointer identity.

                  At exit it will contain a single member which will
                  be the most recent common ancestor.
    @return  Returns the number of links we had to follow (>=0).  If
             `cur_states.size() == 1` this will be zero.
 */
int32_t GetMostRecentCommonAncestor(
    std::unordered_set<LogSumTracebackState*> *cur_states) {
  int32_t ans = 0;
  std::unordered_set<LogSumTracebackState*> prev_states;
  for (; cur_states->size() != 1; ans++) {
    CHECK(!cur_states->empty());
    for (LogSumTracebackState *s: cur_states) {
      for (const LogSumTracebackLink &l: s->prev_elements) {
        prev_states.insert(l.prev_state.get());
      }
    }
    cur_states->clear();
    cur_states->swap(prev_states);
  }
  return ans;
}


// Version of GetMostRecentCommonAncestor() for MaxTracebackState;
// see documentation for the other version.
int32_t GetMostRecentCommonAncestor(
    std::unordered_set<MaxTracebackState*> *cur_states) {
  int32_t ans = 0;
  std::unordered_set<MaxTracebackState*> prev_states;
  for (; cur_states->size() != 1; ans++) {
    CHECK(!cur_states->empty());
    for (MaxTracebackState *s: cur_states) {
      prev_states.insert(s->prev_state.get());
    }
    cur_states->clear();
    cur_states->swap(prev_states);
  }
  return ans;
}


/**
   A TraceBack() function exists for LogSumTracebackState and MaxTracebackState;
   it's used in DetState::Normalize().  It finds the cost and derivative
   information from getting rid of `num_steps` symbols from a symbol sequence.

       @param [in] cur_states   (This is consumed destructively, i.e. don't
                       expect it to contain the same set on exit).
                       A set of states; we'll iteratively trace back this
                       set one step at a time.    At entry it must have
                       size() == 1; it will also have size() == 1 after
                       `num_steps` steps.
       @param [in] num_steps   The number of steps to trace back
       @param [in] arc_weights_in  Weights on the arcs of the input FSA
       @param [out] weight_out  The output weight; will be the forward-backward
                       weight of the sub-graph whose final-state is
                       (*cur_states).front() and whose start-state is
                       the result of following that back for `num_steps` steps
                       (which will also be a single state, by virtue of how
                       the whole determinization algorithm works).  Will be
                       zero if num_steps == 0.
       @param [out] deriv_out  Some derivative information at the output
                       will be written to here, which tells us how the weight
                       `weight_out` varies as a function of the weights
                       on the arcs of the input FSA; it's a list
                       (input_arc_id, deriv) where, mathematically, 0 < deriv <= 1
                       (but we might still get exact zeros due to limitations
                       of floating point representation).
                       Note: the sum of the float values in this vector should
                       be equal to `num_steps`.

 */
void TraceBack(std::unordered_set<LogSumTracebackState*> *cur_states,
               int32_t num_steps,
               const float *arc_weights_in,
               float *weight_out,
               std::vector<std::pair<int32_t, float> > *deriv_out) {
  std::unordered_set<LogSumTracebackState*> prev_states;
  assert(cur_states.size() == 1);
  // In the standard forward-backward algorithm for HMMs this backward_prob
  // would, mathematically, be 0.0, but if we set it to the negative of the
  // forward prob we can avoid having to subtract the total log-prob
  // when we compute posterior/occupation probabilities for arcs.
  double cur_forward_prob = cur_states.front()->forward_prob;
  cur_states.front()->backward_prob = cur_forward_prob;
  deriv_out->clear();
  for (int32_t i = 0; i < num_steps; i++) {
    for (LogSumTracebackState *state_ptr: *cur_states) {
      double backward_prob = state_ptr->backward_prob;
      for (auto link: state_tr->prev_elements) {
        float arc_log_posterior = link.forward_prob + backward_prob;
        deriv_out->push_back(std::pair<int32_t, float>(link.arc_index, expf(log_posterior)));
        LogSumTracebackState *prev_state = link.prev_state.get();
        double new_backward_prob = backward_prob + arc_weights_in[link.arc_index];
        if (prev_states.insert(prev_state).second) {  // newly inserted
          prev_state->backward_prob = new_backward_prob;
        } else {
          prev_state->backward_prob = LogAdd(new_backward_prob,
                                             prev_state->backward_prob);
        }
      }
    }
    cur_states->clear();
    cur_states->swap(prev_states);
  }
  // failure of the next assertion may indicate many kinds of bugs in the
  // algorithm.
  CHECK_EQ(cur_states.size(), 1);
  double prev_forward_prob = cur_states.front()->forward_prob;
  *weight_out = cur_forward_prob - prev_forward_prob;
  // The following is mostly for ease of interpretability of the output;
  // conceptually the order makes no difference.
  std::reverse(deriv_out->begin(), deriv_out->end());
}


// See documentation of TraceBack for LogSumTracebackState, above.
// This version is simpler.
void TraceBack(std::unordered_set<MaxTracebackState*> *cur_states,
               int32_t num_steps,
               const float *,  // arc_weights_in, unused.
               float *weight_out,
               std::vector<int32_t> *deriv_out) {
  // we recompute the arc weight sum from arc_weights_in, which should
  // hopefully give
  float arc_weight_sum = 0.0;
  CHECK_EQ(cur_states.size(), 1);
  MaxTracebackState *state = cur_states->front();
  double cur_forward_prob = state->forward_prob;
  deriv_out->resize(num_steps);
  for (int32_t i = num_steps - 1; i >= 0; i--) {
    (*deriv_out)[i] = state->arc_id;
  }
  double prev_forward_prob = state->forward_prob;
  *weight_out = cur_forward_prob - prev_forward_prob;
}

// Priority queue templated on:
//   item queued = unique_ptr<DetState> (using pointer equality as comparison)
//   container type = vector<unique_ptr<DetState> >
//   less-than operator = DetStateCompare (which compares the forward_backward_prob).

template<class TracebackState>
using DetStatePriorityQueue = priority_queue<unique_ptr<DetState<TracebackState> >,
                                             vector<unique_ptr<DetState<TracebackState> > >,
                                             DetStateCompare<TracebackState> >;


template <class TracebackState>
class DetStateMap;

/*
  Conceptually, a determinized state in weighted FSA determinization would
  normally be a weighted subset of states in the input FSA, with the weights
  normalized somehow (e.g. subtracting the sum of the weights).

  Two determinized states are equal if the states and weights are the same.  To
  ensure differentiability, our assumption is that in general no two arcs in the
  input FSA have identical weights.  We argue that two determinized states can
  always be represented as a base-state and a symbol sequence.  Imagine that we
  follow arcs with that symbol sequence from the base-state, and then in case we
  reach the same states in the different ways we always select the best path
  from the base-state.  That process gives us a set of states and weights.  We
  argue that this representation is unique.  (If not, it won't matter actually;
  it will just give us an output that's less minimal than it could be).


  We're not really following the Google guidelines by not having _ at the end of
  class members, but this is more struct-like (members are public).
 */
template <class TracebackState>  // TracebackState == MaxTracebackState or LogSumTracebackState
class DetState {


 public:
  using DerivOutputType = typename TracebackState::DerivOutputType;
  // .. and DerivOutputType == int32_t or pair<int32_t, float>
  // respectively.

  // Constructor for the initial state of the determinized FSA
  DetState(): seq_len(0), output_state(-1), normalized(true) {
    // the constructor that takes no args gives us what we need for the
    // start-state.
    elements[0] = std::make_shared<TracebackState>();
  }

  // TODO: constructor for start-state.

  DetState(int32_t seq_len, int32_t src_output_state,
           int32_t pending_symbol):
      seq_len(seq_len),
      output_state(-1), // Not known yet
      normalized(false),
      src_output_state(src_output_state),
      pending_symbol(pending_symbol) { } // .. and forward_backward_weight undefined

  /**
     Process incoming arc to this DetState.  See documentation for
     constructor of TracebackState (which will be MaxTracebackState or
     LogSumTracebackState).
         @param [in] state_id  State-id, in input FSA, into which this arc enters.
                   [Note: a DetState is a weighted subset of state-ids.]
         @param [in] incoming_arc_index  Arc in input FSA that enters state
                   `state_id`.
                   @param [in] arc_symbol  The symbol on
   */
  void AcceptIncomingArc(int32_t state_id,
                         const std::shared_ptr<TracebackState> &src,
                         int32_t incoming_arc_index,
                         int32_t arc_weight) {
    auto iter = elements.find(state_id);
    if (iter == elements.end()) {
      elements[state_id] = std::make_shared<TracebackState>(
          state_id, src, incoming_arc_index, arc_weight);
    } else {
      iter.second->Accept(
          src, incoming_arc_index, arc_symbol, arc_weight);
    }
  }

  // Length of sequence of symbols (from the base state) leading to this DetState.
  // each DetState can be described by: start from base_state, follow paths
  // with a specific symbol sequence, and the weighted set of states that you
  // reach corresponds to this DetState.
  //
  // The sequence of symbols, and the base_state, can be found by tracing back
  // one of the DetStateElements in the doubly linked list (it doesn't matter
  // which you pick, the result will be the same).
  int32_t seq_len;

  // `output_state` is the state in the output FSA that this determinized
  // state corresponds to.  (Only known if `normalized` is true; otherwise, -1).
  int32_t output_state;

  // `normalized` is true if this DetState is known to be normalized (meaning:
  // we have reduced seq_len as much as possible).  DetStates that are not
  // normalized will not yet have an `output_state`.
  bool normalized;

  // The following two elements are only relevant if `normalized` is false.
  // The purpose is so that we can delay outputting the arc leading to this
  // DetState, until we know that this DetState will end up having arcs
  // leaving it processed (see ProcessArcs()).  For pruned determinization,
  // this avoids unnecessary work.
  //
  // src_output_state is the state, in the output FSA, of the preceding state
  // from which we generated this state.  In the end any given state in the output FSA may
  // have many preceding-states; this is just the one from which this
  // particular DetState structure was generated.  It's remembered so that
  // we can correctly output the arc in the output FSA from `src_output_state`.
  int32_t src_output_state;
  // pending_symbol is the symbol, in the output FSA, on the arc from
  // src_output_state.
  int32_t pending_symbol;


  // `elements` can be thought of as weighted subsets of states in the input
  // FSA, that also stores some traceback information that lets us compute
  // derivatives.
  // It's a map from (state-id in input FSA) -> its corresponding TracebackState.
  std::unordered_map<int32_t, std::shared_ptr<TracebackState> > elements;

  // This is the weight on the best path that includes this determinized state.
  // It's needed to form a priority queue on DetStates, so we can process them
  // best-first.  It is computed as: the forward-weight on `base_state`,
  // plus the best/most-positive of: (the weight in a DetStateElement plus
  // the backward-weight of the state associated with that DetStateElement).
  double forward_backward_weight;


  /*
    Process arcs leaving this determinized state, possibly creating new determinized
    states in the process.  Note: Normalize() should already have been called on
    *this.

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
              @param [out] arcs_out   Output-FSA arcs-- those leaving this
                                 determinized state-- will be appended to here.
              @param [out] arc_weights_out  Weights for the output arcs will
                                 be appended to here.
              @param [out] derivs_per_arc  Derivative information for the output
                                 arcs will be appended to here: either sequences
                                 of int32_t (corresponding to input arcs), or
                                 lists of pair<int32_t, float>, corresponding
                                 to weighted input arcs.
              @param [in,out] state_map  Maps from DetState to int32_t state-id
                                 in the output FSA.
              @return   Returns a number that approximately indicates how much
                       computation was done (so we can avoid it taking too long).
  */
  int32_t ProcessArcs(const WfsaWithFbWeights &wfsa_in,
                      float prune_cutoff,
                      vector<Arc> *arcs_out,
                      vector<float> *arc_weights_out,
                      vector<DerivOutputType> *derivs_per_arc,
                      DetStateMap<TracebackType> *state_map,
                      DetStatePriorityQueue<TracebackType> *queue);

  // Computes the forward-backward weight of this DetState.  This is
  // related to the best cost of any path through the output FSA
  // that included this determinized state.  I say "related to"
  // because while it should be exact in the Max case, in the
  // LogSum case the relationship is a bit more complicated;
  // maybe just best to say that this is a weight that we use
  // for pruning.
  //   @param [in] backward_state_weight   Array, indexed by
  //                    state in input WFSA, of the weight from this state
  //                    to the end.  (Of the best path or the sum of paths,
  //                    depending how it was computed; this will of
  //                    course affect the pruning).
  void ComputeFbWeight(const float *backward_state_weights);

  /*
    Normalizes this DetState by reducing seq_len to the extent possible
    and outputting the weight and derivative info corresponding to this
    reduction of sequence length.  Recall that these determinized states
    are represented as a (base state, and a sequence of symbols that we
    followed from the base state).  This allows a smaller set of
    input FSAs to be determinized than the normal weighted-subet-of-states
    formulation, equivalent (I believe) to the assumption that all
    the weights are distinct and have no 'special relationships' between
    them, i.e. no equalities like a + b = c.  This kind of requirement is
    necessarly for differentiablity.

    This function also sets the forward_backward_weight field.

       @param [in] wfsa_in  The weighted FSA we are determinizing
       @param [out] removed_weight  The part of the weight that was
                      removed when we reduced `seq_len`, if any, will
                      be written to here (else 0.0).
   */
  void Normalize(const WfsaWithFbWeights &wfsa_in,
                 float *removed_weight,
                 std::vector<DerivOutputType> *deriv_info);
};

template <class TracebackState>
bool DetStateCompare<TracebackState>::operator()(
    const shared_ptr<DetState<TracebackState> > &a,
    const shared_ptr<DetState<TracebackState> > &b) {
  return a->forward_backward_weight < b->forward_backward_weight;
}


/*
  This class maps from determinized states (DetState) to integer state-ids
  in the determinized output.  Caution: it uses a randomized algorithm that
  could in principle produce collisions that would generate wrong output.
  We don't think this will ever happen though (128-bit representations).
 */
template <class TracebackType>
class DetStateMap {
 public:

  /*
    Looks up the output state-id corresponding to a specific DetState structure,
    creating a new output-state if necessary.  This does not store any pointers
    to the DetState or its contents, so you can delete the DetState without
    affecting this object's ability to map an equivalent DetState to the same
    state-id.

       @param [in,out] a  The DetState whose state-id we are looking up.
                         The integer id of the output-FSA state will be written
                         to its `output_state` field, which at entry is assumed
                         to be unset.
        @return  Returns true if this was a NEWLY CREATED state,
              false otherwise.
   */
  bool GetOutputState(DetState<TracebackState> *a) {
    std::pair<uint64_t, uint64_t> compact;
    DetStateToCompact(a, &compact);
    auto p = map_.insert({compact, cur_output_state});
    bool inserted = p.second;
    if (inserted) {
      a->state_id = cur_output_state_++;
      return true;
    } else {
      a->state_id = p.first->second;
      return false;
    }
  }

  int32_t size() const { return cur_output_state_; }

 private:
  struct PairHasher {
    size_t operator () (const std::pair<uint64_t, uint64_t> &p) const {
      return static_cast<size_t>(p.first);
    }
  }


  int32_t cur_output_state_{0};
  /* maps from 128-bit key (stored as a pair of uint64_t's) to the int32_t state-id. */
  std::unordered_map<std::pair<uint64_t, uint64_t>, uint32_t,
                     PairHasher> map_;

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

/*
  Convenience function that normalizes the state and outputs the arc for it.

  Returns true if the state was newly added (not already present in
  `state_map`).
 */
template <class TracebackState>
bool NormalizeStateAndOutputArc(
    DetState *state,
    const WfsaWithFbWeights &wfsa_in,
    float prune_cutoff,
    vector<Arc> *arcs_out,
    vector<float> *arc_weights_out,
    vector<vector<typename TracebackState::DerivOutputType> > *derivs_per_arc,
    DetStateMap *state_map) {
  float arc_weight;
  std::vector<DerivOutputType> deriv_info;
  state->Normalize(wfsa_in, &arc_weight, &deriv_info);
  int32_t next_state_id;
  bool is_new_state = state_map->GetOutputState(state);
  arcs_out->push_back({this->state_id, next_state_id, label});
  arc_weights_out->push_back(arc_weight);
  derivs_per_arc->push_back(std::move(deriv_info));
  return is_new_state;
}


template <class TracebackState>
int32_t DetState<TracebackState>::ProcessArcs(
    const WfsaWithFbWeights &wfsa_in,
    double prune_cutoff,
    vector<Arc> *arcs_out,
    vector<float> *arc_weights_out,
    vector<vector<typename TracebackState::DerivOutputType> > *derivs_per_arc,
    DetStateMap *state_map,
    DetStatePriorityQueue *queue) {
  int32_t num_steps = 0;

  std::unordered_map<int32_t, shared_ptr<DetState<TracebackStats> > > label_to_state;

  Fsa *fsa = wsfa_in.fsa;
  const float *arc_weights = wfsa_in.arc_weights;
  for (const std::shared_ptr<TracebackState> &state_ptr: elements) {
    int32_t state_id = state_ptr->state_id,
        begin_arc = fsa->arc_indexes[state_id],
        end_arc = fsa->arc_indexes[state_id + 1];
    num_steps += end_arc - begin_arc;
    for (int32_t a = begin_arc; a < end_arc; ++a) {
      const Arc &arc = fsa->arcs[a];
      float weight = arc_weights[a];
      int32_t label = arc.label;


      auto ret = label_to_state.insert({label, nullptr});
      auto iter = ret.first;
      if (ret.second) {  // Inserted -> this label was not a key in this map.
                         // Allocate new DetState.
        iter->second = std::make_shared<DetState<TracebackState> >(seq_len + 1,
                                                                   this->output_state,
                                                                   label);
      }
      TracebackState *state = iter->second.get();
      state->Accept(state_ptr, a, arc.label, weight);
    }
  }
  CHECK(!label_to_state.empty() ||
        elements[0]->state_id == fsa->FinalState());  // I'm assuming the input
                                                      // FSA is connected.


  for (auto iter = label_to_state.begin();
       iter != label_to_state.end(); ++iter) {
    std::shared_ptr<DetState> &det_state = iter->second;

    float arc_weight;
    std::vector<DerivOutputType> deriv_info;
    det_state->Normalize(wfsa_in, &arc_weight, &deriv_info);
    if (det_state->forward_backward_weight >= prune_cutoff) {
      bool is_new_state = state_map->GetOutputState(state);
      arcs_out->push_back({this->state_id, next_state_id, label});
      arc_weights_out->push_back(arc_weight);
      derivs_per_arc->push_back(std::move(deriv_info));
      if (is_new_state)
        queue->push(det_state);
    }
  }
  return num_steps;
}

template <class TracebackState>
void DetState<TracebackState>::ComputeFbWeight(
    const float *backward_state_weights) {
  forward_backward_weight = -std::numeric_limits<double>::infinity();
  for (auto p: elements) {
    TracebackState *state = p.second.get();
    forward_backward_weight = max(forward_backward_weight,
                                  state->forward_prob +
                                  backward_state_weights[state->state_id]);
  }
}

template <class TracebackState>
double LogSumOrMax<TracebackState>(double, double);

template <>
double LogSumOrMax<MaxTracebackState>(double a, double b) {
  return max(a, b);
}
template <>
double LogSumOrMax<MaxTracebackState>(double a, double b) {
  return LogSum(a, b);
}


template <class TracebackState>
void DetState<TracebackState>::Normalize(const WfsaWithFbWeights &wfsa_in,
                                         float *removed_weight,
                                         std::vector<DerivOutputType> *deriv_info) {
  std::unordered_set<TracebackState*> cur_states;

  double fb_prob = -std::numeric_limits<double>::infinity();
  for (auto p: elements) {
    TracebackState *state = p.second.get();
    fb_prob = LogSumOrMax<TracebackState>(
        fb_prob,
        state->forward_prob + wfsa_in.backward_state_weights[state->state_d]);
    cur_states.insert(state);
  }

  int32_t new_seq_len = GetMostRecentCommonAncestor(&cur_states);
  // now cur_states.size() == 1.
  CHECK_EQ(cur_states.size(), 1);
  CHECK_LE(new_seq_len, seq_len);

  const TracebackState *base_state = cur_states.front().get();
  // The following statement is a correction term that we add to
  // forward_backward_prob, in which we replace the forward_prob in the DetState
  // (which will have been computed in a path-dependent way) with the
  // forward_prob in wfsa_in.  Note: the values of state->forward_prob above can
  // be thought of as base_state->forward_prob plus some value that only depends
  // on the symbol sequence.  The point of this is to ensure that
  // this->forward_backward_prob (which is used for pruning) depends only on the
  // base_state and the symbol sequence, and not on "how we got here", i.e.  the
  // history of DetStates from which this one is derived via ProcessArcs().
  fb_prob += wfsa_in.forward_state_weights[base_state->state_id] -
      base_state->forward_prob;
  // set thi->forward_backward_prob; it will affect pruning.
  this->forward_backward_prob = fb_prob;
  this->seq_len = new_seq_len;

  // the following will set removed_weight and deriv_info.
  TraceBack(&cur_states, seq_len - new_seq_len,
            wfsa_in.arc_weights,
            removed_weight, deriv_info);

  normalized = true;
}



void DeterminizePrunedLogSum(
    const WfsaWithFbWeights &wfsa_in,
    float beam,
    int64_t max_step,
    Fsa *fsa_out,
    std::vector<float> *arc_weights_out,
    std::vector<std::vector<std::pair<int32_t, float> > > *arc_derivs_out) {
  CHECK_GT(beam, 0);

  DetStatePriorityQueue<LogSumTracebackState> queue;
  DetStateMap<LogSumTracebackState> map;

  std::shared_ptr<DetState> start_state = std::make_shared<DetState>();

  std::vector<Arc> arcs_out;
  arc_weights_out->clear();
  arc_derivs_out->clear();

  bool ans = map.GetOutputState(start_state.get());
  CHECK(ans && ans->state_id == 0);

  if (max_step <= 0)
    max_step = std::numeric_limits<int64_t>::max();
  int64_t num_steps = 0;
  int32_t block_size = 32;  // process a number of queue elements at a time
                            // between certain checks..

  double total_prob = wfsa_in.backward_state_weights[0],
      prune_cutoff = total_prob - beam;
  while (num_steps < max_step && !queue.empty()) {
    std::shared_ptr<DetState> state = queue.top();
    queue.pop();
    num_steps += state->ProcessArcs(wfsa_in, prune_cutoff, arcs_out,
                                    arc_weights_out, arc_derivs_out,
                                    &map, &queue);
  }
}

// TODO: do the max version of Determinize(), which is much the same as the
// LogSum version.

}  // namespace k2
