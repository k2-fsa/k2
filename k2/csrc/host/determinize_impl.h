/**
 * @brief
 * determinize_impl
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_DETERMINIZE_IMPL_H_
#define K2_CSRC_HOST_DETERMINIZE_IMPL_H_

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/host/weights.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {
/*
  HOW THIS WORKS

  This is FSA determinization that also outputs derivative information that says
  how the weights on the arcs in the output FSA vary with the weights on the
  arcs in the input FSA.

  INTRO TO DETERMINIZATION.

  The problem in determinization of a weighted FSA is to find a deterministic
  FSA (i.e. one that has no two arcs leaving any given state with the same
  symbol on), which is equivalent to the input FSA (meaning: the weight it
  assigns to any given symbol-sequence is the same as the input FSA).  In this
  explanation, assume epsilons don't exist, if the input FSA had epsilons we'd
  get rid of them prior to determinization.


  SUBSET-BASED ALGORITHMS

   In general, in determinization algorithms, states in the output FSA
  correspond to weighted subsets of states in the input FSA.  The overall
  structure of these algorithms will be:

   - Let state 0 in the output FSA correspond to the weighted subset { 0, 0.0 }
     in the input FSA, where the 0 is the start state-id in the input FSA
     and the 0.0 is the weight (interpret this as a log-prob).
     Put that in the queue.

   - While (queue is not empty)
     Pop output-state from the queue
     Process successor states of this output-state.

  Obviously most of the detail in the outline above resides in "Process
  successor states of this output-state."  Let's discuss the unweighted
  case first.

   *Unweighted case

   Each output-state corresponds to some subset of input-states, say, { s1, s2,
   .. }.  The set of labels on arcs leaving the output-state will correspond to
   the set of labels on all the arcs leaving s1, s2 and so on; and the
   destination-states of those arcs will correspond to the sets of
   destination-states of those arcs.  The algorithm requires us to store a map
   from (subset of input-state-ids) to (output-state-id).

   *Weighted case

   In the weighted case, the difference is that instead of a set of
   input-state-ids we have a weighted subset of them, and the map is from these
   weighted subsets to state-ids in the output FSA.  The weights in the weighted
   subsets have to be normalized somehow.  The natural normalization is in the
   "max/tropical-semiring" case to have the most negative weight be 0.0, and in
   the "log-sum/log-semiring" case to have the log-sum be 0.0.  Imagine
   we have a function:
      Normalize (unnormalized-weighted-subset) -> normalized-weighted-subset,
  leftover-weight E.g., in the Max case: Normalize( { (6, 1.0), (7, 5.0) } ) ->
  { (6, 0.0), (7, 4.0) }, 1.0 The "leftover-weights" become the weights on the
  arcs in the output FSA.


   *The problem with differentiability

   Consider how to differentiate the weights of the output weighted FSA
   w.r.t. those of the input.  The problem with differentiability if we use the
   algorithm above is the case of special symmetries.  What if two weighted
   subsets happen to coincide because there was an exact relationship between
  the values of the weights in the input FSA, but there was no *structural*
  reason in the input FSA why those weighted subsets have to be the same?  Then
  we have a problem with how to differentiate, because any small change in the
   input weights would lead to a structural change in the output FSA.


  OUR ALGORITHM

    *Different representation of subsets

    Our algorithm is still very similar to the subset-based algorithms mentioned
    above, and it still involves weighted subsets, but we use a different
    representation of them.  Our representation (think of this as the key in
    the map) is:  ( base_state, symbol_sequence ).  The relationship with
    the weighted subset is: start from state `base_state` in the input FSA,
    traverse all sequences of arcs that have sequence `symbol_sequence` on them,
    and the weighted set of states you end up with is the weighted subset
    in the algorithm above.

    *Different normalization

    Our form of "normalization" of this representation is different too.  The
    normalization is to make `symbol_sequence` as short as possible, and advance
    `base_state` to compensate.  For instance, if `symbol_sequence` is `a b c
    d`, but the weighted subset of states we can reach by this symbol sequence
    is the same as what we'd get by moving `base_state` forward two steps and
    letting `symbol_sequence` be just `c d`, then the latter representation is
    the canonical one (assuming that was the largest prefix we could remove).
    Algorithmically, finding the "most recent base_state" involves finding the
    most recent common ancestor in a directed tree of paths through the input
    FSA (in the max case) or a directed lattice of paths through the input FSA
    (in the log-sum case).

    The weights on arcs are related to the total weight of paths from `original
    base_state` to `new base_state`, (counting only paths that have the removed
    symbol sequence `a b`).  Just as with the subset algorithm, these
    weights are what gets "spit out" by the normalization process; we simply
    have a different normalization process.


  PRUNED DETERMINIZATION

    We support pruned determinization.  We won't describe all of the details
    here, but it involves a priority queue of determinized states, so we always
    process the "best" queue element first, and we may terminate before the
    queue is empty.


  IMPLEMENTATION DETAILS

    A few details on the implementation:

     - To save memory space, the process of hashing from `base_state,
  symbol_seq` to output state-id maps them to a fixed-size 128-bit value.  This
  could in principle generate collisions which would generate incorrect output,
       but we consider that vanishingly improbable.

 */

struct MaxTracebackState {
  using DerivType = int32_t;

  int32_t state_id;  // state-id in the input FSA

  int32_t arc_id;  // arc-id in input FSA of the arc that enters state
                   // `state_id` (or -1 if this is the start state).  It will
                   // be the best arc if there were multiple possible arcs
                   // from the base_state to this state with the same symbol
                   // sequence.

  // prev_state is the state we trace back to (the previous state), which is the
  // src_state of the arc numbered arc_id.  It will be nullptr if state_id == 0
  // (start state).
  std::shared_ptr<MaxTracebackState> prev_state;

  double forward_prob;  // The best forward log-probability from the start
                        // state to this state (along whichever specific
                        // sequence of symbols we took to get here)

  // This constructor is for the start-state (state zero) of the input FSA.
  explicit MaxTracebackState(int32_t state_id = 0, double forward_prob = 0.0)
      : state_id(state_id),
        arc_id(-1),
        prev_state(nullptr),
        forward_prob(forward_prob) {}

  /**
     @param [in] state_id  State in input FSA that this corresponds to
     @param [in] src   Previous MaxTracebackState that we'll point back
                      to, or NULL
     @param [in] incoming_arc_index  Arc-index in input FSA.
                      Its src_state will equal src->state_id,
                      its dest_state will equal state_id.
     @param [in] arc_weight   Weight on the input arc
   */
  MaxTracebackState(int32_t state_id,
                    const std::shared_ptr<MaxTracebackState> &src,
                    int32_t incoming_arc_index, float arc_weight)
      : state_id(state_id),
        arc_id(incoming_arc_index),
        prev_state(src),
        forward_prob(src->forward_prob + arc_weight) {}

  /*
    This takes the same args as the constructor.  It will update the traceback
    info if this incoming arc had higher weight.
  */
  void Accept(const std::shared_ptr<MaxTracebackState> &src, int32_t arc_index,
              float arc_weight) {
    double new_forward_prob = src->forward_prob + arc_weight;
    if (new_forward_prob > forward_prob) {
      forward_prob = new_forward_prob;
      arc_id = arc_index;
      prev_state = src;
      // state_id doesn't change.
    }
  }
};

class LogSumTracebackState;

/*
  This struct is used inside LogSumTracebackState; it represents an
  arc that traces back to a previous LogSumTracebackState.
  A LogSumTracebackState represents a weighted collection of paths
  terminating in a specific state.
*/
struct LogSumTracebackLink {
  std::shared_ptr<LogSumTracebackState> prev_state;
  // `prev_state` is the state that this points back to.

  int32_t arc_index;  // Index (in input FSA) of this arc from prev_state to the
                      // destination state (in whose LogSumTracebackState this
                      // LogSumTracebackLink will be located).

  double
      forward_prob;  // The total forward log-probability from the start
                     // state to the end of this arc just before it joins the
                     // node (conceptually).  Note: this is only the total
                     // forward log-prob limited to whatever symbol-sequence
                     // got us to this point...  there may be other
                     // symbol-sequences terminating in this state.
                     // (That symbol-sequence can be obtained by tracing back
                     // this data structure till you hit a LogSumTracebackState
                     // with prev_state == nullptr (and state_id == 0).

  LogSumTracebackLink(const std::shared_ptr<LogSumTracebackState> &src,
                      int32_t arc_index, float arc_weight);
};

/*
  This stores traceback information for the log-sum case.  Rather than a tree
  structure, the LogSumTracebackStates interconnect with a lattice structure.

  It can be thought of as as a weighted set of paths from the start state to a
  particular state.  It will be limited to the subset of paths that have a
  specific symbol sequence.
*/
struct LogSumTracebackState {
  using DerivType = std::pair<int32_t, float>;

  // `prev_elements` is, conceptually, a list of incoming arcs with associated
  // weights.
  std::vector<LogSumTracebackLink> prev_elements;

  int32_t state_id;  // The state-id in the input FSA that this
                     // LogSumTracebackState corresponds to.

  double forward_prob;  // The total forward log-probability from the start
                        // state to this state (along whichever specific
                        // sequence of symbols we took to get here; it's not
                        // necessarily the best forward-prob in the lattice
                        // that would take us to this state).  Will equal the
                        // log-sum of the forward_probs of the prev_elements.

  double backward_prob;  // Used temporarily in algorithms as a backward prob.
                         // Undefined most of the time.

  explicit LogSumTracebackState(int32_t state_id = 0, double forward_prob = 0.0)
      : state_id(state_id), forward_prob(forward_prob) {}

  /*
     @param [in] state_id  State in input FSA
     @param [in] src  Previous LogSumTracebackState that we'll point back to
     @param [in] incoming_arc_index.  Arc-index in input FSA.
                      Its src_state will equal src->state_id, its dest_state
                      will equal state_id.
     @param [in] arc_weight   Weight on the arc
   */
  LogSumTracebackState(int32_t state_id,
                       const std::shared_ptr<LogSumTracebackState> &src,
                       int32_t incoming_arc_index, float arc_weight)
      : state_id(state_id), forward_prob(src->forward_prob + arc_weight) {
    prev_elements.emplace_back(src, incoming_arc_index, arc_weight);
  }

  /*
     Accept a new incoming link.  The args are the same as for the constructor
     just above; see documentation there.

      @param [in] src  Previous LogSumTracebackState that we'll point back to
      @param [in] incoming_arc_index.  Arc-index in input FSA.
                      Its src_state will equal src->state_id, its dest_state
                      will equal this->state_id.
     @param [in] arc_weight   Weight on the incoming arc

   */
  void Accept(const std::shared_ptr<LogSumTracebackState> &src,
              int32_t arc_index, float arc_weight) {
    prev_elements.emplace_back(src, arc_index, arc_weight);
    this->forward_prob =
        LogAdd(this->forward_prob, src->forward_prob + arc_weight);
  }
};
/*
  Find the most recent common ancestor LogSumTracebackState of a set of
  LogSumTracebackStates, and return the number of links we had to follow to get
  there (i.e. the length of symbol sequence).

    @param [in,out] cur_states   A set of TracebackStates that we'll
                  trace back from.  Must be nonempty.  Equality
                  here is simply pointer identity.

                  At exit it will contain a single member which will
                  be the most recent common ancestor.

    @return  Returns the number of links we had to follow (>=0).  If
             `cur_states.size() == 1` this will be zero.
 */
int32_t GetMostRecentCommonAncestor(
    std::unordered_set<LogSumTracebackState *> *cur_states);

// Version of GetMostRecentCommonAncestor() for MaxTracebackState;
// see documentation for the other version.
int32_t GetMostRecentCommonAncestor(
    std::unordered_set<MaxTracebackState *> *cur_states);

/**
   A TraceBack() function exists for LogSumTracebackState and MaxTracebackState;
   it's used in DetState::Normalize().  It finds the cost and derivative
   information from getting rid of `num_steps` symbols from a symbol sequence.

       @param [in] cur_states   (This is consumed destructively, i.e. don't
                       expect it to contain the same set on exit).
                       A set of states; we'll iteratively trace back this
                       set one step at a time.    At entry it must have
                       size() == 1; it will also have size() == 1 at exit.
       @param [in] arcs_in    Array of arcs of the FSA that we're doing
                       traceback in; needed only for lookup of the
                       weights.
       @param [in] num_steps   The number of steps to trace back
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
                       (input_arc_id, deriv) where, mathematically, 0 < deriv <=
   1 (but we might still get exact zeros due to limitations of floating point
   representation). Note: the sum of the float values in this vector at exit
   should be equal to `num_steps`.

 */
void TraceBack(std::unordered_set<LogSumTracebackState *> *cur_states,
               int32_t num_steps, const Arc *arcs_in, float *weight_out,
               std::vector<std::pair<int32_t, float>> *deriv_out);

// The TraceBack function for MaxTracebackState.  See documentation of TraceBack
// for LogSumTracebackState, above.  This version is simpler.
void TraceBack(std::unordered_set<MaxTracebackState *> *cur_states,
               int32_t num_steps, const Arc *arcs_in, float *weight_out,
               std::vector<int32_t> *deriv_out);

template <class TracebackState>
class DetState;

template <class TracebackState>
struct DetStateCompare {
  bool operator()(const std::shared_ptr<DetState<TracebackState>> &a,
                  const std::shared_ptr<DetState<TracebackState>> &b) {
    return a->forward_backward_prob < b->forward_backward_prob;
  }
};
// Priority queue template arguments:
//   item queued = shared_ptr<DetState> (using pointer equality as comparison)
//   container type = vector<shared_ptr<DetState> >
//   less-than operator = DetStateCompare (which compares the
//   forward_backward_prob).
template <class TracebackState>
using DetStatePriorityQueue =
    std::priority_queue<std::shared_ptr<DetState<TracebackState>>,
                        std::vector<std::shared_ptr<DetState<TracebackState>>>,
                        DetStateCompare<TracebackState>>;

template <class TracebackState>
class DetStateMap;
/*
  This represents a determinized state.  Initially it has normalized == false
  and it represents an un-normalized determinized state (see intro at the top
  of this file), or an un-normalized determinized state under construction
  (we add to it using AcceptIncomingArc()).

  After we call Normalize() on it, it is a normalized determinized-state (this
  also outputs the weight you need for the incoming arc).

  After that

 */

template <class TracebackState>  // TracebackState == MaxTracebackState or
                                 // LogSumTracebackState
class DetState {
 public:
  using DerivType = typename TracebackState::DerivType;
  // DerivType == int32_t for MaxTracbackState, or
  // pair<int32_t, float> for LogSumTracebackState.

  // Constructor for the initial state of the determinized FSA
  DetState() : seq_len(0), normalized(true) {
    // the constructor of TracebackState that takes no args gives us what we
    // need for the start-state.
    elements[0] = std::make_shared<TracebackState>();
  }

  /*
     Constructor (this is the one that's normally used).
       @param [in] seq_len  Length of symbol sequence from its
                           base_state (this is before normalization).
                           Will be the seq_len of the source det_state plus
                           one.  This seq_len may end up getting reduced
                           when Normalize() is called (reducing seq_len
                           implicitly advances the base_state).
   */
  explicit DetState(int32_t seq_len)
      : seq_len(seq_len),
        normalized(false) {}  // .. and forward_backward_prob undefined

  /**
     Process incoming arc to this DetState.  See documentation for
     constructor of TracebackState (which will be MaxTracebackState or
     LogSumTracebackState).
         @param [in] state_id  State-id, in input FSA, into which this arc
     enters. [Note: a DetState is a weighted subset of state-ids.]
         @param [in] src  The preceding state (from which the arc leaves).
                    This will be a member of the `elements` of the "parent"
                    DetState, i.e. the DetState in whose ProcessArcs() function
                    this DetState was created.
         @param [in] incoming_arc_index  Arc in input FSA that enters state
                   `state_id`.
         @param [in] arc_weight  The weight on this arc
   */
  void AcceptIncomingArc(int32_t state_id,
                         const std::shared_ptr<TracebackState> &src,
                         int32_t incoming_arc_index, float arc_weight) {
    NVTX_RANGE(K2_FUNC);
    auto ret = elements.insert({state_id, nullptr});
    if (ret.second) {  // No such state existed in `elements`
      ret.first->second = std::make_shared<TracebackState>(
          state_id, src, incoming_arc_index, arc_weight);
    } else {  // A state with this staste_id existed in `elements`.
      ret.first->second->Accept(src, incoming_arc_index, arc_weight);
    }
  }

  // State-id in the output FSA.  Only defined after
  // DetStateMap::GetOutputState() is called. A DetState that is in the queue
  // should have this defined.
  int32_t state_id;

  // Length of sequence of symbols (from the base state) leading to this
  // DetState. each DetState can be described by: start from base_state, follow
  // paths with a specific symbol sequence, and the weighted set of states that
  // you reach corresponds to this DetState.
  //
  // The sequence of symbols, and the base_state, can be found by tracing back
  // one of the DetStateElements in the doubly linked list (it doesn't matter
  // which you pick, the result will be the same).
  int32_t seq_len;

  // `normalized` is true if this DetState is known to be normalized (meaning:
  // we have reduced seq_len as much as possible).  DetStates that are not
  // normalized will not yet have an `output_state`.
  bool normalized;

  // `elements` can be thought of as weighted subsets of states in the input
  // FSA, that also stores some traceback information that lets us compute
  // derivatives.
  // It's a map from (state-id in input FSA) -> its corresponding
  // TracebackState.
  std::unordered_map<int32_t, std::shared_ptr<TracebackState>> elements;

  // This is the weight on the best path that includes this determinized state.
  // It's needed to form a priority queue on DetStates, so we can process them
  // best-first.  It is computed as: the forward-weight on `base_state`,
  // plus the best/most-positive of: (the weight in a DetStateElement plus
  // the backward-weight of the state associated with that DetStateElement).
  double forward_backward_prob;

  /*
    Process arcs leaving this determinized state, possibly creating new
    determinized states in the process.  Note: Normalize() should already have
    been called on *this.

              @param [in] wfsa_in  The input FSA that we are determinizing,
                                   along with forward-backward weights.
                                   The input FSA should normally be
                                   epsilon-free as epsilons are treated as
                                   a normal symbol.
              @param [in] prune_cutoff   Cutoff on forward-backward likelihood
                                 that we use for pruning; will equal
                                 wfsa_in.backward_state_weights[0] - prune_beam.
                                 Will be -infinity if we're not doing pruning.
              @param [out] arcs_out   Output-FSA arcs-- those leaving this
                                 determinized state-- will be appended to here.
              @param [out] derivs_per_arc  Derivative information for the output
                                 arcs will be appended to here: either sequences
                                 of int32_t (corresponding to input arcs), or
                                 lists of pair<int32_t, float>, corresponding
                                 to weighted input arcs.
              @param [in,out] state_map  Maps from DetState to int32_t state-id
                                 in the output FSA.
              @return   Returns a number that approximately indicates how much
                       computation was done (so we can avoid it taking too
                       long).
  */
  int32_t ProcessArcs(const WfsaWithFbWeights &wfsa_in, double prune_cutoff,
                      std::vector<Arc> *arcs_out,
                      std::vector<std::vector<DerivType>> *derivs_per_arc,
                      DetStateMap<TracebackState> *state_map,
                      DetStatePriorityQueue<TracebackState> *queue);

  /*
    A version of `ProcessArcs` above without pruning.
  */
  int32_t ProcessArcs(const Fsa &fsa_in, std::vector<Arc> *arcs_out,
                      std::vector<std::vector<DerivType>> *derivs_per_arc,
                      DetStateMap<TracebackState> *state_map,
                      DetStatePriorityQueue<TracebackState> *queue);

 private:
  /*
    Process arcs leaving this determinized state and write its successor
    DetStates (unnormalized) to label_to_state. Will be called in `ProcessArcs`.
          @param [in] fsa     The input FSA that we are determinizing.
          @param [out] label_to_state Maps from label to the successor
                              DetStates (unnormalized) of this determinized
                              state.
          @return   Returns a number that approximately indicates how much
                    computation was done (so we can avoid it taking too long).
  */
  int32_t GetDetStatesSuccessor(
      const Fsa &fsa,
      std::unordered_map<uint32_t, DetState<TracebackState> *> &label_to_state);

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

    This function also sets the forward_backward_prob field.

       @param [in] wfsa_in  The weighted FSA we are determinizing
       @param [out] removed_weight  The part of the weight that was
                      removed when we reduced `seq_len`, if any, will
                      be written to here (else 0.0).
   */
  void Normalize(const WfsaWithFbWeights &wfsa_in, float *removed_weight,
                 std::vector<DerivType> *deriv_info);
  /*
    A version of `Normalize` which does not require forward/backward weights,
    it will be called in the un-pruned version of `ProcessArcs`.
  */
  void Normalize(const Fsa &fsa_in, float *removed_weight,
                 std::vector<DerivType> *deriv_info);
};

template <class TracebackState>
int32_t DetState<TracebackState>::GetDetStatesSuccessor(
    const Fsa &fsa,
    std::unordered_map<uint32_t, DetState<TracebackState> *> &label_to_state) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_steps = 0;
  const auto arcs = fsa.data + fsa.indexes[0];
  for (const auto &elem : elements) {
    const auto &state_ptr = elem.second;
    int32_t state_id = state_ptr->state_id, begin_arc = fsa.indexes[state_id],
            end_arc = fsa.indexes[state_id + 1];
    num_steps += end_arc - begin_arc;
    for (int32_t a = begin_arc; a < end_arc; ++a) {
      const int32_t curr_arc = a - fsa.indexes[0];
      const Arc &arc = arcs[curr_arc];
      int32_t label = arc.label;
      auto ret = label_to_state.insert({label, nullptr});
      auto iter = ret.first;
      if (ret.second) {  // Inserted -> this label was not a key in this map.
                         // Allocate new DetState.
        iter->second = new DetState<TracebackState>(seq_len + 1);
      }
      DetState<TracebackState> *det_state = iter->second;
      det_state->AcceptIncomingArc(arc.dest_state, state_ptr, curr_arc,
                                   arc.weight);
    }
  }
  K2_CHECK(!label_to_state.empty() ||
           elements.begin()->second->state_id ==
               fsa.FinalState());  // I'm assuming the input
                                   // FSA is connected.
  return num_steps;
}

template <class TracebackState>
int32_t DetState<TracebackState>::ProcessArcs(
    const WfsaWithFbWeights &wfsa_in, double prune_cutoff,
    std::vector<Arc> *arcs_out,
    std::vector<std::vector<typename TracebackState::DerivType>>
        *derivs_per_arc,
    DetStateMap<TracebackState> *state_map,
    DetStatePriorityQueue<TracebackState> *queue) {
  NVTX_RANGE(K2_FUNC);
  const Fsa &fsa = wfsa_in.fsa;
  std::unordered_map<uint32_t, DetState<TracebackState> *> label_to_state;
  int32_t num_steps = GetDetStatesSuccessor(fsa, label_to_state);
  // The following loop normalizes successor det-states, outputs the arcs
  // that lead to them, and adds them to the queue if necessary.
  for (auto iter = label_to_state.begin(); iter != label_to_state.end();
       ++iter) {
    DetState<TracebackState> *det_state = iter->second;

    float arc_weight;
    std::vector<DerivType> deriv_info;
    det_state->Normalize(wfsa_in, &arc_weight, &deriv_info);
    if (det_state->forward_backward_prob >= prune_cutoff) {
      bool is_new_state = state_map->GetOutputState(det_state, fsa);
      arcs_out->push_back({this->state_id, det_state->state_id,
                           static_cast<int32_t>(iter->first), arc_weight});
      derivs_per_arc->push_back(std::move(deriv_info));
      if (is_new_state)
        queue->push(std::unique_ptr<DetState<TracebackState>>(det_state));
      else
        delete det_state;
    } else {
      delete det_state;
    }
  }
  return num_steps;
}

template <class TracebackState>
int32_t DetState<TracebackState>::ProcessArcs(
    const Fsa &fsa_in, std::vector<Arc> *arcs_out,
    std::vector<std::vector<typename TracebackState::DerivType>>
        *derivs_per_arc,
    DetStateMap<TracebackState> *state_map,
    DetStatePriorityQueue<TracebackState> *queue) {
  NVTX_RANGE(K2_FUNC);
  std::unordered_map<uint32_t, DetState<TracebackState> *> label_to_state;
  int32_t num_steps = GetDetStatesSuccessor(fsa_in, label_to_state);
  // The following loop normalizes successor det-states, outputs the arcs
  // that lead to them, and adds them to the queue if necessary.
  for (auto iter = label_to_state.begin(); iter != label_to_state.end();
       ++iter) {
    DetState<TracebackState> *det_state = iter->second;

    float arc_weight;
    std::vector<DerivType> deriv_info;
    det_state->Normalize(fsa_in, &arc_weight, &deriv_info);
    bool is_new_state = state_map->GetOutputState(det_state, fsa_in);
    arcs_out->push_back({this->state_id, det_state->state_id,
                         static_cast<int32_t>(iter->first), arc_weight});
    derivs_per_arc->push_back(std::move(deriv_info));
    if (is_new_state)
      queue->push(std::unique_ptr<DetState<TracebackState>>(det_state));
    else
      delete det_state;
  }
  return num_steps;
}

template <class TracebackState>
double LogSumOrMax(double, double);

template <>
inline double LogSumOrMax<MaxTracebackState>(double a, double b) {
  return std::max(a, b);
}
template <>
inline double LogSumOrMax<LogSumTracebackState>(double a, double b) {
  return LogAdd(a, b);
}

template <class TracebackState>
void DetState<TracebackState>::Normalize(const WfsaWithFbWeights &wfsa_in,
                                         float *removed_weight,
                                         std::vector<DerivType> *deriv_info) {
  NVTX_RANGE(K2_FUNC);
  std::unordered_set<TracebackState *> cur_states;

  double fb_prob = -std::numeric_limits<double>::infinity();
  for (const auto &p : elements) {
    TracebackState *state = p.second.get();
    fb_prob = LogSumOrMax<TracebackState>(
        fb_prob,
        state->forward_prob + wfsa_in.BackwardStateWeights()[state->state_id]);
    cur_states.insert(state);
  }

  int32_t new_seq_len = GetMostRecentCommonAncestor(&cur_states);
  // now cur_states.size() == 1.
  K2_CHECK_EQ(cur_states.size(), 1);
  K2_CHECK_LE(new_seq_len, seq_len);

  const TracebackState *base_state = *(cur_states.begin());
  // The following statement is a correction term that we add to
  // forward_backward_prob, in which we replace the forward_prob in the DetState
  // (which will have been computed in a path-dependent way) with the
  // forward_prob in wfsa_in.  Note: the values of state->forward_prob above can
  // be thought of as base_state->forward_prob plus some value that only depends
  // on the symbol sequence.  The point of this is to ensure that
  // this->forward_backward_prob (which is used for pruning) depends only on the
  // base_state and the symbol sequence, and not on "how we got here", i.e.  the
  // history of DetStates from which this one is derived via ProcessArcs().
  fb_prob += wfsa_in.ForwardStateWeights()[base_state->state_id] -
             base_state->forward_prob;
  // set thi->forward_backward_prob; it will affect pruning.
  this->forward_backward_prob = fb_prob;
  int32_t num_steps = seq_len - new_seq_len;
  this->seq_len = new_seq_len;

  // the following will set removed_weight and deriv_info.
  // `arcs` is needed to look up the weight.
  const Arc *arcs = wfsa_in.fsa.data;
  TraceBack(&cur_states, num_steps, arcs, removed_weight, deriv_info);

  normalized = true;
}

template <class TracebackState>
void DetState<TracebackState>::Normalize(const Fsa &fsa_in,
                                         float *removed_weight,
                                         std::vector<DerivType> *deriv_info) {
  NVTX_RANGE(K2_FUNC);
  std::unordered_set<TracebackState *> cur_states;
  for (const auto &p : elements) {
    TracebackState *state = p.second.get();
    cur_states.insert(state);
  }

  int32_t new_seq_len = GetMostRecentCommonAncestor(&cur_states);
  // now cur_states.size() == 1.
  K2_CHECK_EQ(cur_states.size(), 1);
  K2_CHECK_LE(new_seq_len, seq_len);

  const TracebackState *base_state = *(cur_states.begin());
  this->forward_backward_prob = 0;  // always set to 0 in un-pruned version
  int32_t num_steps = seq_len - new_seq_len;
  this->seq_len = new_seq_len;

  // the following will set removed_weight and deriv_info.
  // `arcs` is needed to look up the weight.
  const Arc *arcs = fsa_in.data;
  TraceBack(&cur_states, num_steps, arcs, removed_weight, deriv_info);

  normalized = true;
}

/*
  This class maps from determinized states (DetState) to integer state-ids
  in the determinized output.  Caution: it uses a randomized algorithm that
  could in principle produce collisions that would generate wrong output.
  We don't think this will ever happen though (128-bit representations).
 */
template <class TracebackState>
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
  bool GetOutputState(DetState<TracebackState> *a, const Fsa &fsa) {
    NVTX_RANGE(K2_FUNC);
    std::pair<uint64_t, uint64_t> compact;
    DetStateToCompact(*a, fsa, &compact);
    auto p = map_.insert({compact, cur_output_state_});
    bool inserted = p.second;
    if (inserted) {
      a->state_id = cur_output_state_++;
      return true;
    }

    a->state_id = p.first->second;
    return false;
  }

  int32_t size() const { return cur_output_state_; }

 private:
  // simple hashing function that just takes the first element of the pair.
  struct PairHasher {
    std::size_t operator()(const std::pair<uint64_t, uint64_t> &p) const {
      return static_cast<std::size_t>(p.first);
    }
  };

  int32_t cur_output_state_{0};
  /* maps from 128-bit key (stored as a pair of uint64_t's) to the int32_t
   * state-id. */
  std::unordered_map<std::pair<uint64_t, uint64_t>, uint32_t, PairHasher> map_;

  /* Turns DetState into a compact form of 128 bits.  Technically there
     could be collisions, which would be fatal for the algorithm, but this
     is one of those lifetime-of-the-universe type of things (kind of like
     the theoretical potential for git hash collision) that we ignore.

     The normalized form

  */
  void DetStateToCompact(const DetState<MaxTracebackState> &d, const Fsa &fsa,
                         std::pair<uint64_t, uint64_t> *vec) {
    NVTX_RANGE(K2_FUNC);
    assert(d.normalized);

    uint64_t a = 17489 * d.seq_len, b = d.seq_len;

    // We choose an arbitrary DetStateElement (the first one in the list) to
    // read the symbol sequence from; the symbol sequence will be the same no
    // matter which element we choose to trace back.
    auto elem = d.elements.begin()->second;
    int32_t seq_len = d.seq_len;
    const auto &arcs = fsa.data + fsa.indexes[0];
    for (int32_t i = 0; i < seq_len; ++i) {
      int32_t symbol = arcs[elem->arc_id].label;
      a = symbol + 102299 * a;
      b = symbol + 102983 * b;
      elem = elem->prev_state;
    }
    // This is `base_state`: the state from which we
    // start (and accept the specified symbol sequence).
    a = elem->state_id + 14051 * a;
    vec->first = a;
    vec->second = b;
  }

  void DetStateToCompact(const DetState<LogSumTracebackState> &d,
                         const Fsa &fsa, std::pair<uint64_t, uint64_t> *vec) {
    NVTX_RANGE(K2_FUNC);
    assert(d.normalized);

    uint64_t a = 17489 * d.seq_len, b = d.seq_len;

    // We choose an arbitrary DetStateElement (the first one in the list) to
    // read the symbol sequence from; the symbol sequence will be the same no
    // matter which element we choose to trace back.
    auto elem = d.elements.begin()->second;
    int32_t seq_len = d.seq_len;
    const auto &arcs = fsa.data + fsa.indexes[0];
    for (int32_t i = 0; i < seq_len; ++i) {
      int32_t symbol = arcs[elem->prev_elements[0].arc_index].label;
      a = symbol + 102299 * a;
      b = symbol + 102983 * b;
      elem = elem->prev_elements[0].prev_state;
    }
    // This is `base_state`: the state from which we
    // start (and accept the specified symbol sequence).
    a = elem->state_id + 14051 * a;
    vec->first = a;
    vec->second = b;
  }

  struct DetStateHasher {
    std::size_t operator()(const std::pair<uint64_t, uint64_t> &p) const {
      return p.first;
    }
  };
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_DETERMINIZE_IMPL_H_
