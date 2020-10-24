/**
 * @brief
 * weights
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_WEIGHTS_H_
#define K2_CSRC_HOST_WEIGHTS_H_

#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"

namespace k2host {

constexpr float kFloatInfinity = std::numeric_limits<float>::infinity();
constexpr float kFloatNegativeInfinity =
    -std::numeric_limits<float>::infinity();
constexpr double kDoubleInfinity = std::numeric_limits<double>::infinity();
constexpr double kDoubleNegativeInfinity =
    -std::numeric_limits<double>::infinity();

/*
  This header contains various utilities that operate on weights and costs.
  We use the term "weights" when we have a positive-is-better sign,
  as in logprobs, and "costs" when we have a negative-is-better sign,
  as in negative logprobs.

  Note: weights are stored separately from the FSA.
 */

/*
  Does the 'forward' computation; this is as in the tropical semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.
  It's like shortest path but with the opposite sign.

   @param [in]  fsa  The fsa we are doing the forward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the start-state (if fsa is
                nonempty), and for later states will be the
                largest (most positive) weight from the start-state
                to that state along any path, or `kNegativeInfinity` if no such
                path exists.
   @param [out] arc_indexes  The arc indexes of the best path from the start
                state to the final state. It is empty if there is no such
                path.
 */
void ComputeForwardMaxWeights(const Fsa &fsa, double *state_weights,
                              std::vector<int32_t> *arc_indexes = nullptr);

/*
  Does the 'backward' computation; this is as in the tropical semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.
  It's like shortest path but with the opposite sign.

   @param [in]  fsa  The fsa we are doing the backward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the final-state (if fsa is
                nonempty), and for earlier states will be the
                largest (most positive) weight from that
                to the final state along any path, or `kNegativeInfinity` if no
  such path exists.
 */
void ComputeBackwardMaxWeights(const Fsa &fsa, double *state_weights);

/*
  Does the 'forward' computation; this is as in the log semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.

   @param [in]  fsa  The fsa we are doing the forward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the start-state (if fsa is
                nonempty), and for later states will be the
                log sum of all paths' weights from the start-state
                to that state, or `kNegativeInfinity` if no such
                path exists.
 */
void ComputeForwardLogSumWeights(const Fsa &fsa, double *state_weights);

/*
  Does the 'backward' computation; this is as in the log semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.

   @param [in]  fsa  The fsa we are doing the backward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the final-state (if fsa is
                nonempty), and for earlier states will be the
                log sum of all paths' weights from that state to the final
  state, or `kNegativeInfinity` if no such path exists.
 */
void ComputeBackwardLogSumWeights(const Fsa &fsa, double *state_weights);

enum FbWeightType { kMaxWeight, kLogSumWeight };

// Version of `ComputeForwardWeights` as a template interface, see documentation
// of `ComputeForwardMaxWeights` or `ComputeForwardLogSumWeights`
template <FbWeightType Type>
void ComputeForwardWeights(const Fsa &fsa, double *state_weights);

template <>
inline void ComputeForwardWeights<kMaxWeight>(const Fsa &fsa,
                                              double *state_weights) {
  ComputeForwardMaxWeights(fsa, state_weights);
}

template <>
inline void ComputeForwardWeights<kLogSumWeight>(const Fsa &fsa,
                                                 double *state_weights) {
  ComputeForwardLogSumWeights(fsa, state_weights);
}

// Version of `ComputeBackwardWeights` as a template interface, see
// documentation of `ComputeBackwardMaxWeights` or
// `ComputeBackwardLogSumWeights`
template <FbWeightType Type>
void ComputeBackwardWeights(const Fsa &fsa, double *state_weights);
template <>
inline void ComputeBackwardWeights<kMaxWeight>(const Fsa &fsa,
                                               double *state_weights) {
  ComputeBackwardMaxWeights(fsa, state_weights);
}

template <>
inline void ComputeBackwardWeights<kLogSumWeight>(const Fsa &fsa,
                                                  double *state_weights) {
  ComputeBackwardLogSumWeights(fsa, state_weights);
}
/*
  Returns the sum of the weights of all successful paths in an FSA, i.e., the
  shortest-distance from the initial state to the final states

   @param [in]  fsa  The fsa we'll get the shortest distance on.
                Must satisfy IsValid(fsa) (and IsTopSorted(fsa)?).
 */
template <FbWeightType Type>
double ShortestDistance(const Fsa &fsa) {
  if (IsEmpty(fsa)) return kDoubleNegativeInfinity;
  std::vector<double> state_weights(fsa.NumStates());
  ComputeForwardWeights<Type>(fsa, state_weights.data());
  return state_weights[fsa.FinalState()];
}

struct WfsaWithFbWeights {
  const Fsa &fsa;

  // Records whether we use max or log-sum.
  FbWeightType weight_type;

  /*
    Constructor.
       @param [in] fsa  Reference to an FSA; must satisfy
            IsValid(fsa) and IsTopSorted(fsa).
       @param [in]  t   Type of operation used to get forward
                        weights.  kMaxWeight == Viterbi, i.e. get
                        path with most positive weight sequence.
                        kLogSumWeight == Baum Welch, i.e. sum probs
                        over paths, treating weights as log-probs.
       @prams [out] forward_state_weights   Will be used to store
                        log-sum or max of weights along all paths
                        from the start state to each state in `fsa`.
                        We use `double` here because for long FSAs
                        round-off effects can cause nasty errors in
                        pruning. At entry it must be allocated with
                        size `fsa.NumStates()`. Note that the caller
                        should make sure the array is not freed
                        as long as object WfsaWithFbWeights exists.
       @prams [out] backward_state_weights   Will be used to store
                        log-sum or max of weights along all paths
                        from each state to the final state in `fsa`.
                        At entry it must be allocated with size
                        `fsa.NumStates()`. Note that the caller
                        should make sure the array is not freed
                        as long as object WfsaWithFbWeights exists.

   */
  WfsaWithFbWeights(const Fsa &fsa, FbWeightType t,
                    double *forward_state_weights,
                    double *backward_state_weights);

  const double *ForwardStateWeights() const { return forward_state_weights; }

  const double *BackwardStateWeights() const { return backward_state_weights; }

 private:
  double *forward_state_weights;
  double *backward_state_weights;
  void ComputeForwardWeights();
  void ComputeBackardWeights();
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_WEIGHTS_H_
