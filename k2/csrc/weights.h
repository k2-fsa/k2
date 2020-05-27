// k2/csrc/weights.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_WEIGHTS_H_
#define K2_CSRC_WEIGHTS_H_

#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {

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
   @param [in]  arc_weights  Arc weights, indexed by arc in `fsa`.
                             Usually logprobs.
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the start-state (if fsa is
                nonempty), and for later states will be the
                largest (most positive) weight from the start-state
                to that state along any path, or `kNegativeInfinity` if no such
                path exists.
 */
void ComputeForwardMaxWeights(const Fsa &fsa, const float *arc_weights,
                              double *state_weights);

/*
  Does the 'backward' computation; this is as in the tropical semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.
  It's like shortest path but with the opposite sign.

   @param [in]  fsa  The fsa we are doing the backward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [in]  arc_weights  Arc weights, indexed by arc in `fsa`.
                             Usually logprobs.
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the final-state (if fsa is
                nonempty), and for earlier states will be the
                largest (most positive) weight from that
                to the final state along any path, or `kNegativeInfinity` if no
  such path exists.
 */
void ComputeBackwardMaxWeights(const Fsa &fsa, const float *arc_weights,
                               double *state_weights);

/*
  Does the 'forward' computation; this is as in the log semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.

   @param [in]  fsa  The fsa we are doing the forward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [in]  arc_weights  Arc weights, indexed by arc in `fsa`.
                             Usually logprobs.
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the start-state (if fsa is
                nonempty), and for later states will be the
                log sum of all paths' weights from the start-state
                to that state, or `kNegativeInfinity` if no such
                path exists.
 */
void ComputeForwardLogSumWeights(const Fsa &fsa, const float *arc_weights,
                                 double *state_weights);

/*
  Does the 'backward' computation; this is as in the log semiring
  but with the opposite sign, as in logprobs rather than negative logprobs.

   @param [in]  fsa  The fsa we are doing the backward computation on.
                Must satisfy IsValid(fsa) and IsTopSorted(fsa).
   @param [in]  arc_weights  Arc weights, indexed by arc in `fsa`.
                             Usually logprobs.
   @param [out] state_weights  The per-state weights will be written to here.
                They will be 0 for the final-state (if fsa is
                nonempty), and for earlier states will be the
                log sum of all paths' weights from that state to the final
  state, or `kNegativeInfinity` if no such path exists.
 */
void ComputeBackwardLogSumWeights(const Fsa &fsa, const float *arc_weights,
                                  double *state_weights);

enum FbWeightType { kMaxWeight, kLogSumWeight };

// Version of `ComputeForwardWeights` as a template interface, see documentation
// of `ComputeForwardMaxWeights` or `ComputeForwardLogSumWeights`
template <FbWeightType Type>
void ComputeForwardWeights(const Fsa &fsa, const float *arc_weights,
                           double *state_weights);

template <>
inline void ComputeForwardWeights<kMaxWeight>(const Fsa &fsa,
                                              const float *arc_weights,
                                              double *state_weights) {
  ComputeForwardMaxWeights(fsa, arc_weights, state_weights);
}

template <>
inline void ComputeForwardWeights<kLogSumWeight>(const Fsa &fsa,
                                                 const float *arc_weights,
                                                 double *state_weights) {
  ComputeForwardLogSumWeights(fsa, arc_weights, state_weights);
}

// Version of `ComputeBackwardWeights` as a template interface, see
// documentation of `ComputeBackwardMaxWeights` or
// `ComputeBackwardLogSumWeights`
template <FbWeightType Type>
void ComputeBackwardWeights(const Fsa &fsa, const float *arc_weights,
                            double *state_weights);
template <>
inline void ComputeBackwardWeights<kMaxWeight>(const Fsa &fsa,
                                               const float *arc_weights,
                                               double *state_weights) {
  ComputeBackwardMaxWeights(fsa, arc_weights, state_weights);
}

template <>
inline void ComputeBackwardWeights<kLogSumWeight>(const Fsa &fsa,
                                                  const float *arc_weights,
                                                  double *state_weights) {
  ComputeBackwardLogSumWeights(fsa, arc_weights, state_weights);
}
/*
  Returns the sum of the weights of all successful paths in an FSA, i.e., the
  shortest-distance from the initial state to the final states

   @param [in]  fsa  The fsa we'll get the shortest distance on.
                Must satisfy IsValid(fsa) (and IsTopSorted(fsa)?).
   @param [in]  arc_weights  Arc weights, indexed by arc in `fsa`.
                             Usually logprobs.
 */
template <FbWeightType Type>
double ShortestDistance(const Fsa &fsa, const float *arc_weights) {
  if (IsEmpty(fsa)) return kDoubleNegativeInfinity;
  std::vector<double> state_weights(fsa.NumStates());
  ComputeForwardWeights<Type>(fsa, arc_weights, state_weights.data());
  return state_weights[fsa.FinalState()];
}

struct WfsaWithFbWeights {
  const Fsa &fsa;
  const float *arc_weights;

  // Records whether we use max or log-sum.
  FbWeightType weight_type;

  /*
    Constructor.
       @param [in] fsa  Reference to an FSA; must satisfy
            IsValid(fsa) and IsTopSorted(fsa).
       @param [in]  arc_weights  Arc weights, indexed by arc in `fsa`.
                                 Usually logprobs.
       @param [in]  t   Type of operation used to get forward
                        weights.  kMaxWeight == Viterbi, i.e. get
                        path with most positive weight sequence.
                        kLogSumWeight == Baum Welch, i.e. sum probs
                        over paths, treating weights as log-probs.
   */
  WfsaWithFbWeights(const Fsa &fsa, const float *arc_weights, FbWeightType t);

  const double *ForwardStateWeights() const {
    return forward_state_weights.get();
  }

  const double *BackwardStateWeights() const {
    return backward_state_weights.get();
  }

 private:
  // forward_state_weights are the log-sum or max of weights along all paths
  // from the start-state to each state.  We use double because for long FSAs
  // roundoff effects can cause nasty errors in pruning.
  std::unique_ptr<double[]> forward_state_weights;
  // backward_state_weights are the log-sum or max of weights along all paths
  // from each state to the final state.
  std::unique_ptr<double[]> backward_state_weights;
  void ComputeForwardWeights();
  void ComputeBackardWeights();
};

}  // namespace k2

#endif  // K2_CSRC_WEIGHTS_H_
