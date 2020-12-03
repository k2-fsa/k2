/**
 * @brief
 * rmepsilon
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_RMEPSILON_H_
#define K2_CSRC_HOST_RMEPSILON_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "k2/csrc/host/determinize_impl.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/weights.h"

namespace k2host {

/*
   Pruned version of `remove epsilon` which outputs an Fsa that is equivalent to
   the input but which has no epsilons. `TracebackState` could be either
   `MaxTracebackState` or `LogSumTracebackState`, search in `determinize_impl.h`
   for their definitions.

   The input FSA needs to have associated weights, because they will be used to
   choose the best path among alternative epsilon paths between states when the
   template parameter is MaxTracebackState, or do log-sum on weights along
   alternative epsilon paths when the template parameter is
   LogSumTracebackState.
 */
template <typename TracebackState>
class EpsilonsRemoverPruned {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input FSA, with weights and forward-backward
                         weights as required by this computation. For now
                         we assume that `fsa_in` is topologically sorted,
                         as required by the current constructor of
                         WfsaWithFbWeights.
                         `fsa_in.weight_type` must be kMaxWeight if
                         TracebackState is MaxTracebackState, but could be
                         kMaxWeight or kLogSumWeight for if TracebackState
                         is LogSumTracebackState (the difference will affect
                         pruning slightly).
    @param [in] beam     beam > 0 that affects pruning; this algorithm will
                         keep paths that are within `beam` of the best path.
                         Just make this very large if you don't want pruning.
  */
  EpsilonsRemoverPruned(const WfsaWithFbWeights &fsa_in, float beam)
      : fsa_in_(fsa_in), beam_(beam) {
    K2_CHECK_GT(beam, 0);
    if (std::is_same<TracebackState, MaxTracebackState>::value)
      K2_CHECK_EQ(fsa_in_.weight_type, kMaxWeight);
  }

  /*
    Do enough work to know how much memory will be needed, and output
    that information
        @param [out] fsa_size   The num-states and num-arcs of the output FSA
                                will be written to here
        @param [out] arc_derivs_size  The num-arcs of the output FSA and
                                      the number of arc-derivs elements (see
                                      `arc_derivs` definition in `GetOutput`
                                      below for details) will be written to
                                      here.
  */
  void GetSizes(Array2Size<int32_t> *fsa_size,
                Array2Size<int32_t> *arc_derivs_size);

  /*
    Finish the operation and output the epsilon-free FSA to `fsa_out` and
    arc derivative information to `arc_derivs`.
    @param [out] fsa_out   The output FSA; will be epsilon-free, and the states
                           will be in the same order that they were in `fsa_in`.
                           Must be initialized; search for 'initialized
                           definition' in class Array2 in array.h for meaning.
    @param [out] arc_derivs Indexed by arc in `fsa_out`, must be initialized.

                       When TracebackState is MaxTracebackState,
                       `arc_derivs.data[arc_derivs.indexes[i]]` through
                       `arc_derivs.data[arc_derivs.indexes[i+1] - 1]` is the
                        sequence of arcs in `fsa_in` that arc `i` in `fsa_out`
                        corresponds to; the weight of the arc in `fsa_out`
                        will equal the sum of those input arcs' weights.

                        When TracebackState is LogSumTracebackState,
                       `arc_derivs.data[arc_derivs.indexes[i]]` through
                       `arc_derivs.data[arc_derivs.indexes[i+1] - 1]` is a
                        list of pairs (input_arc, deriv) where 0 < deriv <= 1
                        is the derivative of arc_i's weight w.r.t. the weight
                        of `input_arc` in `fsa_in`. It could be interpreted as
                        a CSR-format matrix of dimension num_arcs_out by
                        num_arcs_in which gives the derivatives of output-arcs
                        weights w.r.t. input-arc weights.
   */
  void GetOutput(
      Fsa *fsa_out,
      Array2<typename TracebackState::DerivType *, int32_t> *arc_derivs);

 private:
  const WfsaWithFbWeights &fsa_in_;
  const float beam_;

  std::vector<int32_t> arc_indexes_;  // arc_index of fsa_out
  std::vector<Arc> arcs_;             // arcs of fsa_out
  std::vector<float> arc_weights_;    // arc_weights of fsa_out
  std::vector<std::vector<typename TracebackState::DerivType>> arc_derivs_;
};

using EpsilonsRemoverPrunedMax = EpsilonsRemoverPruned<MaxTracebackState>;
using EpsilonsRemoverPrunedLogSum = EpsilonsRemoverPruned<LogSumTracebackState>;

}  // namespace k2host

#endif  // K2_CSRC_HOST_RMEPSILON_H_
