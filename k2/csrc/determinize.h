// k2/csrc/determinize.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_DETERMINIZE_H_
#define K2_CSRC_DETERMINIZE_H_

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

#include "glog/logging.h"
#include "k2/csrc/determinize_impl.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/util.h"
#include "k2/csrc/weights.h"

namespace k2 {

/*
   Pruned determinization with log-sum (equivalent to log semiring) or max
   (equivalent to tropical semiring) on weights (interpret them as log-probs).

   `TracebackState` could be either `MaxTracebackState` or
   `LogSumTracebackState`, search in `determinize_impl.h` for their definitions.
 */
template <typename TracebackState>
class Determinizer {
 public:
  /* Lightweight constructor that just keeps const references to the input
     parameters.
     @param [in] fsa_in  The input FSA to be determinized.  Expected to be
                         epsilon free, but this is not checked; in any case,
                         epsilon will be treated as a normal symbol.
                         Forward-backward weights must be provided for pruning
                         purposes; fsa_in.weight_type must be kMaxWeight
                         (kLogSumWeight) if TracebackState is
                         MaxTracebackState (LogSumTracebackState).
    @param [in] beam     beam > 0 that affects pruning; this algorithm will
                         keep paths that are within `beam` of the best path.
                         Just make this very large if you don't want pruning.
    @param [in] max_step Maximum number of computation steps before we return
                         (or if <= 0, there is no limit); provided so users can
                         limit the time taken in pathological cases.
  */
  Determinizer(const WfsaWithFbWeights &fsa_in, float beam, int64_t max_step)
      : fsa_in_(fsa_in), beam_(beam), max_step_(max_step) {
    CHECK_GT(beam, 0);
    if (std::is_same<TracebackState, MaxTracebackState>::value)
      CHECK_EQ(fsa_in_.weight_type, kMaxWeight);
    else if (std::is_same<TracebackState, LogSumTracebackState>::value)
      CHECK_EQ(fsa_in_.weight_type, kLogSumWeight);
    else
      LOG(FATAL) << "Unreachable code is executed!";
  }

  /*
    Do enough work that know now much memory will be needed, and output
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
    Finish the operation and output the determinized FSA to `fsa_out` and
    arc derivative information to `arc_derivs`.
    @param [out] fsa_out   The output FSA; will be deterministic. For a symbol
                           sequence S accepted by fsa_in, the total (best)
                           weight of S in fsa_in should equal the total (best)
                           weight of S in fsa_out (as discoverable by
                           composition then finding the total weight of the
                           result), except as affected by pruning of course,
                           where `total (best)` means we take `log-sum` if
                           `fsa_in.weight_type` is `kLogSumWeight` or `max`
                           if `fsa_in.weight_type` is `kMaxWeight`.
                           Must be initialized; search for 'initialized
                           definition' in class Array2 in array.h for meaning.
    @param [out] arc_weights_out   Weights per arc of output FSA;
                                   Must be allocated with size num-arcs of
                                  `fsa_out` at entry. Note: for max version
                                   (MaxTracebackState), these can be computed
                                   from arc_derivs and the weights of fsa_in,
                                   so this output is not strictly necessary;
                                   it's provided mostly due to sharing the
                                   internal code with the log-sum version.
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
                        weights w.r.t. input-arc weights. Note: the deriv values
                        may actually be zero if the pruning beam is very large,
                        due to limited floating point range.

    @return   Returns the effective pruning beam, a value >= 0 which is the
              difference between the total weight of the output FSA and the cost
              of the last arc expanded.
   */
  float GetOutput(
      Fsa *fsa_out, float *arc_weights_out,
      Array2<typename TracebackState::DerivType *, int32_t> *arc_derivs);

 private:
  const WfsaWithFbWeights &fsa_in_;
  const float beam_;
  int64_t max_step_;

  float effective_beam_;
  std::vector<Arc> arcs_;           // arcs of fsa_out
  std::vector<float> arc_weights_;  // arc_weights of fsa_out
  std::vector<std::vector<typename TracebackState::DerivType>> arc_derivs_;
};

using DeterminizerMax = Determinizer<MaxTracebackState>;
using DeterminizerLogSum = Determinizer<LogSumTracebackState>;

}  // namespace k2

#endif  // K2_CSRC_DETERMINIZE_H_
