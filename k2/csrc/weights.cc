// k2/csrc/weights.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/weights.h"

#include <algorithm>
#include <queue>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace {
void CheckInput(const k2::Fsa &fsa, const float *arc_weights) {
  CHECK(IsValid(fsa));
  CHECK_NOTNULL(arc_weights);
}
}  // namespace

namespace k2 {

void ComputeForwardMaxWeights(const Fsa &fsa, const float *arc_weights,
                              double *state_weights) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  CHECK_NOTNULL(state_weights);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[0] = 0;
  for (int32_t i = 0; i != fsa.size2; ++i) {
    const auto &arc = arcs[i];
    DCHECK_GE(arc.dest_state, arc.src_state);
    auto src_weight = state_weights[arc.src_state];
    auto &dest_weight = state_weights[arc.dest_state];
    dest_weight = std::max(dest_weight, src_weight + arc_weights[i]);
  }
}

void ComputeBackwardMaxWeights(const Fsa &fsa, const float *arc_weights,
                               double *state_weights) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  CHECK_NOTNULL(state_weights);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[fsa.FinalState()] = 0;
  for (int32_t i = fsa.size2 - 1; i >= 0; --i) {
    const auto &arc = arcs[i];
    DCHECK_GE(arc.dest_state, arc.src_state);
    auto &src_weight = state_weights[arc.src_state];
    auto dest_weight = state_weights[arc.dest_state];
    src_weight = std::max(src_weight, dest_weight + arc_weights[i]);
  }
}

void ComputeForwardLogSumWeights(const Fsa &fsa, const float *arc_weights,
                                 double *state_weights) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  CHECK_NOTNULL(state_weights);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[0] = 0;
  for (int32_t i = 0; i != fsa.size2; ++i) {
    const auto &arc = arcs[i];
    DCHECK_GE(arc.dest_state, arc.src_state);
    auto src_weight = state_weights[arc.src_state];
    auto &dest_weight = state_weights[arc.dest_state];
    dest_weight = LogAdd(dest_weight, src_weight + arc_weights[i]);
  }
}

void ComputeBackwardLogSumWeights(const Fsa &fsa, const float *arc_weights,
                                  double *state_weights) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  CHECK_NOTNULL(state_weights);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[fsa.FinalState()] = 0;
  for (int32_t i = fsa.size2 - 1; i >= 0; --i) {
    const auto &arc = arcs[i];
    DCHECK_GE(arc.dest_state, arc.src_state);
    auto &src_weight = state_weights[arc.src_state];
    auto dest_weight = state_weights[arc.dest_state];
    src_weight = LogAdd(src_weight, dest_weight + arc_weights[i]);
  }
}

WfsaWithFbWeights::WfsaWithFbWeights(const Fsa &fsa, const float *arc_weights,
                                     FbWeightType t)
    : fsa(fsa), arc_weights(arc_weights), weight_type(t) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  ComputeForwardWeights();
  ComputeBackardWeights();
}

// Mohri, M. 2002. Semiring framework and algorithms for shortest-distance
// problems, Journal of Automata, Languages and Combinatorics 7(3): 321-350,
// 2002.
void WfsaWithFbWeights::ComputeForwardWeights() {
  auto num_states = fsa.NumStates();
  forward_state_weights = std::unique_ptr<double[]>(new double[num_states]);
  std::fill_n(forward_state_weights.get(), num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  forward_state_weights[0] = 0;
  if (weight_type == kMaxWeight) {
    for (auto i = 0; i != fsa.size2; ++i) {
      const auto &arc = arcs[i];
      DCHECK_GE(arc.dest_state, arc.src_state);
      auto src_weight = forward_state_weights[arc.src_state];
      auto &dest_weight = forward_state_weights[arc.dest_state];

      double r = src_weight + arc_weights[i];
      dest_weight = std::max(dest_weight, r);
    }
  } else if (weight_type == kLogSumWeight) {
    for (std::size_t i = 0; i != fsa.size2; ++i) {
      const auto &arc = arcs[i];
      DCHECK_GE(arc.dest_state, arc.src_state);
      auto src_weight = forward_state_weights[arc.src_state];
      auto &dest_weight = forward_state_weights[arc.dest_state];

      double r = src_weight + arc_weights[i];
      dest_weight = LogAdd(dest_weight, r);
    }
  } else {
    LOG(FATAL) << "Unreachable code is executed!";
  }
}

void WfsaWithFbWeights::ComputeBackardWeights() {
  auto num_states = fsa.NumStates();
  backward_state_weights = std::unique_ptr<double[]>(new double[num_states]);
  std::fill_n(backward_state_weights.get(), num_states,
              kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  backward_state_weights[fsa.FinalState()] = 0;
  if (weight_type == kMaxWeight) {
    for (auto i = fsa.size2 - 1; i >= 0; --i) {
      const auto &arc = arcs[i];
      DCHECK_GE(arc.dest_state, arc.src_state);
      auto &src_weight = backward_state_weights[arc.src_state];
      auto dest_weight = backward_state_weights[arc.dest_state];

      double r = dest_weight + arc_weights[i];
      src_weight = std::max(src_weight, r);
    }
  } else if (weight_type == kLogSumWeight) {
    for (auto i = fsa.size2 - 1; i >= 0; --i) {
      const auto &arc = arcs[i];
      DCHECK_GE(arc.dest_state, arc.src_state);
      auto &src_weight = backward_state_weights[arc.src_state];
      auto dest_weight = backward_state_weights[arc.dest_state];

      double r = dest_weight + arc_weights[i];
      src_weight = LogAdd(src_weight, r);
    }
  } else {
    LOG(FATAL) << "Unreachable code is executed!";
  }
}

}  // namespace k2
