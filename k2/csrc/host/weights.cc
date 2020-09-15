/**
 * @brief
 * weights
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/weights.h"

#include <algorithm>
#include <queue>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/util.h"


namespace k2host {

void ComputeForwardMaxWeights(const Fsa &fsa,
                              double *state_weights) {
  if (IsEmpty(fsa)) return;
  K2_DCHECK(IsValid(fsa));  // TODO(dan): make this run only in paranoid mode.
  K2_CHECK_NE(state_weights, nullptr);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[0] = 0;
  for (int32_t i = 0; i != fsa.size2; ++i) {
    const auto &arc = arcs[i];
    K2_DCHECK_GE(arc.dest_state, arc.src_state);
    auto src_weight = state_weights[arc.src_state];
    auto &dest_weight = state_weights[arc.dest_state];
    dest_weight = std::max(dest_weight, src_weight + arc.weight);
  }
}

void ComputeBackwardMaxWeights(const Fsa &fsa,
                               double *state_weights) {
  if (IsEmpty(fsa)) return;
  K2_CHECK_NE(state_weights, nullptr);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[fsa.FinalState()] = 0;
  for (int32_t i = fsa.size2 - 1; i >= 0; --i) {
    const auto &arc = arcs[i];
    K2_DCHECK_GE(arc.dest_state, arc.src_state);
    auto &src_weight = state_weights[arc.src_state];
    auto dest_weight = state_weights[arc.dest_state];
    src_weight = std::max(src_weight, dest_weight + arc.weight);
  }
}

void ComputeForwardLogSumWeights(const Fsa &fsa,
                                 double *state_weights) {
  if (IsEmpty(fsa)) return;
  K2_DCHECK(IsValid(fsa));  // TODO(dan): make this run only in paranoid mode.
  K2_CHECK_NE(state_weights, nullptr);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[0] = 0;
  for (int32_t i = 0; i != fsa.size2; ++i) {
    const auto &arc = arcs[i];
    K2_DCHECK_GE(arc.dest_state, arc.src_state);
    auto src_weight = state_weights[arc.src_state];
    auto &dest_weight = state_weights[arc.dest_state];
    dest_weight = LogAdd(dest_weight, src_weight + arc.weight);
  }
}

void ComputeBackwardLogSumWeights(const Fsa &fsa,
                                  double *state_weights) {
  if (IsEmpty(fsa)) return;
  K2_DCHECK(IsValid(fsa));  // TODO(dan): make this run only in paranoid mode.
  K2_CHECK_NE(state_weights, nullptr);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  state_weights[fsa.FinalState()] = 0;
  for (int32_t i = fsa.size2 - 1; i >= 0; --i) {
    const auto &arc = arcs[i];
    K2_DCHECK_GE(arc.dest_state, arc.src_state);
    auto &src_weight = state_weights[arc.src_state];
    auto dest_weight = state_weights[arc.dest_state];
    src_weight = LogAdd(src_weight, dest_weight + arc.weight);
  }
}

WfsaWithFbWeights::WfsaWithFbWeights(const Fsa &fsa,
                                     FbWeightType t,
                                     double *forward_state_weights,
                                     double *backward_state_weights)
    : fsa(fsa),
      weight_type(t),
      forward_state_weights(forward_state_weights),
      backward_state_weights(backward_state_weights) {
  if (IsEmpty(fsa)) return;
  K2_DCHECK(IsValid(fsa));
  ComputeForwardWeights();
  ComputeBackardWeights();
}

// Mohri, M. 2002. Semiring framework and algorithms for shortest-distance
// problems, Journal of Automata, Languages and Combinatorics 7(3): 321-350,
// 2002.
void WfsaWithFbWeights::ComputeForwardWeights() {
  auto num_states = fsa.NumStates();
  std::fill_n(forward_state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  forward_state_weights[0] = 0;
  if (weight_type == kMaxWeight) {
    for (auto i = 0; i != fsa.size2; ++i) {
      const auto &arc = arcs[i];
      K2_DCHECK_GE(arc.dest_state, arc.src_state);
      auto src_weight = forward_state_weights[arc.src_state];
      auto &dest_weight = forward_state_weights[arc.dest_state];

      double r = src_weight + arc.weight;
      dest_weight = std::max(dest_weight, r);
    }
  } else if (weight_type == kLogSumWeight) {
    for (std::size_t i = 0; i != fsa.size2; ++i) {
      const auto &arc = arcs[i];
      K2_DCHECK_GE(arc.dest_state, arc.src_state);
      auto src_weight = forward_state_weights[arc.src_state];
      auto &dest_weight = forward_state_weights[arc.dest_state];

      double r = src_weight + arc.weight;
      dest_weight = LogAdd(dest_weight, r);
    }
  } else {
    K2_LOG(FATAL) << "Unreachable code is executed!";
  }
}

void WfsaWithFbWeights::ComputeBackardWeights() {
  auto num_states = fsa.NumStates();
  std::fill_n(backward_state_weights, num_states, kDoubleNegativeInfinity);

  const auto &arcs = fsa.data + fsa.indexes[0];
  backward_state_weights[fsa.FinalState()] = 0;
  if (weight_type == kMaxWeight) {
    for (auto i = fsa.size2 - 1; i >= 0; --i) {
      const auto &arc = arcs[i];
      K2_DCHECK_GE(arc.dest_state, arc.src_state);
      auto &src_weight = backward_state_weights[arc.src_state];
      auto dest_weight = backward_state_weights[arc.dest_state];

      double r = dest_weight + arc.weight;
      src_weight = std::max(src_weight, r);
    }
  } else if (weight_type == kLogSumWeight) {
    for (auto i = fsa.size2 - 1; i >= 0; --i) {
      const auto &arc = arcs[i];
      K2_DCHECK_GE(arc.dest_state, arc.src_state);
      auto &src_weight = backward_state_weights[arc.src_state];
      auto dest_weight = backward_state_weights[arc.dest_state];

      double r = dest_weight + arc.weight;
      src_weight = LogAdd(src_weight, r);
    }
  } else {
    K2_LOG(FATAL) << "Unreachable code is executed!";
  }
}

}  // namespace k2host
