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

namespace {
void CheckInput(const k2::Fsa &fsa, const float *arc_weights) {
  CHECK(IsValid(fsa));
  CHECK_NOTNULL(arc_weights);
}
}  // namespace

namespace k2 {

void ComputeForwardMaxWeights(const Fsa &fsa, const float *arc_weights,
                              float *state_weights) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  CHECK_NOTNULL(state_weights);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kNegativeInfinity);

  const auto &arcs = fsa.arcs;
  state_weights[0] = 0;
  for (std::size_t i = 0; i != arcs.size(); ++i) {
    const auto &arc = arcs[i];
    DCHECK_GE(arc.dest_state, arc.src_state);
    auto src_weight = state_weights[arc.src_state];
    auto &dest_weight = state_weights[arc.dest_state];
    dest_weight = std::max(dest_weight, src_weight + arc_weights[i]);
  }
}

void ComputeBackwardMaxWeights(const Fsa &fsa, const float *arc_weights,
                               float *state_weights) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);
  CHECK_NOTNULL(state_weights);

  int32_t num_states = fsa.NumStates();
  std::fill_n(state_weights, num_states, kNegativeInfinity);

  const auto &arcs = fsa.arcs;
  state_weights[num_states - 1] = 0;
  for (auto i = static_cast<int32_t>(arcs.size()) - 1; i >= 0; --i) {
    const auto &arc = arcs[i];
    DCHECK_GE(arc.dest_state, arc.src_state);
    auto &src_weight = state_weights[arc.src_state];
    auto dest_weight = state_weights[arc.dest_state];
    src_weight = std::max(src_weight, dest_weight + arc_weights[i]);
  }
}

WfsaWithFbWeights::WfsaWithFbWeights(const Fsa &fsa, const float *arc_weights,
                                     FbWeightType t)
    : fsa(fsa), arc_weights(arc_weights), weight_type(t) {
  if (IsEmpty(fsa)) return;
  CheckInput(fsa, arc_weights);

  auto num_states = fsa.NumStates();
  forward_state_weights = std::unique_ptr<double[]>(new double[num_states]());
  backward_state_weights = std::unique_ptr<double[]>(new double[num_states]());
  // TODO(haowen): compute `forward/backward_state_weights`, also check
  // `IsTopSorted(fsa)`
}
}  // namespace k2
