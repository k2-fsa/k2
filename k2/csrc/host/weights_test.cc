/**
 * @brief
 * weights_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/weights.h"

#include <algorithm>
#include <utility>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_renderer.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/util.h"

namespace k2host {

class WeightsTest : public ::testing::Test {
 protected:
  WeightsTest() {
    std::vector<Arc> arcs = {
        {0, 4, 1}, {0, 1, 1}, {1, 2, 1},  {1, 3, 1},  {2, 7, 1},  {3, 7, 1},
        {4, 6, 1}, {4, 8, 1}, {5, 9, -1}, {6, 9, -1}, {7, 9, -1}, {8, 9, -1},
    };
    fsa_creator_ = new FsaCreator(arcs, 9);
    fsa_ = &fsa_creator_->GetFsa();
    num_states_ = fsa_->NumStates();

    auto num_arcs = fsa_->size2;
    arc_weights_ = new float[num_arcs];
    std::vector<float> weights = {1, 1, 2, 3, 4, 5, 2, 3, 4, 3, 5, 6};
    std::copy_n(weights.begin(), num_arcs, arc_weights_);
  }

  ~WeightsTest() override {
    delete fsa_creator_;
    delete[] arc_weights_;
  }

  FsaCreator *fsa_creator_;
  const Fsa *fsa_;
  int32_t num_states_;
  float *arc_weights_;
  const std::vector<double> forward_max_weights_ = {
      0, 1, 3, 4, 1, kDoubleNegativeInfinity, 3, 9, 4, 14};
  const std::vector<double> backward_max_weights_ = {14, 13, 9, 10, 9,
                                                     4,  3,  5, 6,  0};
  const std::vector<double> forward_logsum_weights_ = {
      0, 1, 3, 4, 1, kDoubleNegativeInfinity, 3, 9.126928, 4, 14.143222};
  const std::vector<double> backward_logsum_weights_ = {
      14.143222, 13.126928, 9, 10, 9.018150, 4, 3, 5, 6, 0};
};

TEST_F(WeightsTest, ComputeForwardMaxWeights) {
  {
    std::vector<double> state_weights(num_states_);
    ComputeForwardMaxWeights(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, forward_max_weights_, 1e-3);
  }

  // template version
  {
    std::vector<double> state_weights(num_states_);
    ComputeForwardWeights<kMaxWeight>(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, forward_max_weights_, 1e-3);
  }
}

TEST_F(WeightsTest, ComputeBackwardMaxWeights) {
  {
    std::vector<double> state_weights(num_states_);
    ComputeBackwardMaxWeights(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, backward_max_weights_, 1e-3);
  }

  // template version
  {
    std::vector<double> state_weights(num_states_);
    ComputeBackwardWeights<kMaxWeight>(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, backward_max_weights_, 1e-3);
  }
}

TEST_F(WeightsTest, ComputeForwardLogSumWeights) {
  {
    std::vector<double> state_weights(num_states_);
    ComputeForwardLogSumWeights(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, forward_logsum_weights_, 1e-3);
  }

  // template version
  {
    std::vector<double> state_weights(num_states_);
    ComputeForwardWeights<kLogSumWeight>(*fsa_, arc_weights_,
                                         &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, forward_logsum_weights_, 1e-3);
  }
}

TEST_F(WeightsTest, ComputeBackwardLogSumWeights) {
  {
    std::vector<double> state_weights(num_states_);
    ComputeBackwardLogSumWeights(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, backward_logsum_weights_,
                                  1e-3);
  }

  // template version
  {
    std::vector<double> state_weights(num_states_);
    ComputeBackwardWeights<kLogSumWeight>(*fsa_, arc_weights_,
                                          &state_weights[0]);
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(state_weights, backward_logsum_weights_,
                                  1e-3);
  }
}

TEST_F(WeightsTest, ShortestDistance) {
  {
    double distance = ShortestDistance<kMaxWeight>(*fsa_, arc_weights_);
    EXPECT_NEAR(distance, 14, 1e-3);
  }

  {
    double distance = ShortestDistance<kLogSumWeight>(*fsa_, arc_weights_);
    EXPECT_NEAR(distance, 14.143222, 1e-3);
  }
}

TEST_F(WeightsTest, WfsaWithFbWeightsMax) {
  const auto num_states = fsa_->NumStates();
  std::vector<double> forward_weights(num_states);
  std::vector<double> backward_weights(num_states);
  WfsaWithFbWeights wfsa(*fsa_, arc_weights_, kMaxWeight,
                         forward_weights.data(), backward_weights.data());
  EXPECT_DOUBLE_ARRAY_APPROX_EQ(forward_weights, forward_max_weights_, 1e-3);
  EXPECT_DOUBLE_ARRAY_APPROX_EQ(backward_weights, backward_max_weights_, 1e-3);
}

TEST_F(WeightsTest, WfsaWithFbWeightsLogSum) {
  const auto num_states = fsa_->NumStates();
  std::vector<double> forward_weights(num_states);
  std::vector<double> backward_weights(num_states);
  WfsaWithFbWeights wfsa(*fsa_, arc_weights_, kLogSumWeight,
                         forward_weights.data(), backward_weights.data());
  EXPECT_DOUBLE_ARRAY_APPROX_EQ(forward_weights, forward_logsum_weights_, 1e-3);
  EXPECT_DOUBLE_ARRAY_APPROX_EQ(backward_weights, backward_logsum_weights_,
                                1e-3);
}

}  // namespace k2host
