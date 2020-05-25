// k2/csrc/weights_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/weights.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_renderer.h"
#include "k2/csrc/util.h"

namespace k2 {

class WeightsTest : public ::testing::Test {
 protected:
  WeightsTest() {
    std::vector<Arc> arcs = {
        {0, 4, 1}, {0, 1, 1}, {1, 2, 1},  {1, 3, 1},  {2, 7, 1},  {3, 7, 1},
        {4, 6, 1}, {4, 8, 1}, {5, 9, -1}, {6, 9, -1}, {7, 9, -1}, {8, 9, -1},
    };
    fsa_ = new Fsa(std::move(arcs), 9);
    num_states_ = fsa_->NumStates();

    auto num_arcs = fsa_->arcs.size();
    arc_weights_ = new float[num_arcs];
    std::vector<float> weights = {1, 1, 2, 3, 4, 5, 2, 3, 4, 3, 5, 6};
    std::copy_n(weights.begin(), num_arcs, arc_weights_);
  }

  ~WeightsTest() {
    delete fsa_;
    delete[] arc_weights_;
  }

  Fsa *fsa_;
  int32_t num_states_;
  float *arc_weights_;
};

TEST_F(WeightsTest, ComputeForwardMaxWeights) {
  {
    std::vector<float> state_weights(num_states_);
    ComputeForwardMaxWeights(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_THAT(state_weights,
                ::testing::ElementsAre(0, 1, 3, 4, 1, kFloatNegativeInfinity, 3,
                                       9, 4, 14));
  }
}

TEST_F(WeightsTest, ComputeBackwardMaxWeights) {
  {
    std::vector<float> state_weights(num_states_);
    ComputeBackwardMaxWeights(*fsa_, arc_weights_, &state_weights[0]);
    EXPECT_THAT(state_weights,
                ::testing::ElementsAre(14, 13, 9, 10, 9, 4, 3, 5, 6, 0));
  }
}

TEST_F(WeightsTest, WfsaWithFbWeightsMax) {
  {
    WfsaWithFbWeights wfsa(*fsa_, arc_weights_, kMaxWeight);
    std::vector<double> weights(num_states_);
    std::copy_n(wfsa.ForwardStateWeights(), num_states_, weights.begin());
    EXPECT_THAT(weights,
                ::testing::ElementsAre(0, 1, 3, 4, 1, kFloatNegativeInfinity, 3,
                                       9, 4, 14));
    std::copy_n(wfsa.BackwardStateWeights(), num_states_, weights.begin());
    EXPECT_THAT(weights,
                ::testing::ElementsAre(14, 13, 9, 10, 9, 4, 3, 5, 6, 0));
  }
}

TEST_F(WeightsTest, WfsaWithFbWeightsLogSum) {
  {
    WfsaWithFbWeights wfsa(*fsa_, arc_weights_, kLogSumWeight);
    std::vector<double> weights(num_states_);
    std::copy_n(wfsa.ForwardStateWeights(), num_states_, weights.begin());
    std::vector<double> forward_weights = {
        0, 1, 3, 4, 1, kDoubleNegativeInfinity, 3, 9.126928, 4, 14.143222};
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(weights, forward_weights, 1e-3);

    std::copy_n(wfsa.BackwardStateWeights(), num_states_, weights.begin());
    std::vector<double> backward_weights = {
        14.143222, 13.126928, 9, 10, 9.018150, 4, 3, 5, 6, 0};
    EXPECT_DOUBLE_ARRAY_APPROX_EQ(weights, backward_weights, 1e-3);
  }
}

}  // namespace k2
