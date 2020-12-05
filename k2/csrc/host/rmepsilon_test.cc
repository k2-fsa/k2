/**
 * @brief
 * rmepsilon_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/rmepsilon.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_equivalent.h"
#include "k2/csrc/host/fsa_renderer.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"

namespace k2host {
class RmEpsilonTest : public ::testing::Test {
 protected:
  RmEpsilonTest() {
    std::vector<Arc> arcs = {
        {0, 4, 1, 1},        {0, 1, 1, 1},        {1, 2, kEpsilon, 2},
        {1, 3, kEpsilon, 3}, {1, 4, kEpsilon, 2}, {2, 7, kEpsilon, 4},
        {3, 7, kEpsilon, 5}, {4, 6, 1, 2},        {4, 6, kEpsilon, 3},
        {4, 8, 1, 3},        {4, 9, -1, 2},       {5, 9, -1, 4},
        {6, 9, -1, 3},       {7, 9, -1, 5},       {8, 9, -1, 6},
    };
    fsa_creator_ = new FsaCreator(arcs, 9);
    fsa_ = &fsa_creator_->GetFsa();
    num_states_ = fsa_->NumStates();

    max_forward_weights_.resize(num_states_);
    max_backward_weights_.resize(num_states_);
    logsum_forward_weights_.resize(num_states_);
    logsum_backward_weights_.resize(num_states_);
    max_wfsa_ =
        new WfsaWithFbWeights(*fsa_, kMaxWeight, max_forward_weights_.data(),
                              max_backward_weights_.data());
    log_wfsa_ = new WfsaWithFbWeights(*fsa_, kLogSumWeight,
                                      logsum_forward_weights_.data(),
                                      logsum_backward_weights_.data());
  }

  ~RmEpsilonTest() override {
    delete max_wfsa_;
    delete log_wfsa_;
    delete fsa_creator_;
  }

  WfsaWithFbWeights *max_wfsa_;
  WfsaWithFbWeights *log_wfsa_;
  FsaCreator *fsa_creator_;
  const Fsa *fsa_;
  int32_t num_states_;
  std::vector<double> logsum_forward_weights_;
  std::vector<double> logsum_backward_weights_;
  std::vector<double> max_forward_weights_;
  std::vector<double> max_backward_weights_;
};

TEST_F(RmEpsilonTest, RmEpsilonsPrunedMax) {
  float beam = 8.0;
  EpsilonsRemoverPrunedMax eps_remover(*max_wfsa_, beam);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  eps_remover.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  Array2Storage<typename MaxTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  eps_remover.GetOutput(&fsa_out, &arc_derivs);

  EXPECT_TRUE(IsEpsilonFree(fsa_out));

  ASSERT_EQ(fsa_out.size1, 6);
  ASSERT_EQ(fsa_out.size2, 11);
  ASSERT_EQ(arc_derivs.size1, 11);
  ASSERT_EQ(arc_derivs.size2, 18);

  EXPECT_TRUE(IsRandEquivalent<kMaxWeight>(max_wfsa_->fsa, fsa_out, beam));
}

TEST_F(RmEpsilonTest, RmEpsilonsPrunedLogSum) {
  float beam = 8.0;
  EpsilonsRemoverPrunedLogSum eps_remover(*log_wfsa_, beam);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  eps_remover.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();

  Array2Storage<typename LogSumTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  eps_remover.GetOutput(&fsa_out, &arc_derivs);

  EXPECT_TRUE(IsEpsilonFree(fsa_out));

  ASSERT_EQ(fsa_out.size1, 6);
  ASSERT_EQ(fsa_out.size2, 11);
  ASSERT_EQ(arc_derivs.size1, 11);
  ASSERT_EQ(arc_derivs.size2, 20);

  K2_LOG(INFO) << "log_wfsa_->fsa is: " << log_wfsa_->fsa
               << ", fsa_out is: " << fsa_out;
  // TODO(haowen): uncomment this after re-implementing
  // IsRandEquivalentAfterRmEpsPrunedLogSum
  // EXPECT_TRUE(
  //    IsRandEquivalentAfterRmEpsPrunedLogSum(log_wfsa_->fsa, fsa_out, beam));

  // TODO(haowen): how to check arc_derivs
}

}  // namespace k2host
