/**
 * @brief
 * determinize_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/determinize.h"

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
class DeterminizeTest : public ::testing::Test {
 protected:
  DeterminizeTest() {
    std::vector<Arc> arcs = {
        {0, 4, 1, 1},  {0, 1, 1, 1},  {1, 2, 2, 2}, {1, 3, 3, 3}, {2, 7, 1, 4},
        {3, 7, 1, 5},  {4, 6, 1, 2},  {4, 6, 1, 3}, {4, 5, 1, 3}, {4, 8, -1, 2},
        {5, 8, -1, 4}, {6, 8, -1, 3}, {7, 8, -1, 5}};
    fsa_creator_ = new FsaCreator(arcs, 8);
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

  ~DeterminizeTest() override {
    delete max_wfsa_;
    delete log_wfsa_;
    delete fsa_creator_;
  }

  WfsaWithFbWeights *max_wfsa_;
  WfsaWithFbWeights *log_wfsa_;
  FsaCreator *fsa_creator_;
  const Fsa *fsa_;
  int32_t num_states_;
  Fsa output_fsa;
  std::vector<double> logsum_forward_weights_;
  std::vector<double> logsum_backward_weights_;
  std::vector<double> max_forward_weights_;
  std::vector<double> max_backward_weights_;
};

TEST_F(DeterminizeTest, DeterminizePrunedMax) {
  float beam = 10.0;
  DeterminizerPrunedMax determinizer(*max_wfsa_, beam, 100);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  Array2Storage<typename MaxTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  determinizer.GetOutput(&fsa_out, &arc_derivs);

  EXPECT_TRUE(IsDeterministic(fsa_out));

  ASSERT_EQ(fsa_out.size1, 7);
  ASSERT_EQ(fsa_out.size2, 9);
  ASSERT_EQ(arc_derivs.size1, 9);
  ASSERT_EQ(arc_derivs.size2, 12);

  EXPECT_TRUE(IsRandEquivalent<kMaxWeight>(max_wfsa_->fsa, fsa_out, beam));
}

TEST_F(DeterminizeTest, DeterminizePrunedLogSum) {
  float beam = 10.0;
  DeterminizerPrunedLogSum determinizer(*log_wfsa_, beam, 100);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  Array2Storage<typename LogSumTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  determinizer.GetOutput(&fsa_out, &arc_derivs);

  EXPECT_TRUE(IsDeterministic(fsa_out));

  ASSERT_EQ(fsa_out.size1, 7);
  ASSERT_EQ(fsa_out.size2, 9);
  ASSERT_EQ(arc_derivs.size1, 9);
  ASSERT_EQ(arc_derivs.size2, 15);

  EXPECT_TRUE(IsRandEquivalent<kLogSumWeight>(log_wfsa_->fsa, fsa_out, beam));
  // TODO(haowen): how to check `arc_derivs_out` here, may return `num_steps` to
  // check the sum of `derivs_out` for each output arc?
}

TEST_F(DeterminizeTest, DeterminizeMax) {
  DeterminizerMax determinizer(*fsa_, 100);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  Array2Storage<typename MaxTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  determinizer.GetOutput(&fsa_out, &arc_derivs);

  EXPECT_TRUE(IsDeterministic(fsa_out));

  ASSERT_EQ(fsa_out.size1, 7);
  ASSERT_EQ(fsa_out.size2, 9);
  ASSERT_EQ(arc_derivs.size1, 9);
  ASSERT_EQ(arc_derivs.size2, 12);

  EXPECT_TRUE(IsRandEquivalent<kMaxWeight>(*fsa_, fsa_out));
}

TEST_F(DeterminizeTest, DeterminizeLogSum) {
  DeterminizerLogSum determinizer(*fsa_, 100);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  Array2Storage<typename LogSumTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  determinizer.GetOutput(&fsa_out, &arc_derivs);

  EXPECT_TRUE(IsDeterministic(fsa_out));

  ASSERT_EQ(fsa_out.size1, 7);
  ASSERT_EQ(fsa_out.size2, 9);
  ASSERT_EQ(arc_derivs.size1, 9);
  ASSERT_EQ(arc_derivs.size2, 15);

  EXPECT_TRUE(IsRandEquivalent<kLogSumWeight>(*fsa_, fsa_out));
}

}  // namespace k2host
