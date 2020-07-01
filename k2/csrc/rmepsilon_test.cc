// k2/csrc/rmepsilon_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/rmepsilon.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_equivalent.h"
#include "k2/csrc/fsa_renderer.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {
class RmEpsilonTest : public ::testing::Test {
 protected:
  RmEpsilonTest() {
    std::vector<Arc> arcs = {
        {0, 4, 1},        {0, 1, 1},        {1, 2, kEpsilon}, {1, 3, kEpsilon},
        {1, 4, kEpsilon}, {2, 7, kEpsilon}, {3, 7, kEpsilon}, {4, 6, 1},
        {4, 6, kEpsilon}, {4, 8, 1},        {4, 9, -1},       {5, 9, -1},
        {6, 9, -1},       {7, 9, -1},       {8, 9, -1},
    };
    fsa_creator_ = new FsaCreator(arcs, 9);
    fsa_ = &fsa_creator_->GetFsa();
    num_states_ = fsa_->NumStates();

    auto num_arcs = fsa_->size2;
    arc_weights_ = new float[num_arcs];
    std::vector<float> weights = {1, 1, 2, 3, 2, 4, 5, 2, 3, 3, 2, 4, 3, 5, 6};
    std::copy_n(weights.begin(), num_arcs, arc_weights_);

    max_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kMaxWeight);
    log_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kLogSumWeight);
  }

  ~RmEpsilonTest() override {
    delete[] arc_weights_;
    delete max_wfsa_;
    delete log_wfsa_;
    delete fsa_creator_;
  }

  WfsaWithFbWeights *max_wfsa_;
  WfsaWithFbWeights *log_wfsa_;
  FsaCreator *fsa_creator_;
  const Fsa *fsa_;
  int32_t num_states_;
  float *arc_weights_;
};

TEST_F(RmEpsilonTest, RmEpsilonsPrunedMax) {
  float beam = 8.0;
  EpsilonsRemoverMax eps_remover(*max_wfsa_, beam);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  eps_remover.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  std::vector<float> arc_weights_out(fsa_size.size2);
  Array2Storage<typename MaxTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  eps_remover.GetOutput(&fsa_out, arc_weights_out.data(), &arc_derivs);

  EXPECT_TRUE(IsEpsilonFree(fsa_out));

  ASSERT_EQ(fsa_out.size1, 6);
  ASSERT_EQ(fsa_out.size2, 11);
  ASSERT_EQ(arc_derivs.size1, 11);
  ASSERT_EQ(arc_derivs.size2, 18);

  EXPECT_TRUE(IsRandEquivalent<kMaxWeight>(max_wfsa_->fsa,
                                           max_wfsa_->arc_weights, fsa_out,
                                           arc_weights_out.data(), beam));
}

TEST_F(RmEpsilonTest, RmEpsilonsPrunedLogSum) {
  float beam = 8.0;
  EpsilonsRemoverLogSum eps_remover(*log_wfsa_, beam);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  eps_remover.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  std::vector<float> arc_weights_out(fsa_size.size2);
  Array2Storage<typename LogSumTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  eps_remover.GetOutput(&fsa_out, arc_weights_out.data(), &arc_derivs);

  EXPECT_TRUE(IsEpsilonFree(fsa_out));

  ASSERT_EQ(fsa_out.size1, 6);
  ASSERT_EQ(fsa_out.size2, 11);
  ASSERT_EQ(arc_derivs.size1, 11);
  ASSERT_EQ(arc_derivs.size2, 20);

  EXPECT_TRUE(IsRandEquivalentAfterRmEpsPrunedLogSum(
      log_wfsa_->fsa, log_wfsa_->arc_weights, fsa_out, arc_weights_out.data(),
      beam));

  // TODO(haowen): how to check arc_derivs
}

}  // namespace k2
