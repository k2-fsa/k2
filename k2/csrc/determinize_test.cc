// k2/csrc/determinize_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/determinize.h"

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
class DeterminizeTest : public ::testing::Test {
 protected:
  DeterminizeTest() {
    std::vector<Arc> arcs = {{0, 4, 1}, {0, 1, 1},  {1, 2, 2},  {1, 3, 3},
                             {2, 7, 1}, {3, 7, 1},  {4, 6, 1},  {4, 6, 1},
                             {4, 5, 1}, {4, 8, -1}, {5, 8, -1}, {6, 8, -1},
                             {7, 8, -1}};
    fsa_creator_ = new FsaCreator(arcs, 8);
    fsa_ = &fsa_creator_->GetFsa();
    num_states_ = fsa_->NumStates();

    auto num_arcs = fsa_->size2;
    arc_weights_ = new float[num_arcs];
    std::vector<float> weights = {1, 1, 2, 3, 4, 5, 2, 3, 3, 2, 4, 3, 5};
    std::copy_n(weights.begin(), num_arcs, arc_weights_);

    max_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kMaxWeight);
    log_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kLogSumWeight);
  }

  ~DeterminizeTest() override {
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
  Fsa output_fsa;
};

TEST_F(DeterminizeTest, DeterminizePrunedMax) {
  float beam = 10.0;
  DeterminizerMax determinizer(*max_wfsa_, beam, 100);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  std::vector<float> arc_weights_out(fsa_size.size2);
  Array2Storage<typename MaxTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  determinizer.GetOutput(&fsa_out, arc_weights_out.data(), &arc_derivs);

  EXPECT_TRUE(IsDeterministic(fsa_out));

  ASSERT_EQ(fsa_out.size1, 7);
  ASSERT_EQ(fsa_out.size2, 9);
  ASSERT_EQ(arc_derivs.size1, 9);
  ASSERT_EQ(arc_derivs.size2, 12);

  EXPECT_TRUE(IsRandEquivalent<kMaxWeight>(max_wfsa_->fsa,
                                           max_wfsa_->arc_weights, fsa_out,
                                           arc_weights_out.data(), beam));
}

TEST_F(DeterminizeTest, DeterminizePrunedLogSum) {
  float beam = 10.0;
  DeterminizerLogSum determinizer(*log_wfsa_, beam, 100);
  Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);

  FsaCreator fsa_creator(fsa_size);
  auto &fsa_out = fsa_creator.GetFsa();
  std::vector<float> arc_weights_out(fsa_size.size2);
  Array2Storage<typename LogSumTracebackState::DerivType *, int32_t>
      derivs_storage(arc_derivs_size, 1);
  auto &arc_derivs = derivs_storage.GetArray2();

  determinizer.GetOutput(&fsa_out, arc_weights_out.data(), &arc_derivs);

  EXPECT_TRUE(IsDeterministic(fsa_out));

  ASSERT_EQ(fsa_out.size1, 7);
  ASSERT_EQ(fsa_out.size2, 9);
  ASSERT_EQ(arc_derivs.size1, 9);
  ASSERT_EQ(arc_derivs.size2, 15);

  EXPECT_TRUE(IsRandEquivalent<kLogSumWeight>(log_wfsa_->fsa,
                                              log_wfsa_->arc_weights, fsa_out,
                                              arc_weights_out.data(), beam));
  // TODO(haowen): how to check `arc_derivs_out` here, may return `num_steps` to
  // check the sum of `derivs_out` for each output arc?
}

}  // namespace k2
