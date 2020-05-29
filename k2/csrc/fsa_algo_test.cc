// k2/csrc/fsa_algo_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa_equivalent.h"
#include "k2/csrc/fsa_renderer.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

namespace k2 {

TEST(FsaAlgo, ConnectCore) {
  {
    // case 1: an empty input fsa
    Fsa a;
    std::vector<int32_t> state_b_to_a(10);
    bool status = ConnectCore(a, &state_b_to_a);
    EXPECT_TRUE(state_b_to_a.empty());
    EXPECT_TRUE(status);
  }
  {
    // case 2: a connected, acyclic FSA
    std::string s = R"(
      0 1 1
      1 2 2
      1 3 3
      2 4 4
      3 4 4
      4
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);
    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(*a, &state_b_to_a);
    ASSERT_EQ(state_b_to_a.size(), 5u);
    // notice that state_b_to_a maps:
    //   2 -> 3
    //   3 -> 2
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 1, 3, 2, 4));
    EXPECT_TRUE(status);
  }
  {
    // case 3: a connected, cyclic FSA
    // the cycle is a self-loop, the output is still topsorted.
    std::string s = R"(
      0 1 1
      1 2 2
      1 3 3
      2 2 2
      2 4 4
      3 4 4
      4
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);
    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(*a, &state_b_to_a);
    ASSERT_EQ(state_b_to_a.size(), 5u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 1, 3, 2, 4));
    EXPECT_TRUE(status);
  }
  {
    // case 4: a non-connected, acyclic, non-topsorted FSA
    std::string s = R"(
      1 0 1
      4 2 2
      3 5 5
      4 3 3
      0 4 4
      0 3 3
      5
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(*a, &state_b_to_a);
    EXPECT_TRUE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    /*                                               0  1  2  3 */
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 4, 3, 5));  // topsorted
  }

  {
    // case 5: a non-connected, cyclic, non-topsorted FSA
    // the output fsa will contain a cycle
    std::string s = R"(
      1 0 1
      4 2 2
      3 5 5
      4 3 3
      0 4 4
      0 3 3
      3 0 3
      5
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(*a, &state_b_to_a);
    EXPECT_FALSE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 3, 4, 5));
  }
  {
    // case 6 (another one): a non-connected, cyclic, non-topsorted FSA;
    // the cycle is removed since state 2 is not co-accessible
    std::string s = R"(
      1 0 1
      4 2 2
      3 5 5
      4 3 3
      0 4 4
      0 3 3
      2 2 2
      5
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a;
    bool status = ConnectCore(*a, &state_b_to_a);
    EXPECT_TRUE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 4, 3, 5));
  }
}
TEST(FsaAlgo, Connect) {
  {
    // case 1: a non-connected, non-topsorted, acyclic input fsa;
    // the output fsa is topsorted.
    std::string s = R"(
      0 1 1
      0 2 2
      1 3 3
      1 6 6
      2 4 2
      2 6 3
      5 0 1
      2 1 1
      6
    )";

    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    std::vector<int32_t> state_b_to_a(10);  // an arbitrary number
    bool status = ConnectCore(*a, &state_b_to_a);
    EXPECT_TRUE(status);

    ASSERT_EQ(state_b_to_a.size(), 4u);
    EXPECT_THAT(state_b_to_a, ::testing::ElementsAre(0, 2, 1, 6));

    Fsa b;
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    status = Connect(*a, &b, &arc_map);
    EXPECT_TRUE(IsTopSorted(b));
    EXPECT_TRUE(status);

    ASSERT_EQ(b.NumStates(), 4u);  // state 3,4,5 from `a` are removed
    EXPECT_THAT(b.arc_indexes, ::testing::ElementsAre(0, 2, 4, 5, 5));

    std::vector<Arc> target_arcs = {
        {0, 2, 1}, {0, 1, 2}, {1, 3, 3}, {1, 2, 1}, {2, 3, 6},
    };
    for (auto i = 0; i != target_arcs.size(); ++i)
      EXPECT_EQ(b.arcs[i], target_arcs[i]);

    ASSERT_EQ(arc_map.size(), 5u);
    EXPECT_THAT(arc_map, ::testing::ElementsAre(0, 1, 5, 6, 3));
  }

  {
    // A non-empty fsa that after trimming, it returns an empty fsa.
    std::string s = R"(
      0 1 1
      0 2 2
      1 3 3
      1 6 6
      2 4 2
      2 6 3
      5 0 1
      5 7 2
      7
    )";

    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    Fsa b;
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    bool status = Connect(*a, &b, &arc_map);
    EXPECT_TRUE(IsEmpty(b));
    EXPECT_TRUE(status);
    EXPECT_TRUE(arc_map.empty());
  }
  {
    // a cyclic input fsa
    // after trimming, the cycle is removed;
    // so the output fsa should be topsorted.
    std::string s = R"(
      0 3 3
      0 5 5
      3 5 5
      3 2 2
      3 4 4
      2 1 1
      1 2 2
      3 6 6
      4 5 5
      4 6 6
      5 6 6
      6
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    Fsa b;
    Connect(*a, &b);
    EXPECT_TRUE(IsTopSorted(b));
  }

  {
    // a cyclic input fsa
    // after trimming, the cycle remains (it is not a self-loop);
    // so the output fsa is NOT topsorted.
    std::string s = R"(
      1 0 1
      0 3 3
      0 2 2
      3 2 2
      3 5 5
      5 3 3
      5 4 4
      4 4 4
      2 6 6
      6
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    Fsa b;
    bool status = Connect(*a, &b);
    EXPECT_FALSE(IsTopSorted(b));
    EXPECT_FALSE(status);
  }

  {
    // a cyclic input fsa
    // after trimming, the cycle remains (it is not a self-loop);
    // so the output fsa is NOT topsorted.
    std::string s = R"(
      0 1 1
      0 2 2
      2 2 2
      2 1 1
      1 1 1
      1 3 3
      2 3 3
      3
    )";
    auto a = StringToFsa(s);
    EXPECT_NE(a.get(), nullptr);

    Fsa b;
    bool status = Connect(*a, &b);
    EXPECT_TRUE(IsTopSorted(b));
    EXPECT_TRUE(status);
  }
}

class RmEpsilonTest : public ::testing::Test {
 protected:
  RmEpsilonTest() {
    std::vector<Arc> arcs = {
        {0, 4, 1},  {0, 1, 1},  {1, 2, 0},  {1, 3, 0},  {1, 4, 0},
        {2, 7, 0},  {3, 7, 0},  {4, 6, 1},  {4, 6, 0},  {4, 8, 1},
        {4, 9, -1}, {5, 9, -1}, {6, 9, -1}, {7, 9, -1}, {8, 9, -1},
    };
    fsa_ = new Fsa(std::move(arcs), 9);
    num_states_ = fsa_->NumStates();

    auto num_arcs = fsa_->arcs.size();
    arc_weights_ = new float[num_arcs];
    std::vector<float> weights = {1, 1, 2, 3, 2, 4, 5, 2, 3, 3, 2, 4, 3, 5, 6};
    std::copy_n(weights.begin(), num_arcs, arc_weights_);

    max_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kMaxWeight);
    log_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kLogSumWeight);
  }

  ~RmEpsilonTest() {
    delete fsa_;
    delete[] arc_weights_;
    delete max_wfsa_;
    delete log_wfsa_;
  }

  WfsaWithFbWeights *max_wfsa_;
  WfsaWithFbWeights *log_wfsa_;
  Fsa *fsa_;
  int32_t num_states_;
  float *arc_weights_;
};

TEST_F(RmEpsilonTest, RmEpsilonsPrunedMax) {
  Fsa b;
  std::vector<std::vector<int32_t>> arc_derivs_b;
  RmEpsilonsPrunedMax(*max_wfsa_, 8.0, &b, &arc_derivs_b);

  EXPECT_TRUE(IsEpsilonFree(b));
  ASSERT_EQ(b.arcs.size(), 11);
  ASSERT_EQ(b.arc_indexes.size(), 7);
  ASSERT_EQ(arc_derivs_b.size(), 11);

  // TODO(haowen): check the equivalence after implementing RandEquivalent for
  // WFSA
}

TEST_F(RmEpsilonTest, RmEpsilonsPrunedLogSum) {
  Fsa b;
  std::vector<float> arc_weights_b;
  std::vector<std::vector<std::pair<int32_t, float>>> arc_derivs_b;
  RmEpsilonsPrunedLogSum(*log_wfsa_, 8.0, &b, &arc_weights_b, &arc_derivs_b);

  EXPECT_TRUE(IsEpsilonFree(b));
  ASSERT_EQ(b.arcs.size(), 11);
  ASSERT_EQ(b.arc_indexes.size(), 7);
  ASSERT_EQ(arc_weights_b.size(), 11);
  ASSERT_EQ(arc_derivs_b.size(), 11);

  // TODO(haowen): check the equivalence after implementing RandEquivalent for
  // RmEpsilonPrunedLogSum
}

TEST(FsaAlgo, Intersect) {
  // empty fsa
  {
    Fsa a;
    Fsa b;
    Fsa c;
    // arbitrary states and arcs
    c.arcs = {{0, 1, 2}};
    c.arc_indexes = {1};
    std::vector<int32_t> arc_map_a(10);  // an arbitrary number
    std::vector<int32_t> arc_map_b(5);
    bool status = Intersect(a, b, &c, &arc_map_a, &arc_map_b);
    EXPECT_TRUE(status);
    EXPECT_TRUE(c.arc_indexes.empty());
    EXPECT_TRUE(c.arcs.empty());
    EXPECT_TRUE(arc_map_a.empty());
    EXPECT_TRUE(arc_map_b.empty());
  }

  {
    std::vector<Arc> arcs_a = {{0, 1, 1}, {1, 2, 0}, {1, 3, 1},
                               {1, 4, 2}, {2, 2, 1}, {2, 3, 1},
                               {2, 3, 2}, {3, 3, 0}, {3, 4, 1}};
    Fsa a(std::move(arcs_a), 4);

    std::vector<Arc> arcs_b = {
        {0, 1, 1},
        {1, 3, 1},
        {1, 2, 2},
        {2, 3, 1},
    };
    Fsa b(std::move(arcs_b), 3);

    Fsa c;
    std::vector<int32_t> arc_map_a(10);  // an arbitrary number
    std::vector<int32_t> arc_map_b(5);
    bool status = Intersect(a, b, &c, &arc_map_a, &arc_map_b);
    EXPECT_TRUE(status);

    std::vector<Arc> arcs_c = {
        {0, 1, 1}, {1, 2, 0}, {1, 3, 1}, {1, 4, 2}, {2, 5, 1},
        {2, 6, 1}, {2, 6, 2}, {3, 3, 0}, {6, 6, 0}, {6, 7, 1},
    };
    std::vector<int32_t> arc_indexes_c = {0, 1, 4, 6, 8, 8, 8, 10, 10};

    ASSERT_EQ(c.arc_indexes.size(), arc_indexes_c.size());
    EXPECT_THAT(c.arc_indexes,
                ::testing::ElementsAre(0, 1, 4, 7, 8, 8, 8, 10, 10));
    ASSERT_EQ(c.arcs.size(), arcs_c.size());
    for (std::size_t i = 0; i != arcs_c.size(); ++i)
      EXPECT_EQ(c.arcs[i], arcs_c[i]);

    // arc index in `c` -> arc index in `a`
    // 0 -> 0
    // 1 -> 1
    // 2 -> 2
    // 3 -> 3
    // 4 -> 4
    // 5 -> 5
    // 6 -> 6
    // 7 -> 7
    // 8 -> 7
    // 9 -> 8
    EXPECT_THAT(arc_map_a,
                ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 7, 8));

    // arc index in `c` -> arc index in `b`
    // 0 -> 0
    // 1 -> -1
    // 2 -> 1
    // 3 -> 2
    // 4 -> 1
    // 5 -> 1
    // 6 -> 2
    // 7 -> -1
    // 8 -> -1
    // 9 -> 3
    EXPECT_THAT(arc_map_b,
                ::testing::ElementsAre(0, -1, 1, 2, 1, 1, 2, -1, -1, 3));
  }
}

TEST(FsaAlgo, ArcSort) {
  // empty fsa
  {
    Fsa fsa;
    Fsa arc_sorted;
    // arbitrary states and arcs
    arc_sorted.arcs = {{0, 1, 2}};
    arc_sorted.arc_indexes = {1};
    std::vector<int32_t> arc_map(10);  // an arbitrary number
    ArcSort(fsa, &arc_sorted, &arc_map);
    EXPECT_TRUE(arc_sorted.arcs.empty());
    EXPECT_TRUE(arc_sorted.arc_indexes.empty());
    EXPECT_TRUE(arc_map.empty());
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 2}, {0, 4, 0}, {0, 2, 0}, {1, 2, 1}, {1, 3, 0}, {2, 1, 0},
    };
    Fsa fsa(std::move(arcs), 4);
    Fsa arc_sorted;
    std::vector<int32_t> arc_map;
    ArcSort(fsa, &arc_sorted, &arc_map);
    EXPECT_THAT(arc_sorted.arc_indexes,
                ::testing::ElementsAre(0, 3, 5, 6, 6, 6));
    ASSERT_EQ(arc_sorted.arcs.size(), fsa.arcs.size());
    std::vector<Arc> target_arcs = {
        {0, 2, 0}, {0, 4, 0}, {0, 1, 2}, {1, 3, 0}, {1, 2, 1}, {2, 1, 0},
    };
    for (std::size_t i = 0; i != target_arcs.size(); ++i)
      EXPECT_EQ(arc_sorted.arcs[i], target_arcs[i]);

    // arc index in `arc_sortd` -> arc index in original `fsa`
    // 0 -> 2
    // 1 -> 1
    // 2 -> 0
    // 3 -> 4
    // 4 -> 3
    // 5 -> 5
    EXPECT_THAT(arc_map, ::testing::ElementsAre(2, 1, 0, 4, 3, 5));
  }
}

TEST(FsaAlgo, TopSort) {
  {
    // case 1: empty input fsa
    Fsa fsa;
    Fsa top_sorted;
    std::vector<int32_t> state_map(10);
    bool status = TopSort(fsa, &top_sorted, &state_map);

    EXPECT_TRUE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 2: non-connected fsa (not co-accessible)
    std::string s = R"(
      0 2 3
      1 2 1
      2
    )";
    auto fsa = StringToFsa(s);
    ASSERT_NE(fsa.get(), nullptr);

    Fsa top_sorted;
    std::vector<int32_t> state_map(10);
    bool status = TopSort(*fsa, &top_sorted, &state_map);

    EXPECT_FALSE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 3: non-connected fsa (not accessible)
    std::string s = R"(
      0 2 3
      1 0 1
      2
    )";
    auto fsa = StringToFsa(s);
    ASSERT_NE(fsa.get(), nullptr);

    Fsa top_sorted;
    std::vector<int32_t> state_map(10);
    bool status = TopSort(*fsa, &top_sorted, &state_map);

    EXPECT_FALSE(status);
    EXPECT_TRUE(IsEmpty(top_sorted));
    EXPECT_TRUE(state_map.empty());
  }

  {
    // case 4: connected fsa
    std::string s = R"(
      0 4 40
      0 2 20
      1 6 2
      2 3 30
      3 6 60
      3 1 10
      4 5 50
      5 2 8
      6
    )";
    auto fsa = StringToFsa(s);
    ASSERT_NE(fsa.get(), nullptr);

    Fsa top_sorted;
    std::vector<int32_t> state_map;

    TopSort(*fsa, &top_sorted, &state_map);

    ASSERT_EQ(top_sorted.NumStates(), fsa->NumStates());

    ASSERT_FALSE(state_map.empty());
    EXPECT_THAT(state_map, ::testing::ElementsAre(0, 4, 5, 2, 3, 1, 6));

    ASSERT_FALSE(IsEmpty(top_sorted));

    const auto &arc_indexes = top_sorted.arc_indexes;
    const auto &arcs = top_sorted.arcs;

    ASSERT_EQ(arc_indexes.size(), 8u);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 2, 3, 4, 5, 7, 8, 8));
    std::vector<Arc> expected_arcs = {
        {0, 1, 40}, {0, 3, 20}, {1, 2, 50}, {2, 3, 8},
        {3, 4, 30}, {4, 6, 60}, {4, 5, 10}, {5, 6, 2},
    };

    for (auto i = 0; i != 8; ++i) {
      EXPECT_EQ(arcs[i], expected_arcs[i]);
    }
  }
}

class DeterminizeTest : public ::testing::Test {
 protected:
  DeterminizeTest() {
    std::vector<Arc> arcs = {{0, 4, 1}, {0, 1, 1},  {1, 2, 2},  {1, 3, 3},
                             {2, 7, 1}, {3, 7, 1},  {4, 6, 1},  {4, 6, 1},
                             {4, 5, 1}, {4, 8, -1}, {5, 8, -1}, {6, 8, -1},
                             {7, 8, -1}};
    fsa_ = new Fsa(std::move(arcs), 8);
    num_states_ = fsa_->NumStates();

    auto num_arcs = fsa_->arcs.size();
    arc_weights_ = new float[num_arcs];
    std::vector<float> weights = {1, 1, 2, 3, 4, 5, 2, 3, 3, 2, 4, 3, 5};
    std::copy_n(weights.begin(), num_arcs, arc_weights_);

    max_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kMaxWeight);
    log_wfsa_ = new WfsaWithFbWeights(*fsa_, arc_weights_, kLogSumWeight);

    output_fsa.arc_indexes = {0, 1, 5, 6, 7, 8, 9, 9};
    output_fsa.arcs = {{0, 1, 1}, {1, 6, -1}, {1, 5, 1},  {1, 3, 3}, {1, 2, 2},
                       {2, 4, 1}, {3, 4, 1},  {4, 6, -1}, {5, 6, -1}};
  }

  ~DeterminizeTest() {
    delete fsa_;
    delete[] arc_weights_;
    delete max_wfsa_;
    delete log_wfsa_;
  }

  WfsaWithFbWeights *max_wfsa_;
  WfsaWithFbWeights *log_wfsa_;
  Fsa *fsa_;
  int32_t num_states_;
  float *arc_weights_;
  Fsa output_fsa;
};

TEST_F(DeterminizeTest, DeterminizePrunedMax) {
  Fsa b;
  std::vector<float> b_arc_weights;
  std::vector<std::vector<int32_t>> arc_derivs;
  DeterminizePrunedMax(*max_wfsa_, 10.0, 100, &b, &b_arc_weights, &arc_derivs);

  EXPECT_TRUE(IsDeterministic(b));
  EXPECT_TRUE(IsRandEquivalent<kMaxWeight>(
      max_wfsa_->fsa, max_wfsa_->arc_weights, b, b_arc_weights.data(), 10.0));
}

TEST_F(DeterminizeTest, DeterminizePrunedLogSum) {
  Fsa b;
  std::vector<float> b_arc_weights;
  std::vector<std::vector<std::pair<int32_t, float>>> arc_derivs;
  DeterminizePrunedLogSum(*log_wfsa_, 10.0, 100, &b, &b_arc_weights,
                          &arc_derivs);

  EXPECT_TRUE(IsDeterministic(b));
  EXPECT_TRUE(IsRandEquivalent<kLogSumWeight>(
      log_wfsa_->fsa, log_wfsa_->arc_weights, b, b_arc_weights.data(), 10.0));

  // TODO(haowen): how to check `arc_derivs_out` here, may return `num_steps` to
  // check the sum of `derivs_out` for each output arc?
}

TEST(FsaAlgo, CreateFsa) {
  {
    // clang-format off
    std::vector<Arc> arcs = {
      {0, 3, 3},
      {0, 2, 2},
      {2, 3, 3},
      {2, 4, 4},
      {3, 1, 1},
      {1, 4, 4},
      {1, 8, 8},
      {4, 8, 8},
      {8, 6, 6},
      {8, 7, 7},
      {6, 7, 7},
      {7, 5, 5},
    };
    // clang-format on
    Fsa a;
    std::vector<int32_t> arc_map;
    CreateFsa(arcs, &a, &arc_map);

    auto num_states = a.NumStates();

    Fsa b;
    Swap(&a, &b);
    EXPECT_EQ(a.NumStates(), 0);
    EXPECT_EQ(b.NumStates(), num_states);
    EXPECT_EQ(arc_map.size(), arcs.size());
    EXPECT_THAT(arc_map,
                ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
  }
}

}  // namespace k2
