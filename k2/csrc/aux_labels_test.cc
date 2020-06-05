// k2/csrc/aux_labels_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/aux_labels.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/properties.h"

namespace k2 {

class AuxLablesTest : public ::testing::Test {
 protected:
  AuxLablesTest() {
    std::vector<int32_t> start_pos = {0, 1, 3, 6, 7};
    std::vector<int32_t> labels = {1, 2, 3, 4, 5, 6, 7};
    aux_labels_in_.start_pos = std::move(start_pos);
    aux_labels_in_.labels = std::move(labels);
  }

  AuxLabels aux_labels_in_;
};

TEST_F(AuxLablesTest, MapAuxLabels1) {
  {
    // empty arc_map
    std::vector<int32_t> arc_map;
    AuxLabels aux_labels_out;
    // some dirty data
    aux_labels_out.start_pos = {1, 2, 3};
    aux_labels_out.labels = {4, 5};
    MapAuxLabels1(aux_labels_in_, arc_map, &aux_labels_out);

    EXPECT_TRUE(aux_labels_out.labels.empty());
    ASSERT_EQ(aux_labels_out.start_pos.size(), 1);
    EXPECT_EQ(aux_labels_out.start_pos[0], 0);
  }

  {
    std::vector<int32_t> arc_map = {2, 0, 3};
    AuxLabels aux_labels_out;
    MapAuxLabels1(aux_labels_in_, arc_map, &aux_labels_out);

    ASSERT_EQ(aux_labels_out.start_pos.size(), 4);
    EXPECT_THAT(aux_labels_out.start_pos, ::testing::ElementsAre(0, 3, 4, 5));
    ASSERT_EQ(aux_labels_out.labels.size(), 5);
    EXPECT_THAT(aux_labels_out.labels, ::testing::ElementsAre(4, 5, 6, 1, 7));
  }

  {
    // all arcs in input fsa are remained
    std::vector<int32_t> arc_map = {2, 0, 3, 1};
    AuxLabels aux_labels_out;
    MapAuxLabels1(aux_labels_in_, arc_map, &aux_labels_out);

    ASSERT_EQ(aux_labels_out.start_pos.size(), 5);
    EXPECT_THAT(aux_labels_out.start_pos,
                ::testing::ElementsAre(0, 3, 4, 5, 7));
    ASSERT_EQ(aux_labels_out.labels.size(), 7);
    EXPECT_THAT(aux_labels_out.labels,
                ::testing::ElementsAre(4, 5, 6, 1, 7, 2, 3));
  }
}

TEST_F(AuxLablesTest, MapAuxLabels2) {
  {
    // empty arc_map
    std::vector<std::vector<int32_t>> arc_map;
    AuxLabels aux_labels_out;
    // some dirty data
    aux_labels_out.start_pos = {1, 2, 3};
    aux_labels_out.labels = {4, 5};
    MapAuxLabels2(aux_labels_in_, arc_map, &aux_labels_out);

    EXPECT_TRUE(aux_labels_out.labels.empty());
    ASSERT_EQ(aux_labels_out.start_pos.size(), 1);
    EXPECT_EQ(aux_labels_out.start_pos[0], 0);
  }

  {
    std::vector<std::vector<int32_t>> arc_map = {{2, 3}, {0, 1}, {0}, {2}};
    AuxLabels aux_labels_out;
    MapAuxLabels2(aux_labels_in_, arc_map, &aux_labels_out);

    ASSERT_EQ(aux_labels_out.start_pos.size(), 5);
    EXPECT_THAT(aux_labels_out.start_pos,
                ::testing::ElementsAre(0, 4, 7, 8, 11));
    ASSERT_EQ(aux_labels_out.labels.size(), 11);
    EXPECT_THAT(aux_labels_out.labels,
                ::testing::ElementsAre(4, 5, 6, 7, 1, 2, 3, 1, 4, 5, 6));
  }
}

TEST(AuxLabels, InvertFst) {
  {
    // empty input FSA
    Fsa fsa_in;
    AuxLabels labels_in;
    std::vector<int32_t> start_pos = {0, 1, 3, 6, 7};
    std::vector<int32_t> labels = {1, 2, 3, 4, 5, 6, 7};
    labels_in.start_pos = std::move(start_pos);
    labels_in.labels = std::move(labels);

    std::vector<Arc> arcs = {{0, 1, 1}, {1, 2, -1}};
    Fsa fsa_out(std::move(arcs), 2);
    AuxLabels labels_out;
    // some dirty data
    labels_out.start_pos = {1, 2, 3};
    labels_out.labels = {4, 5};
    InvertFst(fsa_in, labels_in, &fsa_out, &labels_out);

    EXPECT_TRUE(IsEmpty(fsa_out));
    EXPECT_TRUE(labels_out.labels.empty());
    ASSERT_EQ(labels_out.start_pos.size(), 1);
    EXPECT_EQ(labels_out.start_pos[0], 0);
  }

  {
    // top-sorted input FSA
    std::vector<Arc> arcs = {{0, 1, 1}, {0, 1, 0},  {0, 3, 2},
                             {1, 2, 3}, {1, 3, 4},  {1, 5, -1},
                             {2, 3, 0}, {2, 5, -1}, {4, 5, -1}};
    Fsa fsa_in(std::move(arcs), 5);
    EXPECT_TRUE(IsTopSorted(fsa_in));
    AuxLabels labels_in;
    std::vector<int32_t> start_pos = {0, 2, 3, 3, 6, 6, 7, 7, 8, 9};
    EXPECT_EQ(start_pos.size(), fsa_in.arcs.size() + 1);
    std::vector<int32_t> labels = {1, 2, 3, 5, 6, 7, -1, -1, -1};
    labels_in.start_pos = std::move(start_pos);
    labels_in.labels = std::move(labels);

    Fsa fsa_out;
    AuxLabels labels_out;
    InvertFst(fsa_in, labels_in, &fsa_out, &labels_out);

    EXPECT_TRUE(IsTopSorted(fsa_out));
    std::vector<Arc> arcs_out = {
        {0, 1, 1},  {0, 2, 3}, {0, 6, 0}, {1, 2, 2}, {2, 3, 5},  {2, 6, 0},
        {2, 8, -1}, {3, 4, 6}, {4, 5, 7}, {5, 6, 0}, {5, 8, -1}, {7, 8, -1},
    };
    ASSERT_EQ(fsa_out.arcs.size(), arcs_out.size());
    for (auto i = 0; i != arcs_out.size(); ++i) {
      EXPECT_EQ(fsa_out.arcs[i], arcs_out[i]);
    }
    ASSERT_EQ(fsa_out.arc_indexes.size(), 10);
    EXPECT_THAT(fsa_out.arc_indexes,
                ::testing::ElementsAre(0, 3, 4, 7, 8, 9, 11, 11, 12, 12));
    ASSERT_EQ(labels_out.labels.size(), 7);
    EXPECT_THAT(labels_out.labels,
                ::testing::ElementsAre(2, 1, 4, -1, 3, -1, -1));
    ASSERT_EQ(labels_out.start_pos.size(), 13);
    EXPECT_THAT(labels_out.start_pos,
                ::testing::ElementsAre(0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7));
  }

  {
    // non-top-sorted input FSA
    std::vector<Arc> arcs = {{0, 1, 1},  {0, 1, 0}, {0, 3, 2},
                             {1, 2, 3},  {1, 3, 4}, {2, 1, 5},
                             {2, 5, -1}, {3, 1, 6}, {4, 5, -1}};
    Fsa fsa_in(std::move(arcs), 5);
    EXPECT_FALSE(IsTopSorted(fsa_in));
    AuxLabels labels_in;
    std::vector<int32_t> start_pos = {0, 2, 3, 3, 6, 6, 7, 8, 10, 11};
    EXPECT_EQ(start_pos.size(), fsa_in.arcs.size() + 1);
    std::vector<int32_t> labels = {1, 2, 3, 5, 6, 7, 8, -1, 9, 10, -1};
    labels_in.start_pos = std::move(start_pos);
    labels_in.labels = std::move(labels);

    Fsa fsa_out;
    AuxLabels labels_out;
    InvertFst(fsa_in, labels_in, &fsa_out, &labels_out);

    EXPECT_FALSE(IsTopSorted(fsa_out));
    std::vector<Arc> arcs_out = {{0, 1, 1},  {0, 3, 3}, {0, 7, 0},  {1, 3, 2},
                                 {2, 3, 10}, {3, 4, 5}, {3, 7, 0},  {4, 5, 6},
                                 {5, 6, 7},  {6, 3, 8}, {6, 9, -1}, {7, 2, 9},
                                 {8, 9, -1}};
    ASSERT_EQ(fsa_out.arcs.size(), arcs_out.size());
    for (auto i = 0; i != arcs_out.size(); ++i) {
      EXPECT_EQ(fsa_out.arcs[i], arcs_out[i]);
    }
    ASSERT_EQ(fsa_out.arc_indexes.size(), 11);
    EXPECT_THAT(fsa_out.arc_indexes,
                ::testing::ElementsAre(0, 3, 4, 5, 7, 8, 9, 11, 12, 13, 13));
    ASSERT_EQ(labels_out.labels.size(), 8);
    EXPECT_THAT(labels_out.labels,
                ::testing::ElementsAre(2, 1, 6, 4, 3, 5, -1, -1));
    ASSERT_EQ(labels_out.start_pos.size(), 14);
    EXPECT_THAT(
        labels_out.start_pos,
        ::testing::ElementsAre(0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8));
  }
}

}  // namespace k2
