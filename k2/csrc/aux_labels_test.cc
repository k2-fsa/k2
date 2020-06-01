// k2/csrc/aux_labels_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/aux_labels.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"

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
    EXPECT_EQ(aux_labels_out.start_pos.size(), 1);
    EXPECT_EQ(aux_labels_out.start_pos[0], 0);
  }

  {
    std::vector<int32_t> arc_map = {2, 0, 3};
    AuxLabels aux_labels_out;
    MapAuxLabels1(aux_labels_in_, arc_map, &aux_labels_out);

    EXPECT_EQ(aux_labels_out.start_pos.size(), 4);
    EXPECT_THAT(aux_labels_out.start_pos, ::testing::ElementsAre(0, 3, 4, 5));
    EXPECT_EQ(aux_labels_out.labels.size(), 5);
    EXPECT_THAT(aux_labels_out.labels, ::testing::ElementsAre(4, 5, 6, 1, 7));
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
    EXPECT_EQ(aux_labels_out.start_pos.size(), 1);
    EXPECT_EQ(aux_labels_out.start_pos[0], 0);
  }

  {
    std::vector<std::vector<int32_t>> arc_map = {{2, 3}, {0, 1}, {0}, {2}};
    AuxLabels aux_labels_out;
    MapAuxLabels2(aux_labels_in_, arc_map, &aux_labels_out);

    EXPECT_EQ(aux_labels_out.start_pos.size(), 5);
    EXPECT_THAT(aux_labels_out.start_pos,
                ::testing::ElementsAre(0, 4, 7, 8, 11));
    EXPECT_EQ(aux_labels_out.labels.size(), 11);
    EXPECT_THAT(aux_labels_out.labels,
                ::testing::ElementsAre(4, 5, 6, 7, 1, 2, 3, 1, 4, 5, 6));
  }
}

}  // namespace k2
