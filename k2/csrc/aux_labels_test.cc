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
    aux_labels_in_.size1 = static_cast<int32_t>(start_pos_.size()) - 1;
    aux_labels_in_.size2 = static_cast<int32_t>(labels_.size());
    aux_labels_in_.indexes = start_pos_.data();
    aux_labels_in_.data = labels_.data();
  }

  std::vector<int32_t> start_pos_ = {0, 1, 3, 6, 7};
  std::vector<int32_t> labels_ = {1, 2, 3, 4, 5, 6, 7};
  AuxLabels aux_labels_in_;
};

TEST_F(AuxLablesTest, AuxLabels1Mapper) {
  {
    // empty arc_map
    std::vector<int32_t> arc_map;
    AuxLabels1Mapper aux_mapper(aux_labels_in_, arc_map);
    Array2Size<int32_t> aux_size;
    aux_mapper.GetSizes(&aux_size);
    Array2Storage<int32_t *, int32_t> storage(aux_size, 1);
    auto aux_labels_out = storage.GetArray2();
    aux_mapper.GetOutput(&aux_labels_out);

    ASSERT_EQ(aux_labels_out.size1, 0);
    EXPECT_EQ(aux_labels_out.indexes[0], 0);
    EXPECT_EQ(aux_labels_out.size2, 0);
  }

  {
    std::vector<int32_t> arc_map = {2, 0, 3};
    AuxLabels1Mapper aux_mapper(aux_labels_in_, arc_map);
    Array2Size<int32_t> aux_size;
    aux_mapper.GetSizes(&aux_size);
    Array2Storage<int32_t *, int32_t> storage(aux_size, 1);
    auto aux_labels_out = storage.GetArray2();
    aux_mapper.GetOutput(&aux_labels_out);

    ASSERT_EQ(aux_labels_out.size1, 3);
    ASSERT_EQ(aux_labels_out.size2, 5);
    std::vector<int32_t> out_indexes(
        aux_labels_out.indexes,
        aux_labels_out.indexes + aux_labels_out.size1 + 1);
    std::vector<int32_t> out_data(aux_labels_out.data,
                                  aux_labels_out.data + aux_labels_out.size2);
    EXPECT_THAT(out_indexes, ::testing::ElementsAre(0, 3, 4, 5));
    EXPECT_THAT(out_data, ::testing::ElementsAre(4, 5, 6, 1, 7));
  }

  {
    // all arcs in input fsa are remained
    std::vector<int32_t> arc_map = {2, 0, 3, 1};
    AuxLabels1Mapper aux_mapper(aux_labels_in_, arc_map);
    Array2Size<int32_t> aux_size;
    aux_mapper.GetSizes(&aux_size);
    Array2Storage<int32_t *, int32_t> storage(aux_size, 1);
    auto aux_labels_out = storage.GetArray2();
    aux_mapper.GetOutput(&aux_labels_out);

    ASSERT_EQ(aux_labels_out.size2, 7);
    ASSERT_EQ(aux_labels_out.size1, 4);
    std::vector<int32_t> out_indexes(
        aux_labels_out.indexes,
        aux_labels_out.indexes + aux_labels_out.size1 + 1);
    std::vector<int32_t> out_data(aux_labels_out.data,
                                  aux_labels_out.data + aux_labels_out.size2);
    EXPECT_THAT(out_indexes, ::testing::ElementsAre(0, 3, 4, 5, 7));
    EXPECT_THAT(out_data, ::testing::ElementsAre(4, 5, 6, 1, 7, 2, 3));
  }
}

TEST_F(AuxLablesTest, AuxLabels2Mapper) {
  {
    // empty arc_map
    std::vector<std::vector<int32_t>> arc_map;
    AuxLabels2Mapper aux_mapper(aux_labels_in_, arc_map);
    Array2Size<int32_t> aux_size;
    aux_mapper.GetSizes(&aux_size);
    Array2Storage<int32_t *, int32_t> storage(aux_size, 1);
    auto aux_labels_out = storage.GetArray2();
    aux_mapper.GetOutput(&aux_labels_out);

    ASSERT_EQ(aux_labels_out.size1, 0);
    EXPECT_EQ(aux_labels_out.indexes[0], 0);
    EXPECT_EQ(aux_labels_out.size2, 0);
  }

  {
    std::vector<std::vector<int32_t>> arc_map = {{2, 3}, {0, 1}, {0}, {2}};
    AuxLabels2Mapper aux_mapper(aux_labels_in_, arc_map);
    Array2Size<int32_t> aux_size;
    aux_mapper.GetSizes(&aux_size);
    Array2Storage<int32_t *, int32_t> storage(aux_size, 1);
    auto aux_labels_out = storage.GetArray2();
    aux_mapper.GetOutput(&aux_labels_out);

    ASSERT_EQ(aux_labels_out.size2, 11);
    ASSERT_EQ(aux_labels_out.size1, 4);
    std::vector<int32_t> out_indexes(
        aux_labels_out.indexes,
        aux_labels_out.indexes + aux_labels_out.size1 + 1);
    std::vector<int32_t> out_data(aux_labels_out.data,
                                  aux_labels_out.data + aux_labels_out.size2);
    EXPECT_THAT(out_indexes, ::testing::ElementsAre(0, 4, 7, 8, 11));
    EXPECT_THAT(out_data,
                ::testing::ElementsAre(4, 5, 6, 7, 1, 2, 3, 1, 4, 5, 6));
  }
}

TEST(AuxLabels, InvertFst) {
  {
    // empty input FSA
    FsaCreator fsa_in_creator({0, 0});
    const auto &fsa_in = fsa_in_creator.GetFsa();
    std::vector<int32_t> start_pos = {0, 1, 3, 6, 7};
    std::vector<int32_t> labels = {1, 2, 3, 4, 5, 6, 7};
    AuxLabels labels_in(static_cast<int32_t>(start_pos.size()) - 1,
                        static_cast<int32_t>(labels.size()), start_pos.data(),
                        labels.data());

    FstInverter fst_inverter(fsa_in, labels_in);
    Array2Size<int32_t> fsa_size, aux_size;
    fst_inverter.GetSizes(&fsa_size, &aux_size);
    Array2Storage<int32_t *, int32_t> aux_storage(aux_size, 1);
    auto labels_out = aux_storage.GetArray2();
    FsaCreator fsa_creator(fsa_size);
    auto &fsa_out = fsa_creator.GetFsa();
    fst_inverter.GetOutput(&fsa_out, &labels_out);

    EXPECT_TRUE(IsEmpty(fsa_out));
    EXPECT_TRUE(labels_out.Empty());
  }

  {
    // top-sorted input FSA
    std::vector<Arc> arcs = {{0, 1, 1}, {0, 1, 0},  {0, 3, 2},
                             {1, 2, 3}, {1, 3, 4},  {1, 5, -1},
                             {2, 3, 0}, {2, 5, -1}, {4, 5, -1}};
    FsaCreator fsa_in_creator(arcs, 5);
    const auto &fsa_in = fsa_in_creator.GetFsa();
    EXPECT_TRUE(IsTopSorted(fsa_in));
    std::vector<int32_t> start_pos = {0, 2, 3, 3, 6, 6, 7, 7, 8, 9};
    EXPECT_EQ(start_pos.size(), fsa_in.size2 + 1);
    std::vector<int32_t> labels = {1, 2, 3, 5, 6, 7, -1, -1, -1};
    AuxLabels labels_in(static_cast<int32_t>(start_pos.size()) - 1,
                        static_cast<int32_t>(labels.size()), start_pos.data(),
                        labels.data());

    FstInverter fst_inverter(fsa_in, labels_in);
    Array2Size<int32_t> fsa_size, aux_size;
    fst_inverter.GetSizes(&fsa_size, &aux_size);
    Array2Storage<int32_t *, int32_t> aux_storage(aux_size, 1);
    auto labels_out = aux_storage.GetArray2();
    FsaCreator fsa_creator(fsa_size);
    auto &fsa_out = fsa_creator.GetFsa();
    fst_inverter.GetOutput(&fsa_out, &labels_out);

    EXPECT_TRUE(IsTopSorted(fsa_out));
    std::vector<Arc> arcs_out = {
        {0, 1, 1},  {0, 2, 3}, {0, 6, 0}, {1, 2, 2}, {2, 3, 5},  {2, 6, 0},
        {2, 8, -1}, {3, 4, 6}, {4, 5, 7}, {5, 6, 0}, {5, 8, -1}, {7, 8, -1},
    };
    ASSERT_EQ(fsa_out.size2, arcs_out.size());
    for (auto i = 0; i != arcs_out.size(); ++i) {
      EXPECT_EQ(fsa_out.data[i], arcs_out[i]);
    }
    ASSERT_EQ(fsa_out.size1, 9);
    std::vector<int32_t> arc_indexes(fsa_out.indexes,
                                     fsa_out.indexes + fsa_out.size1 + 1);
    EXPECT_THAT(arc_indexes,
                ::testing::ElementsAre(0, 3, 4, 7, 8, 9, 11, 11, 12, 12));

    ASSERT_EQ(labels_out.size1, 12);
    ASSERT_EQ(labels_out.size2, 7);
    std::vector<int32_t> out_indexes(labels_out.indexes,
                                     labels_out.indexes + labels_out.size1 + 1);
    std::vector<int32_t> out_data(labels_out.data,
                                  labels_out.data + labels_out.size2);
    EXPECT_THAT(out_data, ::testing::ElementsAre(2, 1, 4, -1, 3, -1, -1));
    EXPECT_THAT(out_indexes,
                ::testing::ElementsAre(0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7));
  }

  {
    // non-top-sorted input FSA
    std::vector<Arc> arcs = {{0, 1, 1},  {0, 1, 0}, {0, 3, 2},
                             {1, 2, 3},  {1, 3, 4}, {2, 1, 5},
                             {2, 5, -1}, {3, 1, 6}, {4, 5, -1}};
    FsaCreator fsa_in_creator(arcs, 5);
    const auto &fsa_in = fsa_in_creator.GetFsa();
    EXPECT_FALSE(IsTopSorted(fsa_in));
    std::vector<int32_t> start_pos = {0, 2, 3, 3, 6, 6, 7, 8, 10, 11};
    EXPECT_EQ(start_pos.size(), fsa_in.size2 + 1);
    std::vector<int32_t> labels = {1, 2, 3, 5, 6, 7, 8, -1, 9, 10, -1};
    AuxLabels labels_in(static_cast<int32_t>(start_pos.size()) - 1,
                        static_cast<int32_t>(labels.size()), start_pos.data(),
                        labels.data());

    FstInverter fst_inverter(fsa_in, labels_in);
    Array2Size<int32_t> fsa_size, aux_size;
    fst_inverter.GetSizes(&fsa_size, &aux_size);
    Array2Storage<int32_t *, int32_t> aux_storage(aux_size, 1);
    auto labels_out = aux_storage.GetArray2();
    FsaCreator fsa_creator(fsa_size);
    auto &fsa_out = fsa_creator.GetFsa();
    fst_inverter.GetOutput(&fsa_out, &labels_out);

    EXPECT_FALSE(IsTopSorted(fsa_out));
    std::vector<Arc> arcs_out = {{0, 1, 1},  {0, 3, 3}, {0, 7, 0},  {1, 3, 2},
                                 {2, 3, 10}, {3, 4, 5}, {3, 7, 0},  {4, 5, 6},
                                 {5, 6, 7},  {6, 3, 8}, {6, 9, -1}, {7, 2, 9},
                                 {8, 9, -1}};
    ASSERT_EQ(fsa_out.size2, arcs_out.size());
    for (auto i = 0; i != arcs_out.size(); ++i) {
      EXPECT_EQ(fsa_out.data[i], arcs_out[i]);
    }
    ASSERT_EQ(fsa_out.size1, 10);
    std::vector<int32_t> arc_indexes(fsa_out.indexes,
                                     fsa_out.indexes + fsa_out.size1 + 1);
    EXPECT_THAT(arc_indexes,
                ::testing::ElementsAre(0, 3, 4, 5, 7, 8, 9, 11, 12, 13, 13));

    ASSERT_EQ(labels_out.size1, 13);
    ASSERT_EQ(labels_out.size2, 8);
    std::vector<int32_t> out_indexes(labels_out.indexes,
                                     labels_out.indexes + labels_out.size1 + 1);
    std::vector<int32_t> out_data(labels_out.data,
                                  labels_out.data + labels_out.size2);
    EXPECT_THAT(out_data, ::testing::ElementsAre(2, 1, 6, 4, 3, 5, -1, -1));
    EXPECT_THAT(out_indexes, ::testing::ElementsAre(0, 0, 0, 1, 2, 3, 3, 4, 4,
                                                    5, 6, 7, 7, 8));
  }
}

}  // namespace k2
