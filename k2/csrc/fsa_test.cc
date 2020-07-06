// k2/csrc/fsa_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/array.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/util.h"

namespace k2 {

TEST(CfsaVec, CreateCfsa) {
  std::vector<Arc> arcs1 = {
      {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {3, 4, -1},
  };
  FsaCreator fsa_creator1(arcs1, 4);
  Cfsa cfsa1 = fsa_creator1.GetFsa();
  EXPECT_EQ(cfsa1.NumStates(), 5);
  EXPECT_EQ(cfsa1.size2, 5);  // num-arcs

  std::vector<Arc> arcs2 = {
      {0, 2, 1},
      {0, 3, -1},
      {1, 3, -1},
      {2, 3, -1},
  };
  FsaCreator fsa_creator2(arcs2, 3);
  Cfsa cfsa2 = fsa_creator2.GetFsa();
  EXPECT_EQ(cfsa2.NumStates(), 4);
  EXPECT_EQ(cfsa2.size2, 4);  // num-arcs

  std::vector<Cfsa> cfsas;
  cfsas.emplace_back(cfsa1);
  cfsas.emplace_back(cfsa2);

  CfsaVec cfsa_vec;
  cfsa_vec.GetSizes(cfsas.data(), 2);
  EXPECT_EQ(cfsa_vec.size1, 2);
  EXPECT_EQ(cfsa_vec.size2, cfsa1.NumStates() + cfsa2.NumStates());
  EXPECT_EQ(cfsa_vec.size3, cfsa1.size2 + cfsa2.size2);

  // Test CfsaVec Creation
  std::vector<int32_t> cfsa_vec_indexes1(cfsa_vec.size1 + 1);
  std::vector<int32_t> cfsa_vec_indexes2(cfsa_vec.size2 + 1);
  std::vector<Arc> cfsa_vec_data(cfsa_vec.size3);
  cfsa_vec.indexes1 = cfsa_vec_indexes1.data();
  cfsa_vec.indexes2 = cfsa_vec_indexes2.data();
  cfsa_vec.data = cfsa_vec_data.data();

  cfsa_vec.Create(cfsas.data(), 2);
  EXPECT_THAT(cfsa_vec_indexes1, ::testing::ElementsAre(0, 5, 9));
  EXPECT_THAT(cfsa_vec_indexes2,
              ::testing::ElementsAre(0, 2, 4, 4, 5, 5, 7, 8, 9, 9));
  for (auto i = cfsa1.indexes[0]; i != cfsa1.indexes[cfsa1.size1]; ++i) {
    EXPECT_EQ(cfsa_vec.data[i], cfsa1.data[i]);
  }
  for (auto i = cfsa2.indexes[0]; i != cfsa2.indexes[cfsa2.size1]; ++i) {
    EXPECT_EQ(cfsa_vec.data[cfsa1.size2 + i - cfsa2.indexes[0]], cfsa2.data[i]);
  }

  // Test operator[]
  auto array1_copy = cfsa_vec[0];
  Cfsa *cfsa1_copy_ptr = static_cast<Cfsa *>(&array1_copy);  // cast here
  const auto &cfsa1_copy = *cfsa1_copy_ptr;
  // should call `NumStates` successfully
  EXPECT_EQ(cfsa1_copy.NumStates(), cfsa1.NumStates());
  EXPECT_EQ(cfsa1_copy.size2, cfsa1.size2);
  for (auto i = 0; i != cfsa1.size1 + 1; ++i) {
    EXPECT_EQ(cfsa1_copy.indexes[i], cfsa1.indexes[i]);
  }
  for (auto i = cfsa1.indexes[0]; i != cfsa1.indexes[cfsa1.size1]; ++i) {
    EXPECT_EQ(cfsa1_copy.data[i], cfsa1.data[i]);
  }

  auto array2_copy = cfsa_vec[1];
  Cfsa *cfsa2_copy_ptr = static_cast<Cfsa *>(&array2_copy);  // cast here
  const auto &cfsa2_copy = *cfsa2_copy_ptr;
  // should call `NumStates` successfully
  EXPECT_EQ(cfsa2_copy.NumStates(), cfsa2.NumStates());
  EXPECT_EQ(cfsa2_copy.size2, cfsa2.size2);
  for (auto i = 0; i != cfsa2.size1 + 1; ++i) {
    // output indexes may starts from n > 0
    EXPECT_EQ(cfsa2_copy.indexes[i], cfsa2.indexes[i] + cfsa1.size1);
  }
  for (auto i = cfsa2.indexes[0]; i != cfsa2.indexes[cfsa2.size1]; ++i) {
    EXPECT_EQ(cfsa2_copy.data[i + cfsa1.size2 - cfsa2.indexes[0]],
              cfsa2.data[i]);
  }
}

}  // namespace k2
