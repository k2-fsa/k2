// k2/csrc/fsa_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/util.h"

namespace k2 {
TEST(Cfsa, ConstructorNonEmptyFsa) {
  std::vector<Arc> arcs = {
      {0, 1, 1}, {0, 2, 2}, {1, 3, 3}, {2, 3, 3}, {3, 4, -1},
  };
  Fsa fsa(std::move(arcs), 4);
  Cfsa cfsa(fsa);

  EXPECT_EQ(cfsa.num_states, 5);
  EXPECT_EQ(cfsa.begin_arc, 0);
  EXPECT_EQ(cfsa.end_arc, 5);
  EXPECT_EQ(cfsa.arc_indexes, fsa.arc_indexes.data());
  EXPECT_EQ(cfsa.arcs, fsa.arcs.data());

  EXPECT_EQ(cfsa.NumStates(), 5);
  EXPECT_EQ(cfsa.FinalState(), 4);
}

TEST(Cfsa, ConstructorEmptyFsa) {
  Fsa fsa;
  Cfsa cfsa(fsa);
  EXPECT_EQ(cfsa.num_states, 0);
  EXPECT_EQ(cfsa.begin_arc, 0);
  EXPECT_EQ(cfsa.end_arc, 0);
  EXPECT_EQ(cfsa.arc_indexes, nullptr);
  EXPECT_EQ(cfsa.arcs, nullptr);
}

TEST(GetCfsaVecSize, Empty) {
  Cfsa cfsa;
  std::size_t bytes = GetCfsaVecSize(cfsa);
  // 20-byte header             (20)
  // 44-byte padding            (64)
  // 8-byte state_offsets_array (72)
  // 56-byte padding            (128)
  // 4-byte arc_indexes_array   (132)
  EXPECT_EQ(bytes, 132u);

  std::vector<Cfsa> cfsa_vec;
  cfsa_vec.push_back(cfsa);
  bytes = GetCfsaVecSize(cfsa_vec);
  EXPECT_EQ(bytes, 132u);
}

TEST(GetCfsaVecSize, NonEmpty) {
  std::vector<Arc> arcs = {
      {0, 1, 1}, {0, 2, 2}, {1, 3, 3}, {2, 3, 3}, {3, 16, -1},
  };
  Fsa fsa(std::move(arcs), 16);
  Cfsa cfsa(fsa);

  std::size_t bytes = GetCfsaVecSize(cfsa);
  // 20-byte header             (20)
  // 44-byte padding            (64)
  // 8-byte state_offset_array  (72)
  // 56-byte padding            (128)
  // 72-byte arc_indexes_array  (200)
  // 4-byte padding             (204) -> to be multiple of sizeof(Arc)
  // 60-byte arcs_array         (264)
  EXPECT_EQ(bytes, 264u);
  // Note that there are 5 arcs, sizeof(Arc) == 12.
  // There are 17 states and each state needs 4 bytes
  // and the last state is repeated, so the arc_indexes
  // array needs 18*4 = 72-byte

  {
    std::vector<Cfsa> cfsa_vec;
    cfsa_vec.push_back(cfsa);
    bytes = GetCfsaVecSize(cfsa);
    EXPECT_EQ(bytes, 264u);
  }
}

TEST(GetCfsaVecSize, NonEmptyMutlipeFsas) {
  std::vector<Arc> arcs1 = {
      {0, 1, 1}, {0, 2, 2}, {1, 3, 3}, {2, 3, 3}, {3, 16, -1},
  };
  // 5 arcs, 17 states
  Fsa fsa1(std::move(arcs1), 16);
  Cfsa cfsa1(fsa1);

  // 4 arcs, 11 states
  std::vector<Arc> arcs2 = {
      {0, 1, 1},
      {0, 2, 2},
      {1, 3, 3},
      {3, 10, -1},
  };
  // 5 arcs, 17 states
  Fsa fsa2(std::move(arcs2), 10);
  Cfsa cfsa2(fsa2);

  std::vector<Cfsa> cfsa_vec = {cfsa1, cfsa2};

  std::size_t bytes = GetCfsaVecSize(cfsa_vec);
  // 28 states,9 arcs
  //
  // 20-byte header             (20)
  // 44-byte padding            (64)
  // 12-byte state_offset_array (76)
  // 52-byte padding            (128)
  // 120-byte arc_indexes_array (248)
  // 4-byte padding             (252) -> to be multiple of sizeof(Arc)
  // 108-byte arcs_array        (360)
  EXPECT_EQ(bytes, 360u);
}

TEST(CfsaVec, Empty) {
  Cfsa cfsa;
  std::vector<Cfsa> cfsas;
  std::size_t bytes = GetCfsaVecSize(cfsas);
  std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                 &MemFree);

  CreateCfsaVec(cfsas, data.get(), bytes);

  CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
  EXPECT_EQ(cfsa_vec.NumFsas(), 0);
}

TEST(CfsaVec, OneEmptyCfsa) {
  Cfsa cfsa;
  std::vector<Cfsa> cfsas = {cfsa};
  std::size_t bytes = GetCfsaVecSize(cfsas);
  std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                 &MemFree);

  CreateCfsaVec(cfsas, data.get(), bytes);

  CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
  EXPECT_EQ(cfsa_vec.NumFsas(), 1);
}

TEST(CfsaVec, OneNonEmptyCfsa) {
  std::vector<Arc> arcs = {
      {0, 1, 10}, {0, 2, 2}, {1, 3, 3}, {2, 3, 3}, {2, 4, -1}, {3, 4, -1},
  };
  Fsa fsa(std::move(arcs), 16);
  Cfsa cfsa(fsa);

  std::vector<Cfsa> cfsas = {cfsa};
  std::size_t bytes = GetCfsaVecSize(cfsas);
  std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                 &MemFree);

  CreateCfsaVec(cfsas, data.get(), bytes);

  CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
  EXPECT_EQ(cfsa_vec.NumFsas(), 1);

  Cfsa f = cfsa_vec[0];
  EXPECT_EQ(f, cfsa);
}

TEST(CfsaVec, TwoNoneEmptyCfsa) {
  std::vector<Arc> arcs1 = {
      {0, 1, 10}, {0, 2, 2}, {1, 3, 3}, {2, 3, 3}, {2, 4, -1}, {3, 4, -1},
  };
  Fsa fsa1(std::move(arcs1), 4);
  Cfsa cfsa1(fsa1);

  std::vector<Arc> arcs2 = {{0, 1, 10}, {0, 2, 2},   {1, 3, 3},  {2, 3, 3},
                            {2, 4, 3},  {3, 10, -1}, {4, 10, -1}};
  Fsa fsa2(std::move(arcs2), 10);
  Cfsa cfsa2(fsa2);

  {
    // both fsa are not empty
    std::vector<Cfsa> cfsas = {cfsa1, cfsa2};
    std::size_t bytes = GetCfsaVecSize(cfsas);
    std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                   &MemFree);

    CreateCfsaVec(cfsas, data.get(), bytes);

    CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
    EXPECT_EQ(cfsa_vec.NumFsas(), 2);

    Cfsa f = cfsa_vec[0];
    EXPECT_EQ(f, cfsa1);

    Cfsa g = cfsa_vec[1];
    EXPECT_EQ(g, cfsa2);
  }

  {
    // the first fsa is empty
    Cfsa cfsa;
    std::vector<Cfsa> cfsas = {cfsa, cfsa2};
    std::size_t bytes = GetCfsaVecSize(cfsas);
    std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                   &MemFree);

    CreateCfsaVec(cfsas, data.get(), bytes);

    CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
    EXPECT_EQ(cfsa_vec.NumFsas(), 2);

    Cfsa f = cfsa_vec[0];
    EXPECT_EQ(f, cfsa);

    Cfsa g = cfsa_vec[1];
    EXPECT_EQ(g, cfsa2);
  }

  {
    // the second fsa is empty
    Cfsa cfsa;
    std::vector<Cfsa> cfsas = {cfsa1, cfsa};
    std::size_t bytes = GetCfsaVecSize(cfsas);
    std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                   &MemFree);

    CreateCfsaVec(cfsas, data.get(), bytes);

    CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
    EXPECT_EQ(cfsa_vec.NumFsas(), 2);

    Cfsa f = cfsa_vec[0];
    EXPECT_EQ(f, cfsa1);

    Cfsa g = cfsa_vec[1];
    EXPECT_EQ(g, cfsa);
  }
}
// TODO(haowen): un-comment below lines after replacing Cfsa with Array3
/*
TEST(CfsaVec, RandomFsa) {
  RandFsaOptions opts;
  opts.num_syms = 20;
  opts.num_states = 30;
  opts.num_arcs = 50;
  opts.allow_empty = false;
  opts.acyclic = false;
  opts.seed = 20200531;

  int32_t n = 5;
  std::vector<Fsa> fsa_vec;
  fsa_vec.reserve(n);
  for (int32_t i = 0; i != n; ++i) {
    Fsa fsa;
    GenerateRandFsa(opts, &fsa);
    fsa_vec.emplace_back(std::move(fsa));
  }

  std::vector<Cfsa> cfsas;
  cfsas.reserve(n);
  for (const auto &fsa : fsa_vec) {
    cfsas.emplace_back(fsa);
  }

  std::size_t bytes = GetCfsaVecSize(cfsas);
  std::unique_ptr<void, decltype(&MemFree)> data(MemAlignedMalloc(bytes, 64),
                                                 &MemFree);

  CreateCfsaVec(cfsas, data.get(), bytes);

  CfsaVec cfsa_vec(bytes / sizeof(int32_t), data.get());
  EXPECT_EQ(cfsa_vec.NumFsas(), n);

  for (int32_t i = 0; i != n; ++i) {
    EXPECT_EQ(cfsa_vec[i], cfsas[i]);
  }
}
*/

}  // namespace k2
