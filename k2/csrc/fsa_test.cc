// k2/csrc/fsa_test.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa.h"

#include "gtest/gtest.h"

#include "k2/csrc/fsa_util.h"

namespace k2 {

TEST(Cfsa, ConstructorNonEmptyFsa) {
  std::string s = R"(
      0 1 1
      0 2 2
      1 3 3
      2 3 3
      3 4 -1
      4
    )";
  auto fsa = StringToFsa(s);
  Cfsa cfsa(*fsa);

  EXPECT_EQ(cfsa.num_states, 5);
  EXPECT_EQ(cfsa.begin_arc, 0);
  EXPECT_EQ(cfsa.end_arc, 5);
  EXPECT_EQ(cfsa.arc_indexes, fsa->arc_indexes.data());
  EXPECT_EQ(cfsa.arcs, fsa->arcs.data());

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
  size_t bytes = GetCfsaVecSize(cfsa);
  // 20-byte header             (20)
  // 44-byte padding            (64)
  // 8-byte state_offsets_array (68)
  // 60-byte padding            (128)
  // 4-byte arc_indexes_array   (132)
  EXPECT_EQ(bytes, 132u);

  std::vector<Cfsa> cfsa_vec;
  cfsa_vec.push_back(cfsa);
  bytes = GetCfsaVecSize(cfsa_vec);
  EXPECT_EQ(bytes, 132u);
}

TEST(GetCfsaVecSize, NonEmpty) {
  std::string s = R"(
      0 1 1
      0 2 2
      1 3 3
      2 3 3
      3 16 -1
      16
    )";
  auto fsa = StringToFsa(s);
  Cfsa cfsa(*fsa);

  size_t bytes = GetCfsaVecSize(cfsa);
  // 20-byte header             (20)
  // 44-byte padding            (64)
  // 8-byte state_offset_array  (68)
  // 60-byte padding            (128)
  // 72-byte arc_indexes_array  (200)
  // 60-byte arcs_array         (260)
  EXPECT_EQ(bytes, 260u);
  // Note that there are 5 arcs, sizeof(Arc) == 12.
  // There are 17 states and each state needs 4 bytes
  // and the last state is repeated, so the arc_indexes
  // array needs 18*4 = 72-byte

  {
    std::vector<Cfsa> cfsa_vec;
    cfsa_vec.push_back(cfsa);
    bytes = GetCfsaVecSize(cfsa);
    EXPECT_EQ(bytes, 260u);
  }
}

}  // namespace k2
