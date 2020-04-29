// k2/csrc/properties_test.cc

// Copyright (c)  2020  Haowen Qiu
//                      Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/properties.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"

namespace k2 {

// TODO(haowen): create Fsa examples in a more elegant way (add methods
// addState, addArc, etc.) and use Test Fixtures by constructing
// reusable FSA examples.
TEST(Properties, IsNotValid) {
  // fsa should contain at least two states.
  {
    Fsa fsa;
    std::vector<int32_t> arc_indexes = {0};
    fsa.arc_indexes = std::move(arc_indexes);
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // only kFinalSymbol arcs enter the final state
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 1},
        {1, 2, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 3};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // every state contains at least one arc except the final state
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
        {2, 3, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 2, 3};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // every state contains at least one arc except the final state (another case)
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 2};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // `arc_indexes` and `arcs` in this state are not consistent
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 3, 3};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // `arc_indexes` and `arcs` in this state are not consistent (another case)
  {
    Fsa fsa;
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
        {1, 2, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 4};
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }
}

TEST(Properties, IsValid) {
  // empty fsa is valid.
  {
    Fsa fsa;
    bool is_valid = IsValid(fsa);
    EXPECT_TRUE(is_valid);
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, kFinalSymbol},
        {1, 2, kFinalSymbol},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 3};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_valid = IsValid(fsa);
    EXPECT_TRUE(is_valid);
  }
}

TEST(Properties, IsNotTopSorted) {
  std::vector<Arc> arcs = {
      {0, 1, 0},
      {0, 2, 0},
      {2, 1, 0},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool sorted = IsTopSorted(fsa);
  EXPECT_FALSE(sorted);
}

TEST(Properties, IsTopSorted) {
  std::vector<Arc> arcs = {
      {0, 1, 0},
      {0, 2, 0},
      {1, 2, 0},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool sorted = IsTopSorted(fsa);
  EXPECT_TRUE(sorted);
}

TEST(Properties, IsNotArcSorted) {
  {
    std::vector<Arc> arcs = {
        {0, 1, 1},
        {0, 2, 2},
        {1, 2, 2},
        {1, 3, 1},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 4, 4};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool sorted = IsArcSorted(fsa);
    EXPECT_FALSE(sorted);
  }

  // another case with same label on two arcs
  {
    std::vector<Arc> arcs = {
        {0, 2, 0},
        {0, 1, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 2};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool sorted = IsArcSorted(fsa);
    EXPECT_FALSE(sorted);
  }
}

TEST(Properties, IsArcSorted) {
  // empty fsa is arc-sorted.
  {
    Fsa fsa;
    bool sorted = IsArcSorted(fsa);
    EXPECT_TRUE(sorted);
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
        {1, 2, 1},
        {1, 3, 2},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 4, 4};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool sorted = IsArcSorted(fsa);
    EXPECT_TRUE(sorted);
  }
}

TEST(Properties, HasNoSelfLoops) {
  std::vector<Arc> arcs = {
      {0, 1, 0},
      {0, 2, 0},
      {1, 2, 0},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool has_self_loops = HasSelfLoops(fsa);
  EXPECT_FALSE(has_self_loops);
}

TEST(Properties, HasSelfLoops) {
  std::vector<Arc> arcs = {
      {0, 1, 0},
      {1, 2, 0},
      {1, 1, 0},
  };
  std::vector<int32_t> arc_indexes = {0, 1, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool has_self_loops = HasSelfLoops(fsa);
  EXPECT_TRUE(has_self_loops);
}

TEST(Properties, IsNotDeterministic) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {1, 2, 0},
      {1, 3, 0},
  };
  std::vector<int32_t> arc_indexes = {0, 1, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool is_deterministic = IsDeterministic(fsa);
  EXPECT_FALSE(is_deterministic);
}

TEST(Properties, IsDeterministic) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {1, 2, 0},
      {1, 3, 2},
  };
  std::vector<int32_t> arc_indexes = {0, 1, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool is_deterministic = IsDeterministic(fsa);
  EXPECT_TRUE(is_deterministic);
}

TEST(Properties, IsNotEpsilonFree) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {0, 2, 0},
      {1, 2, 1},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool is_epsilon_free = IsEpsilonFree(fsa);
  EXPECT_FALSE(is_epsilon_free);
}

TEST(Properties, IsEpsilonFree) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {0, 2, 1},
      {1, 2, 1},
  };
  std::vector<int32_t> arc_indexes = {0, 2, 3};
  Fsa fsa;
  fsa.arc_indexes = std::move(arc_indexes);
  fsa.arcs = std::move(arcs);
  bool is_epsilon_free = IsEpsilonFree(fsa);
  EXPECT_TRUE(is_epsilon_free);
}

TEST(Properties, IsNoConnected) {
  // state is not accessible
  {
    std::vector<Arc> arcs = {
        {0, 2, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 1, 1};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_connected = IsConnected(fsa);
    EXPECT_FALSE(is_connected);
  }

  // state is not co-accessible
  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 2, 2};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_connected = IsConnected(fsa);
    EXPECT_FALSE(is_connected);
  }
}

TEST(Properties, IsConnected) {
  // empty fsa is connected
  {
    Fsa fsa;
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }
  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 3, 0},
        {1, 2, 0},
        {2, 3, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 2, 3, 4};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }

  // another case: fsa is cyclic and not top-sorted
  {
    std::vector<Arc> arcs = {
        {0, 3, 0}, {1, 2, 0}, {2, 3, 0}, {2, 4, 0}, {3, 1, 0},
    };
    std::vector<int32_t> arc_indexes = {0, 1, 2, 4, 5};
    Fsa fsa;
    fsa.arc_indexes = std::move(arc_indexes);
    fsa.arcs = std::move(arcs);
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }
}

}  // namespace k2
