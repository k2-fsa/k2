// k2/csrc/properties_test.cc

// Copyright (c)  2020  Haowen Qiu
//                      Fangjun Kuang (csukuangfj@gmail.com)
//                      Mahsa Yarmohammadi

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/properties.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"

namespace k2 {

TEST(Properties, IsNotValid) {
  // fsa should contain at least two states.
  {
    Fsa fsa;
    fsa.size1 = 1;
    fsa.indexes = &fsa.size1;
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // only kFinalSymbol arcs enter the final state
  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 1},
        {1, 2, 0},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_valid = IsValid(fsa);
    EXPECT_FALSE(is_valid);
  }

  // `arc_indexes` and `arcs` in this state are not consistent
  {
    std::vector<int32_t> arc_indexes = {0, 2, 2, 2};
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 1},
        {1, 2, 0},
    };
    Fsa fsa(3, 3, arc_indexes.data(), arcs.data());
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
        {0, 2, 0},
        {2, 3, kFinalSymbol},
    };
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_valid = IsValid(fsa);
    EXPECT_TRUE(is_valid);
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, kFinalSymbol},
        {1, 2, kFinalSymbol},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
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
  FsaCreator fsa_creator(arcs, 2);
  const auto &fsa = fsa_creator.GetFsa();
  EXPECT_FALSE(IsTopSorted(fsa));
}

TEST(Properties, IsTopSorted) {
  {
    // empty fsa
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
    EXPECT_TRUE(IsTopSorted(fsa));
  }

  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
        {1, 2, 0},
    };

    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    EXPECT_TRUE(IsTopSorted(fsa));
  }

  {
    RandFsaOptions opts;
    opts.num_syms = 20;
    opts.num_states = 30;
    opts.num_arcs = 50;
    opts.allow_empty = false;
    opts.acyclic = true;
    opts.seed = 20200517;

    RandFsaGenerator generator(opts);
    Array2Size<int32_t> fsa_size;
    generator.GetSizes(&fsa_size);

    FsaCreator fsa_creator(fsa_size);
    auto &fsa = fsa_creator.GetFsa();
    generator.GetOutput(&fsa);

    bool sorted = IsTopSorted(fsa);
    EXPECT_TRUE(sorted);
  }
}

TEST(Properties, IsNotArcSorted) {
  {
    std::vector<Arc> arcs = {
        {0, 1, 1},
        {0, 2, 2},
        {1, 2, 2},
        {1, 3, 1},
    };
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    bool sorted = IsArcSorted(fsa);
    EXPECT_FALSE(sorted);
  }

  // another case with same label on two arcs
  {
    std::vector<Arc> arcs = {
        {0, 2, 0},
        {0, 1, 0},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    bool sorted = IsArcSorted(fsa);
    EXPECT_FALSE(sorted);
  }
}

TEST(Properties, IsArcSorted) {
  // empty fsa is arc-sorted.
  {
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
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
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    bool sorted = IsArcSorted(fsa);
    EXPECT_TRUE(sorted);
  }
}

TEST(Properties, HasNoSelfLoops) {
  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
        {1, 2, 0},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    bool has_self_loops = HasSelfLoops(fsa);
    EXPECT_FALSE(has_self_loops);
  }

  {
    RandFsaOptions opts;
    opts.num_syms = 10;
    opts.num_states = 55;
    opts.num_arcs = 30;
    opts.allow_empty = false;
    opts.acyclic = true;
    opts.seed = 20200517;

    RandFsaGenerator generator(opts);
    Array2Size<int32_t> fsa_size;
    generator.GetSizes(&fsa_size);

    FsaCreator fsa_creator(fsa_size);
    auto &fsa = fsa_creator.GetFsa();
    generator.GetOutput(&fsa);

    bool has_self_loops = HasSelfLoops(fsa);
    EXPECT_FALSE(has_self_loops);
  }
}

TEST(Properties, HasSelfLoops) {
  std::vector<Arc> arcs = {
      {0, 1, 0},
      {1, 2, 0},
      {1, 1, 0},
  };
  FsaCreator fsa_creator(arcs, 2);
  const auto &fsa = fsa_creator.GetFsa();
  bool has_self_loops = HasSelfLoops(fsa);
  EXPECT_TRUE(has_self_loops);
}

TEST(Properties, IsNotDeterministic) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {1, 2, 0},
      {1, 3, 0},
  };
  FsaCreator fsa_creator(arcs, 3);
  const auto &fsa = fsa_creator.GetFsa();
  bool is_deterministic = IsDeterministic(fsa);
  EXPECT_FALSE(is_deterministic);
}

TEST(Properties, IsDeterministic) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {1, 2, 0},
      {1, 3, 2},
  };
  FsaCreator fsa_creator(arcs, 3);
  const auto &fsa = fsa_creator.GetFsa();
  bool is_deterministic = IsDeterministic(fsa);
  EXPECT_TRUE(is_deterministic);
}

TEST(Properties, IsNotEpsilonFree) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {0, 2, kEpsilon},
      {1, 2, 1},
  };
  FsaCreator fsa_creator(arcs, 2);
  const auto &fsa = fsa_creator.GetFsa();
  bool is_epsilon_free = IsEpsilonFree(fsa);
  EXPECT_FALSE(is_epsilon_free);
}

TEST(Properties, IsEpsilonFree) {
  std::vector<Arc> arcs = {
      {0, 1, 2},
      {0, 2, 1},
      {1, 2, 1},
  };
  FsaCreator fsa_creator(arcs, 2);
  const auto &fsa = fsa_creator.GetFsa();
  bool is_epsilon_free = IsEpsilonFree(fsa);
  EXPECT_TRUE(is_epsilon_free);
}

TEST(Properties, IsNotConnected) {
  // state is not accessible
  {
    std::vector<Arc> arcs = {
        {0, 2, 0},
    };
    FsaCreator fsa_creator(arcs, 2);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_connected = IsConnected(fsa);
    EXPECT_FALSE(is_connected);
  }

  // state is not co-accessible
  {
    std::vector<Arc> arcs = {
        {0, 1, 0},
        {0, 2, 0},
    };
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_connected = IsConnected(fsa);
    EXPECT_FALSE(is_connected);
  }
}

TEST(Properties, IsConnected) {
  // empty fsa is connected
  {
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
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
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }

  // another case: fsa is cyclic and not top-sorted
  {
    std::vector<Arc> arcs = {
        {0, 3, 0}, {1, 2, 0}, {2, 3, 0}, {2, 4, 0}, {3, 1, 0},
    };
    FsaCreator fsa_creator(arcs, 4);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }

  {
    // acyclic case
    RandFsaOptions opts;
    opts.num_syms = 10;
    opts.num_states = 20;
    opts.num_arcs = 35;
    opts.allow_empty = false;
    opts.acyclic = true;
    opts.seed = 20200517;

    RandFsaGenerator generator(opts);
    Array2Size<int32_t> fsa_size;
    generator.GetSizes(&fsa_size);

    FsaCreator fsa_creator(fsa_size);
    auto &fsa = fsa_creator.GetFsa();
    generator.GetOutput(&fsa);
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }

  {
    // cyclic case
    RandFsaOptions opts;
    opts.num_syms = 8;
    opts.num_states = 20;
    opts.num_arcs = 30;
    opts.allow_empty = false;
    opts.acyclic = false;
    opts.seed = 20200517;

    RandFsaGenerator generator(opts);
    Array2Size<int32_t> fsa_size;
    generator.GetSizes(&fsa_size);

    FsaCreator fsa_creator(fsa_size);
    auto &fsa = fsa_creator.GetFsa();
    generator.GetOutput(&fsa);
    bool is_connected = IsConnected(fsa);
    EXPECT_TRUE(is_connected);
  }
}

TEST(FsaAlgo, IsAcyclic) {
  // empty fsa is acyclic
  {
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();
    bool is_acyclic = IsAcyclic(fsa);
    EXPECT_TRUE(is_acyclic);
  }

  // an acyclic fsa example
  {
    std::vector<Arc> arcs = {
        {0, 1, 2}, {0, 2, 1}, {1, 2, 0}, {1, 3, 5}, {2, 3, 6},
    };
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_acyclic = IsAcyclic(fsa);
    EXPECT_TRUE(is_acyclic);
  }

  // a cyclic fsa example
  {
    std::vector<Arc> arcs = {
        {0, 1, 2}, {0, 4, 0}, {0, 2, 0}, {1, 2, 1}, {1, 3, 0}, {2, 1, 0},
    };
    FsaCreator fsa_creator(arcs, 4);
    const auto &fsa = fsa_creator.GetFsa();
    bool is_acyclic = IsAcyclic(fsa);
    EXPECT_FALSE(is_acyclic);
  }

  {
    RandFsaOptions opts;
    opts.num_syms = 5;
    opts.num_states = 30;
    opts.num_arcs = 50;
    opts.allow_empty = false;
    opts.acyclic = true;
    opts.seed = 20200517;

    RandFsaGenerator generator(opts);
    Array2Size<int32_t> fsa_size;
    generator.GetSizes(&fsa_size);

    FsaCreator fsa_creator(fsa_size);
    auto &fsa = fsa_creator.GetFsa();
    generator.GetOutput(&fsa);

    bool is_acyclic = IsAcyclic(fsa);
    EXPECT_TRUE(is_acyclic);
  }
}

}  // namespace k2
