// k2/csrc/fsa_equivalent_test.cc

// Copyright (c)  2020  Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_equivalent.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "k2/csrc/fsa.h"

namespace k2 {

TEST(FsaEquivalent, IsNotRandEquivalent) {
  {
    // one fsa will be empty after connecting
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5},
    };
    FsaCreator fsa_creator_a(arcs_a, 3);
    const auto &a = fsa_creator_a.GetFsa();

    std::vector<Arc> arcs_b = {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}};
    FsaCreator fsa_creator_b(arcs_b, 3);
    const auto &b = fsa_creator_b.GetFsa();

    bool status = IsRandEquivalent(a, b);
    EXPECT_FALSE(status);
  }

  {
    // two fsas hold different set of arc labels
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5},
    };
    FsaCreator fsa_creator_a(arcs_a, 3);
    const auto &a = fsa_creator_a.GetFsa();

    std::vector<Arc> arcs_b = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 6},
    };
    FsaCreator fsa_creator_b(arcs_b, 3);
    const auto &b = fsa_creator_b.GetFsa();

    bool status = IsRandEquivalent(a, b);
    EXPECT_FALSE(status);
  }

  {
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5},
    };
    FsaCreator fsa_creator_a(arcs_a, 3);
    const auto &a = fsa_creator_a.GetFsa();

    std::vector<Arc> arcs_b = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5}, {3, 4, 5},
    };
    FsaCreator fsa_creator_b(arcs_b, 4);
    const auto &b = fsa_creator_b.GetFsa();

    bool status = IsRandEquivalent(a, b, 100);
    // Caution: this test may fail with a very low probability if the program
    // generates path from `a` every time
    EXPECT_FALSE(status);
  }
}

TEST(FsaEquivalent, IsRandEquivalent) {
  {
    // both fsas will be empty after connecting
    std::vector<Arc> arcs_a = {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}};
    FsaCreator fsa_creator_a(arcs_a, 3);
    const auto &a = fsa_creator_a.GetFsa();

    std::vector<Arc> arcs_b = {{0, 1, 1}, {0, 2, 2}};
    FsaCreator fsa_creator_b(arcs_b, 3);
    const auto &b = fsa_creator_b.GetFsa();

    bool status = IsRandEquivalent(a, b);
    EXPECT_TRUE(status);
  }

  {
    // same fsas
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {1, 3, 4}, {2, 3, 5},
    };
    FsaCreator fsa_creator_a(arcs_a, 3);
    const auto &a = fsa_creator_a.GetFsa();
    bool status = IsRandEquivalent(a, a);
    EXPECT_TRUE(status);
  }

  {
    std::vector<Arc> arcs_a = {
        {0, 1, 1}, {0, 2, 2}, {0, 3, 8}, {1, 4, 4}, {2, 4, 5},
    };
    FsaCreator fsa_creator_a(arcs_a, 4);
    const auto &a = fsa_creator_a.GetFsa();

    std::vector<Arc> arcs_b = {
        {0, 2, 1}, {0, 1, 2}, {0, 3, 9}, {1, 4, 5}, {2, 4, 4},
    };
    FsaCreator fsa_creator_b(arcs_b, 4);
    const auto &b = fsa_creator_b.GetFsa();

    bool status = IsRandEquivalent(a, b);
    EXPECT_TRUE(status);
  }
}

TEST(FsaEquivalent, IsWfsaRandEquivalent) {
  std::vector<Arc> arcs_a = {{0, 1, 1}, {0, 1, 2},  {0, 1, 3}, {0, 2, 4},
                             {0, 2, 5}, {1, 3, 5},  {1, 3, 6}, {2, 4, 5},
                             {2, 4, 6}, {3, 5, -1}, {4, 5, -1}};
  FsaCreator fsa_creator_a(arcs_a, 5);
  const auto &a = fsa_creator_a.GetFsa();
  std::vector<float> a_weights = {2, 2, 3, 3, 1, 3, 2, 5, 4, 1, 3};

  std::vector<Arc> arcs_b = {
      {0, 1, 1}, {0, 1, 2}, {0, 1, 3}, {0, 1, 4},
      {0, 1, 5}, {1, 2, 5}, {1, 2, 6}, {2, 3, -1},
  };
  FsaCreator fsa_creator_b(arcs_b, 3);
  const auto &b = fsa_creator_b.GetFsa();
  std::vector<float> b_weights = {5, 5, 6, 10, 8, 1, 0, 0};

  std::vector<Arc> arcs_c = {
      {0, 1, 1}, {0, 1, 2}, {0, 1, 3}, {0, 1, 4},
      {0, 1, 5}, {1, 2, 5}, {1, 2, 6}, {2, 3, -1},
  };
  FsaCreator fsa_creator_c(arcs_c, 3);
  const auto &c = fsa_creator_c.GetFsa();
  std::vector<float> c_weights = {5, 5, 6, 10, 9, 1, 0, 0};

  {
    bool status =
        IsRandEquivalent<kMaxWeight>(a, a_weights.data(), b, b_weights.data());
    EXPECT_TRUE(status);
  }

  {
    bool status =
        IsRandEquivalent<kMaxWeight>(a, a_weights.data(), c, c_weights.data());
    EXPECT_FALSE(status);
  }

  {
    bool status = IsRandEquivalent<kLogSumWeight>(a, a_weights.data(), b,
                                                  b_weights.data());
    EXPECT_TRUE(status);
  }

  {
    bool status = IsRandEquivalent<kLogSumWeight>(a, a_weights.data(), c,
                                                  c_weights.data());
    EXPECT_FALSE(status);
  }

  // check equivalence with beam
  {
    bool status = IsRandEquivalent<kMaxWeight>(a, a_weights.data(), b,
                                               b_weights.data(), 4.0);
    EXPECT_TRUE(status);
  }
  // check equivalence with beam
  {
    bool status = IsRandEquivalent<kMaxWeight>(a, a_weights.data(), c,
                                               c_weights.data(), 6.0);
    EXPECT_FALSE(status);
  }
}

TEST(FsaEquivalent, RandomPathFail) {
  {
    FsaCreator fsa_creator;
    const auto &fsa = fsa_creator.GetFsa();

    RandPath rand_path(fsa, false);
    Array2Size<int32_t> fsa_size;
    rand_path.GetSizes(&fsa_size);

    EXPECT_EQ(fsa_size.size1, 0);
    EXPECT_EQ(fsa_size.size2, 0);

    FsaCreator fsa_creator_out(fsa_size);
    auto &path = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    bool status = rand_path.GetOutput(&path, arc_map.data());
    EXPECT_FALSE(status);
    EXPECT_TRUE(arc_map.empty());
  }

  {
    // non-connected fsa
    std::vector<Arc> arcs = {
        {0, 1, 1},
        {0, 2, 2},
        {1, 3, 4},
    };
    FsaCreator fsa_creator(arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    RandPath rand_path(fsa, false);
    Array2Size<int32_t> fsa_size;
    rand_path.GetSizes(&fsa_size);

    EXPECT_EQ(fsa_size.size1, 0);
    EXPECT_EQ(fsa_size.size2, 0);

    FsaCreator fsa_creator_out(fsa_size);
    auto &path = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    bool status = rand_path.GetOutput(&path, arc_map.data());
    EXPECT_FALSE(status);
    EXPECT_TRUE(arc_map.empty());
  }
}

TEST(FsaEquivalent, RandomPathSuccess) {
  {
    std::vector<Arc> arcs = {
        {0, 1, 1}, {0, 2, 2}, {1, 2, 3}, {2, 3, 4},
        {2, 4, 5}, {3, 4, 7}, {4, 5, 9},
    };
    FsaCreator fsa_creator(arcs, 5);
    const auto &fsa = fsa_creator.GetFsa();
    Fsa path;

    {
      RandPath rand_path(fsa, false);
      Array2Size<int32_t> fsa_size;
      rand_path.GetSizes(&fsa_size);

      EXPECT_GT(fsa_size.size1, 0);
      EXPECT_GT(fsa_size.size2, 0);

      FsaCreator fsa_creator_out(fsa_size);
      auto &path = fsa_creator_out.GetFsa();
      bool status = rand_path.GetOutput(&path);
      EXPECT_TRUE(status);
    }

    {
      RandPath rand_path(fsa, false);
      for (auto i = 0; i != 20; ++i) {
        Array2Size<int32_t> fsa_size;
        rand_path.GetSizes(&fsa_size);

        EXPECT_GT(fsa_size.size1, 0);
        EXPECT_GT(fsa_size.size2, 0);

        FsaCreator fsa_creator_out(fsa_size);
        auto &path = fsa_creator_out.GetFsa();
        std::vector<int32_t> arc_map(fsa_size.size2);
        bool status = rand_path.GetOutput(&path, arc_map.data());
        EXPECT_TRUE(status);
        EXPECT_GT(arc_map.size(), 0);
      }
    }
  }

  // test with linear structure fsa to check the resulted path
  {
    std::vector<Arc> src_arcs = {
        {0, 1, 1},
        {1, 2, 3},
        {2, 3, 4},
    };
    FsaCreator fsa_creator(src_arcs, 3);
    const auto &fsa = fsa_creator.GetFsa();
    RandPath rand_path(fsa, false);
    Array2Size<int32_t> fsa_size;
    rand_path.GetSizes(&fsa_size);

    FsaCreator fsa_creator_out(fsa_size);
    auto &path = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    bool status = rand_path.GetOutput(&path, arc_map.data());

    EXPECT_TRUE(status);
    std::vector<int32_t> arc_indexes(path.indexes,
                                     path.indexes + path.size1 + 1);
    std::vector<Arc> arcs(path.data, path.data + path.size2);
    ASSERT_EQ(fsa.size1, path.size1);
    EXPECT_THAT(arc_indexes, ::testing::ElementsAre(0, 1, 2, 3, 3));
    ASSERT_EQ(fsa.size2, path.size2);
    for (std::size_t i = 0; i != path.size2; ++i)
      EXPECT_EQ(arcs[i], src_arcs[i]);
    EXPECT_THAT(arc_map, ::testing::ElementsAre(0, 1, 2));
  }
}

TEST(FsaEquivalent, RandomPathWithoutEpsilonArc) {
  {
    std::vector<Arc> arcs = {
        {0, 1, 1}, {0, 2, 0}, {1, 2, 3}, {2, 3, 0},
        {2, 4, 5}, {3, 4, 7}, {4, 5, 9},
    };
    FsaCreator fsa_creator(arcs, 5);
    const auto &fsa = fsa_creator.GetFsa();

    {
      RandPath rand_path(fsa, true);
      for (auto i = 0; i != 20; ++i) {
        Array2Size<int32_t> fsa_size;
        rand_path.GetSizes(&fsa_size);

        FsaCreator fsa_creator_out(fsa_size);
        auto &path = fsa_creator_out.GetFsa();
        std::vector<int32_t> arc_map(fsa_size.size2);
        bool status = rand_path.GetOutput(&path, arc_map.data());
        EXPECT_TRUE(status);
        EXPECT_GT(arc_map.size(), 0);
        for (const auto &arc : path) {
          EXPECT_NE(arc.label, kEpsilon);
        }
      }
    }
  }

  // there's no epsilon-free path
  {
    std::vector<Arc> arcs = {
        {0, 1, 1}, {0, 2, 0}, {1, 2, 3}, {2, 3, 0},
        {3, 5, 7}, {3, 4, 8}, {4, 5, 9},
    };
    FsaCreator fsa_creator(arcs, 5);
    const auto &fsa = fsa_creator.GetFsa();
    RandPath rand_path(fsa, true);
    Array2Size<int32_t> fsa_size;
    rand_path.GetSizes(&fsa_size);

    EXPECT_EQ(fsa_size.size1, 0);
    EXPECT_EQ(fsa_size.size2, 0);

    FsaCreator fsa_creator_out(fsa_size);
    auto &path = fsa_creator_out.GetFsa();
    std::vector<int32_t> arc_map(fsa_size.size2);
    bool status = rand_path.GetOutput(&path, arc_map.data());
    EXPECT_FALSE(status);
    EXPECT_TRUE(arc_map.empty());
  }
}
}  // namespace k2
