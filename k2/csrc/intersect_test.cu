/**
 * @brief Unittest for compose.cu (actually that implements Intersect()..)
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation    (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/math.h"
#include "k2/csrc/test_utils.h"
#include "k2/csrc/host_shim.h"

namespace k2 {

TEST(Intersect, Simple) {
  std::string s = R"(0 1 1 1.0
    1 1 1 50.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )";
  auto fsa = FsaFromString(s);

  // clang-format off
  DenseFsaVec dfsavec {
    RaggedShape("[ [ x x x ] ]"),
    Array2<float>("[ [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.04 0.05 0.06 ] [ 1.0 -Inf -Inf -Inf]]")  // NOLINT
  };
  // clang-format on

  float beam = 100000;
  int32_t max_active = 10000, min_active = 0;

  FsaVec out_fsas;
  Array1<int32_t> arc_map_a, arc_map_b;
  IntersectDensePruned(fsa, dfsavec, beam, max_active, min_active,
                       &out_fsas, &arc_map_a, &arc_map_b);
  K2_LOG(INFO) << "out_fsas = " << out_fsas
               << ", arc_map_a = " << arc_map_a
               << ", arc_map_b = " << arc_map_b;


  FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
  K2_LOG(INFO) << "fsas_b = " << fsas_b;
  FsaVec out_fsas2;
  Array1<int32_t> arc_map_a2, arc_map_b2;
  // IntersectDensePruned() treats epsilons as normal symbols.
  bool treat_epsilons_specially = false;
  Intersect(fsa, fsas_b, treat_epsilons_specially,
            &out_fsas2,
            &arc_map_a2, &arc_map_b2);
  K2_CHECK(IsRandEquivalent(out_fsas, out_fsas2));

  K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
               << ", arc_map_a2 = " << arc_map_a2
               << ", arc_map_b2 = " << arc_map_b2;


  /*
  int32_t gt = kFsaPropertiesTopSorted | kFsaPropertiesTopSortedAndAcyclic;
  int32_t p = GetFsaBasicProperties(fsa);
  EXPECT_NE(p & gt, gt);
  */

  // CheckArrayData(arc_map, {0, 1, 3, 4, 2});
}


TEST(Intersect, TwoDense) {
  std::string s = R"(0 1 1 1.0
    1 1 1 50.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )";
  auto fsa = FsaFromString(s);

  // clang-format off
  DenseFsaVec dfsavec {
    RaggedShape("[ [ x x x ] [ x x x x ] ]"),
    Array2<float>("[ [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.04 0.05 0.06 ] [ 1.0 -Inf -Inf -Inf] "  // NOLINT
                  "  [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.4 0.5 0.6 ] [ -Inf 0.0 0.0 0.0 ] [ 2.0 -Inf -Inf -Inf] ]")  // NOLINT
  };
  // clang-format on

  float beam = 100000;
  int32_t max_active = 10000, min_active = 0;

  FsaVec out_fsas;
  Array1<int32_t> arc_map_a, arc_map_b;
  IntersectDensePruned(fsa, dfsavec, beam, max_active, min_active,
                       &out_fsas, &arc_map_a, &arc_map_b);
  K2_LOG(INFO) << "out_fsas = " << out_fsas
               << ", arc_map_a = " << arc_map_a
               << ", arc_map_b = " << arc_map_b;


  FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
  K2_LOG(INFO) << "fsas_b = " << fsas_b;
  FsaVec out_fsas2;
  Array1<int32_t> arc_map_a2, arc_map_b2;
  // IntersectDensePruned() treats epsilons as normal symbols.
  bool treat_epsilons_specially = false;
  Intersect(fsa, fsas_b, treat_epsilons_specially,
            &out_fsas2,
            &arc_map_a2, &arc_map_b2);
  K2_CHECK(IsRandEquivalent(out_fsas, out_fsas2));

  K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
               << ", arc_map_a2 = " << arc_map_a2
               << ", arc_map_b2 = " << arc_map_b2;
}

TEST(Intersect, TwoFsas) {
  std::string s1 = R"(0 1 1 1.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )", s2 = R"(0 1 1 1.0
    1 1 1 50.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )";
  Fsa fsa1 = FsaFromString(s1), fsa2 = FsaFromString(s2);
  Fsa *fsa_array[] = {&fsa1, &fsa2};
  FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);

  // clang-format off
  DenseFsaVec dfsavec {
    RaggedShape("[ [ x x x ] [ x x x x ] ]"),
    Array2<float>("[ [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.04 0.05 0.06 ] [ 1.0 -Inf -Inf -Inf] "  // NOLINT
                  "  [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.4 0.5 0.6 ] [ -Inf 0.0 0.0 0.0 ] [ 2.0 -Inf -Inf -Inf] ]")  // NOLINT
  };
  // clang-format on

  float beam = 100000;
  int32_t max_active = 10000, min_active = 0;

  FsaVec out_fsas;
  Array1<int32_t> arc_map_a, arc_map_b;
  IntersectDensePruned(fsa_vec, dfsavec, beam, max_active, min_active,
                       &out_fsas, &arc_map_a, &arc_map_b);
  K2_LOG(INFO) << "out_fsas = " << out_fsas
               << ", arc_map_a = " << arc_map_a
               << ", arc_map_b = " << arc_map_b;


  FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
  K2_LOG(INFO) << "fsas_b = " << fsas_b;
  FsaVec out_fsas2;
  Array1<int32_t> arc_map_a2, arc_map_b2;
  // IntersectDensePruned() treats epsilons as normal symbols.
  bool treat_epsilons_specially = false;
  Intersect(fsa_vec, fsas_b, treat_epsilons_specially,
            &out_fsas2,
            &arc_map_a2, &arc_map_b2);
  K2_CHECK(IsRandEquivalent(out_fsas, out_fsas2));

  K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
               << ", arc_map_a2 = " << arc_map_a2
               << ", arc_map_b2 = " << arc_map_b2;
}



TEST(Intersect, RandomSingle) {
  for (int32_t i = 0; i < 10; i++) {
    int32_t max_symbol = 10, min_num_arcs = 0, max_num_arcs = 10;
    bool acyclic = false, epsilon_free = false;
    Fsa fsa = RandomFsa(acyclic, epsilon_free, max_symbol,
                        min_num_arcs, max_num_arcs);
    ArcSort(&fsa);

    int32_t min_num_fsas = 1, max_num_fsas = 1,
        min_frames = 0, max_frames = 10,
        min_nsymbols = max_symbol + 1, max_nsymbols = max_symbol + 4;
    float scores_scale = 1.0;
    DenseFsaVec dfsavec = RandomDenseFsaVec(min_num_fsas, max_num_fsas,
                                            min_frames, max_frames,
                                            min_nsymbols, max_nsymbols,
                                            scores_scale);

    K2_LOG(INFO) << "fsa = " << fsa;

    K2_LOG(INFO) << "dfsavec= " << dfsavec;

    Array1<int32_t> arc_map_a, arc_map_b;

    FsaVec out_fsas;
    float beam = 10000.0;
    int32_t max_active = 10000, min_active = 0;
    IntersectDensePruned(fsa, dfsavec, beam, max_active, min_active,
                         &out_fsas, &arc_map_a, &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas
                 << ", arc_map_b = " << arc_map_b;

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2;
    Array1<int32_t> arc_map_a2, arc_map_b2;
    // IntersectDensePruned() treats epsilons as normal symbols, so we need to
    // as well.

    ArcSort(&fsa);  // CAUTION if you later test the arc_maps: we arc-sort here,
                    // so the input `fsa` is not the same as before.
    bool treat_epsilons_specially = false;
    Intersect(fsa, fsas_b, treat_epsilons_specially,
              &out_fsas2, &arc_map_a2, &arc_map_b2);
    K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
                 << ", arc_map_a2 = " << arc_map_a2
                 << ", arc_map_b2 = " << arc_map_b2;
    K2_CHECK(IsRandEquivalent(out_fsas, out_fsas2));
  }
}

}  // namespace k2
