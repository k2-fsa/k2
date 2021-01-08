/**
 * @brief Unittests for intersect_pruned.cu and intersect.cu
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
#include "k2/csrc/host_shim.h"
#include "k2/csrc/math.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

bool IsRandEquivalentWrapper(Fsa &a, Fsa &b, bool treat_epsilons_specially) {
  float beam = 10.0;
  bool log_semiring = false;  // actually the intersection algorithm is the same
  // between the semirings; the 2 implementations of
  // intersection should be equivalent in either, I
  // believe, at least with
  // `treat_epsilons_specially=false` in the host
  // one.
  float delta = 0.01;
  std::size_t npath = 100;
  return IsRandEquivalent(a, b, log_semiring, beam, treat_epsilons_specially,
                          delta, npath);
}

TEST(Intersect, Simple) {
  // tests single FSA and also 2 copies of a single FSA.
  for (int i = 0; i < 8; i++) {
    K2_LOG(INFO) << "Intersection for " << (i == 0 ? "CPU" : "GPU");
    ContextPtr c = (i % 2 == 0 ? GetCpuContext() : GetCudaContext());
    std::string s = R"(0 1 1 1.0
    1 1 1 50.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )";

    auto fsa = FsaFromString(s).To(c);

    // clang-format off
    DenseFsaVec dfsavec {
      RaggedShape("[ [ x x x ] ]").To(c),
          Array2<float>("[ [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.04 0.05 0.06 ] [ 1.0 -Inf -Inf -Inf]]").To(c)  // NOLINT
      };
    // clang-format on

    if (i >= 2) {
      // Duplicate fsa and dfsavec, stacking 2 copies.
      {  // fsa
        Fsa *fsa_vec[] = {&fsa, &fsa};
        FsaVec temp = Stack(0, 2, fsa_vec);
        fsa = temp;
      }
      {  // dfsavec
        int32_t nrows = dfsavec.scores.Dim0();
        Array2<float> scores2(c, nrows * 2, dfsavec.scores.Dim1());
        Array2<float> scores2a = scores2.RowArange(0, nrows),
                      scores2b = scores2.RowArange(nrows, nrows * 2);
        Assign(dfsavec.scores, &scores2a);
        Assign(dfsavec.scores, &scores2b);

        RaggedShape *dfsavec_shapes[] = {&dfsavec.shape, &dfsavec.shape};
        RaggedShape stacked_shape = Append(0, 2, dfsavec_shapes);
        dfsavec = DenseFsaVec(stacked_shape, scores2);
      }
    }

    float output_beam = 1000;

    FsaVec out_fsas;
    Array1<int32_t> arc_map_a, arc_map_b;
    IntersectDense(fsa, dfsavec, output_beam, &out_fsas, &arc_map_a,
                   &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas << ", arc_map_a = " << arc_map_a
                 << ", arc_map_b = " << arc_map_b;

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2,
        out_fsas2b;
    ContextPtr cpu = GetCpuContext();
    Array1<int32_t> arc_map_a2, arc_map_b2;

    // IntersectDense() treats epsilons as normal symbols.
    bool treat_epsilons_specially = false;

    Array1<int32_t> arc_map_a3, arc_map_b3;

    {
      FsaVec fsas = FsaToFsaVec(fsa);
      Array1<int32_t> b_to_a_map = Range<int32_t>(c, fsas_b.Dim0(), 0,
                                                  (fsas.Dim0() == 1 ? 0 : 1));

      out_fsas2b = IntersectDevice(fsas, -1, fsas_b, -1, b_to_a_map,
                                   &arc_map_a3, &arc_map_b3);
    }

    {
      fsa = fsa.To(cpu);
      fsas_b = fsas_b.To(cpu);

      Intersect(fsa, -1, fsas_b, -1, treat_epsilons_specially,
                &out_fsas2, &arc_map_a2, &arc_map_b2);
    }

    out_fsas2b = out_fsas2b.To(cpu);
    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas2, out_fsas2b, treat_epsilons_specially));

    /*
    // TODO: really test.
    K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
                 << ", out_fsas2b = " << out_fsas2b
                 << ", arc_map_a2 = " << arc_map_a2
                 << ", arc_map_a3 = " << arc_map_a3
                 << ", arc_map_b2 = " << arc_map_b2
                 << ", arc_map_b3 = " << arc_map_b3;*/


    K2_LOG(INFO) << "out_fsas device type is "
                 << out_fsas.Context()->GetDeviceType();
    out_fsas = out_fsas.To(cpu);
    arc_map_a = arc_map_a.To(cpu);
    arc_map_b = arc_map_b.To(cpu);

    {  // check arc map for out_fsas, arc_map_a, arc_map_b
      DenseFsaVec dfsavec2 = dfsavec.To(cpu);
      int32_t num_arcs = out_fsas.NumElements();
      for (int32_t i = 0; i < num_arcs; i++) {
        int32_t arc_idx_a = arc_map_a[i], arc_idx_b = arc_map_b[i];
        float score_a = fsa.values[arc_idx_a].score,
              score_b = dfsavec2.scores.Data()[arc_idx_b],
              score_composed = out_fsas.values[i].score;
        float margin = 1.0e-04 * (fabs(score_a) + fabs(score_b));
        K2_CHECK((score_a + score_b) == score_composed ||
                 fabs(score_a + score_b - score_composed) < margin);
      }
    }

    {  // check arc map for out_fsas2, arc_map_a2, arc_map_b2
      int32_t num_arcs = out_fsas2.NumElements();
      for (int32_t i = 0; i < num_arcs; i++) {
        int32_t arc_idx_a = arc_map_a2[i], arc_idx_b = arc_map_b2[i];
        float score_a = fsa.values[arc_idx_a].score,
              score_b = fsas_b.values[arc_idx_b].score,
              score_composed = out_fsas2.values[i].score;
        float margin = 1.0e-04 * (fabs(score_a) + fabs(score_b));
        K2_CHECK((score_a + score_b) == score_composed ||
                 fabs(score_a + score_b - score_composed) < margin);
      }
    }

    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));

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
}

TEST(Intersect, RandomSingle) {
  for (int32_t i = 0; i < 10; i++) {
    K2_LOG(INFO) << "Iteration of testing: i = " << i;
    int32_t max_symbol = 10, min_num_arcs = 0, max_num_arcs = 10;
    bool acyclic = false;
    Fsa fsa = RandomFsa(acyclic, max_symbol, min_num_arcs, max_num_arcs);
    ArcSort(&fsa);

    int32_t num_fsas = 1;

    int32_t min_frames = 0, max_frames = 10, min_nsymbols = max_symbol + 1,
            max_nsymbols = max_symbol + 4;
    float scores_scale = 1.0;
    DenseFsaVec dfsavec =
        RandomDenseFsaVec(num_fsas, num_fsas, min_frames, max_frames,
                          min_nsymbols, max_nsymbols, scores_scale);

    K2_LOG(INFO) << "fsa = " << fsa;

    K2_LOG(INFO) << "dfsavec = " << dfsavec;

    if (true) {
      // trying to find bugs where the cutoffs might get mixed up between
      // FSAs
      auto dfsa_acc = dfsavec.scores.Accessor();
      for (int32_t n = 0; n < 10; n++) {
        int32_t i = RandInt(0, dfsavec.scores.Dim0() - 1);
        for (int32_t j = 0; j < dfsavec.scores.Dim1(); j++) {
          dfsa_acc(i, j) += -100.0;
        }
      }
    }

    Array1<int32_t> arc_map_a, arc_map_b;

    FsaVec out_fsas;
    float output_beam = 1000.0;
    IntersectDense(fsa, dfsavec, output_beam, &out_fsas, &arc_map_a,
                   &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas << ", arc_map_b = " << arc_map_b;

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2;
    Array1<int32_t> arc_map_a2, arc_map_b2;
    // IntersectDensePruned() treats epsilons as normal symbols, so we need to
    // as well.

    ArcSort(&fsa);  // CAUTION if you later test the arc_maps: we arc-sort here,
    // so the input `fsa` is not the same as before.
    bool treat_epsilons_specially = false;
    Intersect(fsa, -1, fsas_b, -1, treat_epsilons_specially,
              &out_fsas2, &arc_map_a2, &arc_map_b2);
    K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
                 << ", arc_map_a2 = " << arc_map_a2
                 << ", arc_map_b2 = " << arc_map_b2;
    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));
  }
}

TEST(Intersect, RandomFsaVec) {
  for (int32_t i = 0; i < 10; i++) {
    K2_LOG(INFO) << "Iteration of testing: i = " << i;
    ContextPtr c = (i % 2 == 0 ? GetCpuContext() : GetCudaContext());
    ContextPtr cpu = GetCpuContext();

    int32_t max_symbol = 10, min_num_arcs = 0, max_num_arcs = 200;
    bool acyclic = false;

    int32_t num_b_fsas = RandInt(1, 5), num_a_fsas = num_b_fsas;

    Fsa fsavec = RandomFsaVec(num_a_fsas, num_a_fsas, acyclic, max_symbol,
                              min_num_arcs, max_num_arcs)
                     .To(c);
    ArcSort(&fsavec);

    int32_t min_frames = 0, max_frames = 10, min_nsymbols = max_symbol + 1,
            max_nsymbols = max_symbol + 4;
    float scores_scale = 1.0;
    DenseFsaVec dfsavec =
        RandomDenseFsaVec(num_b_fsas, num_b_fsas, min_frames, max_frames,
                          min_nsymbols, max_nsymbols, scores_scale);

    Array1<int32_t> dfsa_reorder = GetDecreasingSizeOrder(dfsavec.shape);
    dfsavec = dfsavec[dfsa_reorder];
    K2_LOG(INFO) << "Dfsa-vec after reordering is " << dfsavec;

    if (true) {
      // trying to find bugs where the cutoffs might get mixed up between
      // FSAs
      auto dfsa_acc = dfsavec.scores.Accessor();
      for (int32_t n = 0; n < 10; n++) {
        int32_t i = RandInt(0, dfsavec.scores.Dim0() - 1);
        for (int32_t j = 0; j < dfsavec.scores.Dim1(); j++) {
          dfsa_acc(i, j) += -100.0;
        }
      }
    }
    dfsavec = dfsavec.To(c);

    K2_LOG(INFO) << "fsavec = " << fsavec;

    K2_LOG(INFO) << "dfsavec= " << dfsavec;

    Array1<int32_t> arc_map_a, arc_map_b;

    FsaVec out_fsas;
    float output_beam = 100000.0;  // TODO(Dan) ...
    IntersectDense(fsavec, dfsavec, output_beam, &out_fsas, &arc_map_a,
                   &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas
                 << ", arc_map_a = " << arc_map_a
                 << ", arc_map_b = " << arc_map_b;


    fsavec = fsavec.To(cpu);
    out_fsas = out_fsas.To(cpu);
    arc_map_a = arc_map_a.To(cpu);
    arc_map_b = arc_map_b.To(cpu);
    {  // check arc map for out_fsas, arc_map_a, arc_map_b
      DenseFsaVec dfsavec2 = dfsavec.To(cpu);
      int32_t num_arcs = out_fsas.NumElements();
      for (int32_t i = 0; i < num_arcs; i++) {
        int32_t arc_idx_a = arc_map_a[i], arc_idx_b = arc_map_b[i];
        float score_a = fsavec.values[arc_idx_a].score,
              score_b = dfsavec2.scores.Data()[arc_idx_b],
              score_composed = out_fsas.values[i].score;
        float margin = 1.0e-04 * (fabs(score_a) + fabs(score_b));
        K2_CHECK((score_a + score_b) == score_composed ||
                 fabs(score_a + score_b - score_composed) < margin);
      }
    }

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    fsas_b = fsas_b.To(cpu);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2;
    Array1<int32_t> arc_map_a2, arc_map_b2;
    // IntersectDensePruned() treats epsilons as normal symbols, so we need to
    // as well.

    ArcSort(&fsavec);  // CAUTION if you later test the arc_maps: we arc-sort
                       // here, so the input `fsa` is not the same as before.
    bool treat_epsilons_specially = false;


    {
      Array1<int32_t> arc_map_a2_temp,
          arc_map_b2_temp;
      FsaVec out_fsas2_temp;
      Intersect(fsavec, -1, fsas_b, -1, treat_epsilons_specially,
                &out_fsas2_temp, &arc_map_a2_temp, &arc_map_b2_temp);
      Array1<int32_t> connect_arc_map;
      Connect(out_fsas2_temp, &out_fsas2, &connect_arc_map);
      arc_map_a2 = arc_map_a2_temp[connect_arc_map];
      arc_map_b2 = arc_map_b2_temp[connect_arc_map];
    }

    K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
                 << ", arc_map_a2 = " << arc_map_a2
                 << ", arc_map_b2 = " << arc_map_b2;
    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));

    {  // check arc map for out_fsas2, arc_map_a2, arc_map_b2
      int32_t num_arcs = out_fsas2.NumElements();
      for (int32_t i = 0; i < num_arcs; i++) {
        int32_t arc_idx_a = arc_map_a2[i], arc_idx_b = arc_map_b2[i];
        float score_a = fsavec.values[arc_idx_a].score,
              score_b = fsas_b.values[arc_idx_b].score,
              score_composed = out_fsas2.values[i].score;
        float margin = 1.0e-04 * (fabs(score_a) + fabs(score_b));
        K2_CHECK((score_a + score_b) == score_composed ||
                 fabs(score_a + score_b - score_composed) < margin);
      }
    }
  }
}

TEST(IntersectPruned, Simple) {
  for (int i = 0; i < 2; i++) {
    K2_LOG(INFO) << "Intersection for " << (i == 0 ? "CPU" : "GPU");
    ContextPtr c = (i == 0 ? GetCpuContext() : GetCudaContext());
    std::string s = R"(0 1 1 1.0
    1 1 1 50.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )";
    auto fsa = FsaFromString(s).To(c);

    // clang-format off
    DenseFsaVec dfsavec {
      RaggedShape("[ [ x x x ] ]").To(c),
          Array2<float>("[ [ -Inf 0.1 0.2 0.3 ] [ -Inf 0.04 0.05 0.06 ] [ 1.0 -Inf -Inf -Inf]]").To(c)  // NOLINT
      };
    // clang-format on

    float beam = 100000;
    int32_t max_active = 10000, min_active = 0;

    FsaVec out_fsas;
    Array1<int32_t> arc_map_a, arc_map_b;
    IntersectDensePruned(fsa, dfsavec, beam, beam, min_active, max_active,
                         &out_fsas, &arc_map_a, &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas << ", arc_map_a = " << arc_map_a
                 << ", arc_map_b = " << arc_map_b;

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2;
    ContextPtr cpu = GetCpuContext();
    Array1<int32_t> arc_map_a2, arc_map_b2;
    // IntersectDensePruned() treats epsilons as normal symbols.
    bool treat_epsilons_specially = false;
    fsa = fsa.To(cpu);
    fsas_b = fsas_b.To(cpu);
    Intersect(fsa, -1, fsas_b, -1, treat_epsilons_specially,
              &out_fsas2, &arc_map_a2, &arc_map_b2);

    out_fsas = out_fsas.To(cpu);
    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));

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
}

TEST(IntersectPruned, TwoDense) {
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
  IntersectDensePruned(fsa, dfsavec, beam, beam, min_active, max_active,
                       &out_fsas, &arc_map_a, &arc_map_b);
  K2_LOG(INFO) << "out_fsas = " << out_fsas << ", arc_map_a = " << arc_map_a
               << ", arc_map_b = " << arc_map_b;

  FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
  K2_LOG(INFO) << "fsas_b = " << fsas_b;
  FsaVec out_fsas2;
  Array1<int32_t> arc_map_a2, arc_map_b2;
  // IntersectDensePruned() treats epsilons as normal symbols.
  bool treat_epsilons_specially = false;
  Intersect(fsa, -1, fsas_b, -1, treat_epsilons_specially,
            &out_fsas2, &arc_map_a2, &arc_map_b2);
  K2_CHECK(
      IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));

  K2_LOG(INFO) << "out_fsas2 = " << out_fsas2 << ", arc_map_a2 = " << arc_map_a2
               << ", arc_map_b2 = " << arc_map_b2;
}

TEST(IntersectPruned, TwoFsas) {
  std::string s1 = R"(0 1 1 1.0
    1 2 2 2.0
    2 3 -1 3.0
    3
  )",
              s2 = R"(0 1 1 1.0
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
  IntersectDensePruned(fsa_vec, dfsavec, beam, beam, min_active, max_active,
                       &out_fsas, &arc_map_a, &arc_map_b);
  K2_LOG(INFO) << "out_fsas = " << out_fsas << ", arc_map_a = " << arc_map_a
               << ", arc_map_b = " << arc_map_b;

  FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
  K2_LOG(INFO) << "fsas_b = " << fsas_b;
  FsaVec out_fsas2;
  Array1<int32_t> arc_map_a2, arc_map_b2;
  // IntersectDensePruned() treats epsilons as normal symbols.
  bool treat_epsilons_specially = false;
  Intersect(fsa_vec, -1, fsas_b, -1, treat_epsilons_specially,
            &out_fsas2, &arc_map_a2, &arc_map_b2);
  K2_CHECK(
      IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));

  K2_LOG(INFO) << "out_fsas2 = " << out_fsas2 << ", arc_map_a2 = " << arc_map_a2
               << ", arc_map_b2 = " << arc_map_b2;
}

TEST(IntersectPruned, RandomSingle) {
  for (int32_t i = 0; i < 10; i++) {
    K2_LOG(INFO) << "Iteration of testing: i = " << i;
    int32_t max_symbol = 10, min_num_arcs = 0, max_num_arcs = 10;
    bool acyclic = false;
    Fsa fsa = RandomFsa(acyclic, max_symbol, min_num_arcs, max_num_arcs);
    ArcSort(&fsa);

    int32_t num_fsas = 1;
    if (i > 5) {
      K2_LOG(INFO) << "Testing multiple dense fsas in DenseFsaVec.";
      num_fsas = RandInt(2, 5);
    }

    // set max_frames = 50 to be larger than the chunk sizes used for pruning
    // in intersect_pruned.cu (see call to PruneTimeRange()).
    int32_t min_frames = 0, max_frames = 50,
          min_nsymbols = max_symbol + 1,
            max_nsymbols = max_symbol + 4;
    float scores_scale = 1.0;
    DenseFsaVec dfsavec =
        RandomDenseFsaVec(num_fsas, num_fsas, min_frames, max_frames,
                          min_nsymbols, max_nsymbols, scores_scale);

    K2_LOG(INFO) << "fsa = " << fsa;

    K2_LOG(INFO) << "dfsavec= " << dfsavec;

    if (true) {
      // trying to find bugs where the cutoffs might get mixed up between
      // FSAs
      auto dfsa_acc = dfsavec.scores.Accessor();
      for (int32_t n = 0; n < 10; n++) {
        int32_t i = RandInt(0, dfsavec.scores.Dim0() - 1);
        for (int32_t j = 0; j < dfsavec.scores.Dim1(); j++) {
          dfsa_acc(i, j) += -100.0;
        }
      }
    }

    Array1<int32_t> arc_map_a, arc_map_b;

    FsaVec out_fsas;
    float beam = 1000.0;
    int32_t max_active = 10000, min_active = 0;
    IntersectDensePruned(fsa, dfsavec, beam, beam, min_active, max_active,
                         &out_fsas, &arc_map_a, &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas << ", arc_map_b = " << arc_map_b;

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2;
    Array1<int32_t> arc_map_a2, arc_map_b2;
    // IntersectDensePruned() treats epsilons as normal symbols, so we need to
    // as well.

    ArcSort(&fsa);  // CAUTION if you later test the arc_maps: we arc-sort here,
    // so the input `fsa` is not the same as before.
    bool treat_epsilons_specially = false;
    Intersect(fsa, -1, fsas_b, -1, treat_epsilons_specially,
              &out_fsas2, &arc_map_a2, &arc_map_b2);

    K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
                 << ", arc_map_a2 = " << arc_map_a2
                 << ", arc_map_b2 = " << arc_map_b2;


    { // tests IntersectDevice.
      ContextPtr c = (i == 0 ? GetCpuContext() : GetCudaContext());
      FsaVec fsas = FsaToFsaVec(fsa).To(c);
      Array1<int32_t> b_to_a_map = Range<int32_t>(c, fsas_b.Dim0(), 0,
                                                  (fsas.Dim0() == 1 ? 0 : 1));
      fsas_b = fsas_b.To(c);
      Array1<int32_t> arc_map_a3, arc_map_b3;
      FsaVec out_fsas3 = IntersectDevice(fsas, -1, fsas_b, -1, b_to_a_map,
                                         &arc_map_a3, &arc_map_b3).To(GetCpuContext());


      K2_LOG(INFO) << "out_fsas3 = " << out_fsas3;

      K2_CHECK(
          IsRandEquivalentWrapper(out_fsas2, out_fsas3, treat_epsilons_specially));
    }


    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));
  }
}

TEST(IntersectPruned, RandomFsaVec) {
  for (int32_t i = 0; i < 10; i++) {
    K2_LOG(INFO) << "Iteration of testing: i = " << i;
    int32_t max_symbol = 10, min_num_arcs = 0, max_num_arcs = 200;
    bool acyclic = false;
    ContextPtr c = (i % 2 == 0 ? GetCpuContext() : GetCudaContext()),
               cpu = GetCpuContext();

    int32_t num_b_fsas = RandInt(1, 5),
            num_a_fsas = (RandInt(0, 1) ? 1 : num_b_fsas);

    Fsa fsavec = RandomFsaVec(num_a_fsas, num_a_fsas, acyclic, max_symbol,
                              min_num_arcs, max_num_arcs)
                     .To(c);
    ArcSort(&fsavec);

    int32_t min_frames = 0, max_frames = 10, min_nsymbols = max_symbol + 1,
            max_nsymbols = max_symbol + 4;
    float scores_scale = 1.0;
    DenseFsaVec dfsavec =
        RandomDenseFsaVec(num_b_fsas, num_b_fsas, min_frames, max_frames,
                          min_nsymbols, max_nsymbols, scores_scale);

    if (true) {
      // trying to find bugs where the cutoffs might get mixed up between
      // FSAs
      auto dfsa_acc = dfsavec.scores.Accessor();
      for (int32_t n = 0; n < 10; n++) {
        int32_t i = RandInt(0, dfsavec.scores.Dim0() - 1);
        for (int32_t j = 0; j < dfsavec.scores.Dim1(); j++) {
          dfsa_acc(i, j) += -100.0;
        }
      }
    }
    dfsavec = dfsavec.To(c);

    K2_LOG(INFO) << "fsavec = " << fsavec;

    K2_LOG(INFO) << "dfsavec= " << dfsavec;

    Array1<int32_t> arc_map_a, arc_map_b;

    FsaVec out_fsas;
    float search_beam = 1000.0, output_beam = 1000.0;
    int32_t min_active = 0, max_active = 10;
    IntersectDensePruned(fsavec, dfsavec, search_beam, output_beam, min_active,
                         max_active, &out_fsas, &arc_map_a, &arc_map_b);
    K2_LOG(INFO) << "out_fsas = " << out_fsas
                 << ", arc_map_a = " << arc_map_a
                 << ", arc_map_b = " << arc_map_b;

    out_fsas = out_fsas.To(cpu);
    fsavec = fsavec.To(cpu);
    dfsavec = dfsavec.To(cpu);
    arc_map_a = arc_map_a.To(cpu);
    arc_map_b = arc_map_b.To(cpu);

    {  // check arc map for out_fsas, arc_map_a, arc_map_b
      int32_t num_arcs = out_fsas.NumElements();
      for (int32_t i = 0; i < num_arcs; i++) {
        int32_t arc_idx_a = arc_map_a[i], arc_idx_b = arc_map_b[i];
        float score_a = fsavec.values[arc_idx_a].score,
              score_b = dfsavec.scores.Data()[arc_idx_b],
              score_composed = out_fsas.values[i].score;
        float margin = 1.0e-04 * (fabs(score_a) + fabs(score_b));
        K2_CHECK((score_a + score_b) == score_composed ||
                 fabs(score_a + score_b - score_composed) < margin);
      }
    }

    FsaVec fsas_b = ConvertDenseToFsaVec(dfsavec);
    K2_LOG(INFO) << "fsas_b = " << fsas_b;
    FsaVec out_fsas2;
    Array1<int32_t> arc_map_a2, arc_map_b2;
    // IntersectDensePruned() treats epsilons as normal symbols, so we need to
    // as well.

    ArcSort(&fsavec);  // CAUTION if you later test the arc_maps: we arc-sort
                       // here, so the input `fsa` is not the same as before.
    bool treat_epsilons_specially = false;
    {
      Array1<int32_t> arc_map_a2_temp,
          arc_map_b2_temp;
      FsaVec out_fsas2_temp;
      Intersect(fsavec, -1, fsas_b, -1, treat_epsilons_specially,
                &out_fsas2_temp, &arc_map_a2_temp, &arc_map_b2_temp);
      Array1<int32_t> connect_arc_map;
      Connect(out_fsas2_temp, &out_fsas2, &connect_arc_map);
      arc_map_a2 = arc_map_a2_temp[connect_arc_map];
      arc_map_b2 = arc_map_b2_temp[connect_arc_map];
    }
    K2_LOG(INFO) << "out_fsas2 = " << out_fsas2
                 << ", arc_map_a2 = " << arc_map_a2
                 << ", arc_map_b2 = " << arc_map_b2;
    K2_CHECK(
        IsRandEquivalentWrapper(out_fsas, out_fsas2, treat_epsilons_specially));
  }
}

}  // namespace k2
