/**
 * @brief Unittests for rm_epsilon.cu
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation    (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/math.h"
#include "k2/csrc/rm_epsilon.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

void CheckComputeSubset(FsaVec &src, FsaVec &dst, Array1<int32_t> &state_map,
                        Array1<int32_t> &arc_map, bool epsilon_subset) {
  ContextPtr cpu = GetCpuContext();
  src = src.To(cpu);
  dst = dst.To(cpu);
  state_map = state_map.To(cpu);
  arc_map = arc_map.To(cpu);
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data(),
                *src_row_ids1_data = src.RowIds(1).Data(),
                *src_row_ids2_data = src.RowIds(2).Data(),
                *dst_row_splits1_data = dst.RowSplits(1).Data(),
                *dst_row_ids1_data = dst.RowIds(1).Data(),
                *dst_row_ids2_data = dst.RowIds(2).Data();
  const Arc *src_arcs_data = src.values.Data(),
            *dst_arcs_data = dst.values.Data();
  int32_t src_num_arcs = src.NumElements(), dst_num_arcs = dst.NumElements();
  int32_t expected_dst_num_arcs = 0;
  std::set<int32_t> kept_states;
  // get those kept states except start state and final state
  for (int32_t arc_idx012 = 0; arc_idx012 != src_num_arcs; ++arc_idx012) {
    int32_t fsa_idx0 = src_row_ids1_data[src_row_ids2_data[arc_idx012]];
    int32_t start_state_this_fsa = src_row_splits1_data[fsa_idx0],
            start_state_next_fsa = src_row_splits1_data[fsa_idx0 + 1];
    // push start state and final state of each fsa to kept_states
    if (start_state_next_fsa > start_state_this_fsa) {
      kept_states.insert(start_state_this_fsa);
      kept_states.insert(start_state_next_fsa - 1);
    }
    const Arc &src_arc = src_arcs_data[arc_idx012];
    bool keep = (epsilon_subset ? (src_arc.label == 0) : (src_arc.label != 0));
    if (keep) {
      ++expected_dst_num_arcs;
      // convert state_idx1 to state_idx01 and insert to kept_state
      kept_states.insert(start_state_this_fsa + src_arc.src_state);
      kept_states.insert(start_state_this_fsa + src_arc.dest_state);
    }
  }
  // check kept states, noted we use std::set as the type of kept_states, so
  // there's no need to sort
  std::vector<int32_t> expected_state_map(kept_states.begin(),
                                          kept_states.end());
  CheckArrayData(state_map, expected_state_map);

  // check arcs
  EXPECT_EQ(expected_dst_num_arcs, dst_num_arcs);
  for (int32_t dst_arc_idx012 = 0; dst_arc_idx012 != dst_num_arcs;
       ++dst_arc_idx012) {
    int32_t src_arc_idx012 = arc_map[dst_arc_idx012];
    Arc src_arc = src_arcs_data[src_arc_idx012],
        dst_arc = dst_arcs_data[dst_arc_idx012];
    int32_t fsa_idx0 = src_row_ids1_data[src_row_ids2_data[src_arc_idx012]];
    EXPECT_EQ(fsa_idx0, dst_row_ids1_data[dst_row_ids2_data[dst_arc_idx012]]);
    int32_t src_start_state_this_fsa = src_row_splits1_data[fsa_idx0],
            dst_start_state_this_fsa = dst_row_splits1_data[fsa_idx0];
    src_arc.src_state = src_arc.src_state + src_start_state_this_fsa;
    src_arc.dest_state = src_arc.dest_state + src_start_state_this_fsa;
    dst_arc.src_state = state_map[dst_arc.src_state + dst_start_state_this_fsa];
    dst_arc.dest_state =
        state_map[dst_arc.dest_state + dst_start_state_this_fsa];
    EXPECT_EQ(src_arc, dst_arc);
  }
}

TEST(RmEpsilon, ComputeEpsilonAndNonEpsilonSubsetSimple) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 1 1
    1 2 0 1 
    1 3 2 1 
    2 3 3 1 
    3 4 4 1 
    3 5 5 1 
    4 5 6 1 
    4 6 7 1
    5 6 0 1
    5 7 -1 0
    6 7 -1 0
    7
  )";
    std::string s2 = R"(0 1 0 1
    1 2 0 1 
    2 3 0 1 
    3 4 4 1 
    3 5 -1 1 
    4 5 -1 1
    5
  )";
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    // get epsilon subset
    FsaVec eps_subset;
    Array1<int32_t> eps_state_map, eps_arc_map;
    ComputeEpsilonSubset(fsa_vec, &eps_subset, &eps_state_map, &eps_arc_map);
    EXPECT_EQ(eps_subset.Dim0(), fsa_vec.Dim0());
    {
      std::vector<int32_t> expected_state_map = {0, 1, 2,  5,  6, 7,
                                                 8, 9, 10, 11, 13};
      std::vector<int32_t> expected_arc_map = {1, 8, 11, 12, 13};
      CheckArrayData(eps_state_map, expected_state_map);
      CheckArrayData(eps_arc_map, expected_arc_map);
    }

    // get non-epsilon subset
    FsaVec non_eps_subset;
    Renumbering non_eps_state_map_renumbering;
    Array1<int32_t> non_eps_arc_map;
    ComputeNonEpsilonSubset(fsa_vec, &non_eps_subset,
                            &non_eps_state_map_renumbering, &non_eps_arc_map);
    EXPECT_EQ(non_eps_subset.Dim0(), fsa_vec.Dim0());
    Array1<int32_t> non_eps_state_map = non_eps_state_map_renumbering.New2Old();
    {
      std::vector<int32_t> expected_state_map = {0, 1, 2, 3,  4,  5,
                                                 6, 7, 8, 11, 12, 13};
      std::vector<int32_t> expected_arc_map = {0, 2, 3,  4,  5,  6,
                                               7, 9, 10, 14, 15, 16};
      CheckArrayData(non_eps_state_map, expected_state_map);
      CheckArrayData(non_eps_arc_map, expected_arc_map);
    }
    CheckComputeSubset(fsa_vec, eps_subset, eps_state_map, eps_arc_map, true);
    CheckComputeSubset(fsa_vec, non_eps_subset, non_eps_state_map,
                       non_eps_arc_map, false);
    EXPECT_LE(eps_subset.TotSize(1), fsa_vec.TotSize(1));
    EXPECT_LE(non_eps_subset.TotSize(1), fsa_vec.TotSize(1));
    // eps_subset and non_eps_subset may have duplicate states
    EXPECT_GE(eps_subset.TotSize(1) + non_eps_subset.TotSize(1),
              fsa_vec.TotSize(1));
    // check num_arcs
    EXPECT_EQ(eps_subset.NumElements() + non_eps_subset.NumElements(),
              fsa_vec.NumElements());
  }
}
TEST(RmEpsilon, ComputeEpsilonAndNonEpsilonSubsetRandom) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    for (int32_t i = 0; i != 2; ++i) {
      FsaVec fsa_vec = RandomFsaVec(1, 100, false, 50, 0, 1000);
      // get epsilon subset
      FsaVec eps_subset;
      Array1<int32_t> eps_state_map, eps_arc_map;
      ComputeEpsilonSubset(fsa_vec, &eps_subset, &eps_state_map, &eps_arc_map);
      EXPECT_EQ(eps_subset.Dim0(), fsa_vec.Dim0());
      // get non-epsilon subset
      FsaVec non_eps_subset;
      Renumbering non_eps_state_map_renumbering;
      Array1<int32_t> non_eps_arc_map;
      ComputeNonEpsilonSubset(fsa_vec, &non_eps_subset,
                              &non_eps_state_map_renumbering, &non_eps_arc_map);
      EXPECT_EQ(non_eps_subset.Dim0(), fsa_vec.Dim0());
      Array1<int32_t> non_eps_state_map =
          non_eps_state_map_renumbering.New2Old();
      CheckComputeSubset(fsa_vec, eps_subset, eps_state_map, eps_arc_map, true);
      CheckComputeSubset(fsa_vec, non_eps_subset, non_eps_state_map,
                         non_eps_arc_map, false);
      EXPECT_LE(eps_subset.TotSize(1), fsa_vec.TotSize(1));
      EXPECT_LE(non_eps_subset.TotSize(1), fsa_vec.TotSize(1));
      // we cannot do below CHECK for random cases, as there may be some states
      // in `fsa_vec` who has no any entering or leving arcs, then those states
      // would not occur in either eps_subset or non_eps_subset
      // EXPECT_GE(eps_subset.TotSize(1) + non_eps_subset.TotSize(1),
      // fsa_vec.TotSize(1)); check num_arcs
      EXPECT_EQ(eps_subset.NumElements() + non_eps_subset.NumElements(),
                fsa_vec.NumElements());
    }
  }
}

TEST(RmEpsilon, MapFsaVecStatesSimple) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 1 1
    1 2 0 1 
    1 3 2 1 
    2 3 3 1 
    3 4 4 1 
    3 5 5 1 
    4 5 6 1 
    4 6 -1 1
    5 6 -1 1
    6
  )";
    std::string s2 = R"(0 1 0 1
    1 2 0 1 
    2 3 0 1 
    3 4 4 1 
    3 5 -1 1 
    4 5 -1 1
    5
  )";
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    // dest has 3 fsas
    const std::vector<int32_t> dest_row_splits1_values = {0, 8, 12, 16};
    Array1<int32_t> dest_row_splits1(context, dest_row_splits1_values);
    int32_t dest_num_states = dest_row_splits1_values.back();
    Array1<int32_t> dest_row_ids1(context, dest_num_states);
    RowSplitsToRowIds(dest_row_splits1, &dest_row_ids1);
    // keep state 0, 1, 2, 4, 6 in fsa1, map to 0, 2, 4, 6, 7 in dest;
    // keep state 0, 1, 3 in fsa2, map to 13, 14, 15 in dest
    const std::vector<int32_t> state_map_values = {0,  2,  4,  -1, 6,  -1, 7,
                                                   13, 14, -1, 15, -1, -1};
    Array1<int32_t> state_map(context, state_map_values);

    FsaVec dest;
    Array1<int32_t> arc_map;
    MapFsaVecStates(fsa_vec, dest_row_splits1, dest_row_ids1, state_map, &dest,
                    &arc_map);
    EXPECT_EQ(dest.NumAxes(), 3);
    EXPECT_TRUE(Equal(dest.RowSplits(1), dest_row_splits1));
    std::vector<int32_t> expected_dest_row_ids2 = {0, 2, 6, 13};
    CheckArrayData(dest.RowIds(2), expected_dest_row_ids2);

    EXPECT_EQ(dest.NumElements(), arc_map.Dim());
    std::vector<int32_t> expected_arc_map = {0, 1, 7, 9};
    CheckArrayData(arc_map, expected_arc_map);
    K2_LOG(INFO) << dest;
    K2_LOG(INFO) << arc_map;
  }
  // TODO(haowen): add random tests
}
TEST(RmEpsilon, ComputeEpsilonClosureOneIterSimple) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 0 1
    1 2 0 1 
    1 3 0 1 
    2 3 0 1 
    3 4 0 1 
    3 5 0 1 
    4 5 0 1 
    6
  )";
    std::string s2 = R"(0 1 0 1
    1 2 0 1 
    2 3 0 1 
    3 4 0 1 
    5
  )";
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    FsaVec dest;
    Ragged<int32_t> arc_map;
    ComputeEpsilonClosureOneIter(fsa_vec, &dest, &arc_map);
    EXPECT_EQ(dest.NumAxes(), 3);
    EXPECT_EQ(arc_map.NumAxes(), 2);
    K2_LOG(INFO) << dest;
    K2_LOG(INFO) << arc_map;
    EXPECT_EQ(dest.NumElements(), 20);
    EXPECT_EQ(arc_map.Dim0(), 20);
  }
  // TODO(haowen): add random tests
}

TEST(RmEpsilon, ComputeEpsilonClosureSimple) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 0 1
    1 2 0 1 
    1 3 0 1 
    2 3 0 1 
    3 4 0 1 
    3 5 0 1 
    4 5 0 1 
    6
  )";
    std::string s2 = R"(0 1 0 1
    1 2 0 1 
    2 3 0 1 
    3 4 0 1 
    5
  )";
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);
    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    FsaVec dest;
    Ragged<int32_t> arc_map;
    ComputeEpsilonClosure(fsa_vec, &dest, &arc_map);
    EXPECT_EQ(dest.NumAxes(), 3);
    EXPECT_EQ(arc_map.NumAxes(), 2);
    K2_LOG(INFO) << dest;
    K2_LOG(INFO) << arc_map;
    EXPECT_EQ(dest.NumElements(), 25);
    EXPECT_EQ(arc_map.Dim0(), 25);
  }
  // TODO(haowen): add random tests
}

// arc_map is arc_map from dest to src
void CheckArcMap(FsaVec &src, FsaVec &dest, Ragged<int32_t> &arc_map) {
  ContextPtr cpu = GetCpuContext();
  src = src.To(cpu);
  dest = dest.To(cpu);
  arc_map = arc_map.To(cpu);
  int32_t dest_num_arcs = dest.NumElements();
  ASSERT_EQ(arc_map.NumAxes(), 2);
  ASSERT_EQ(dest_num_arcs, arc_map.Dim0());
  const Arc *src_arcs = src.values.Data();
  const Arc *dest_arcs = dest.values.Data();
  const int32_t *arc_map_row_splits = arc_map.RowSplits(1).Data(),
                *arc_map_values = arc_map.values.Data();
  for (int32_t i = 0; i != dest_num_arcs; ++i) {
    float score = 0.0;
    for (int32_t j = arc_map_row_splits[i]; j != arc_map_row_splits[i + 1];
         ++j) {
      score += src_arcs[arc_map_values[j]].score;
    }
    float dest_arc_score = dest_arcs[i].score;
    EXPECT_NEAR(dest_arc_score, score, 0.01);
  }
}

TEST(RmEpsilon, RemoveEpsilonsIterativeTropicalSimple) {
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 1 1
    1 2 0 1 
    1 3 2 1 
    2 3 3 1 
    3 4 4 1 
    3 5 5 1 
    4 5 6 1 
    4 6 7 1
    5 6 0 1
    5 7 -1 0
    6 7 -1 0
    7
  )";
    std::string s2 = R"(0 1 0 1
    1 2 0 1 
    2 3 0 1 
    3 4 4 1 
    3 5 -1 1 
    4 5 -1 1
    5
  )";
    {
      // test with single Fsa
      Fsa fsa = FsaFromString(s2);
      fsa = fsa.To(context);

      FsaVec dest;
      Ragged<int32_t> arc_map;
      RemoveEpsilonsIterativeTropical(fsa, &dest, &arc_map);
      EXPECT_EQ(dest.NumAxes(), 2);
      EXPECT_EQ(arc_map.NumAxes(), 2);
      K2_LOG(INFO) << dest;
      K2_LOG(INFO) << arc_map;
      int32_t p = GetFsaBasicProperties(dest);
      EXPECT_EQ(p & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
      bool log_semiring = false;
      float beam = std::numeric_limits<float>::infinity();
      fsa = fsa.To(GetCpuContext());
      dest = dest.To(GetCpuContext());
      EXPECT_TRUE(IsRandEquivalent(fsa, dest, log_semiring, beam, true, 0.001));
      CheckArcMap(fsa, dest, arc_map);
    }
    {
      // test with FsaVec
      Fsa fsa1 = FsaFromString(s1);
      Fsa fsa2 = FsaFromString(s2);
      Fsa *fsa_array[] = {&fsa1, &fsa2};
      FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
      fsa_vec = fsa_vec.To(context);

      FsaVec dest;
      Ragged<int32_t> arc_map;
      RemoveEpsilonsIterativeTropical(fsa_vec, &dest, &arc_map);
      EXPECT_EQ(dest.NumAxes(), 3);
      EXPECT_EQ(arc_map.NumAxes(), 2);
      K2_LOG(INFO) << dest;
      K2_LOG(INFO) << arc_map;
      Array1<int32_t> properties;
      int32_t p;
      GetFsaVecBasicProperties(dest, &properties, &p);
      EXPECT_EQ(p & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
      bool log_semiring = false;
      float beam = std::numeric_limits<float>::infinity();
      fsa_vec = fsa_vec.To(GetCpuContext());
      dest = dest.To(GetCpuContext());
      EXPECT_TRUE(
          IsRandEquivalent(fsa_vec, dest, log_semiring, beam, true, 0.001));
      CheckArcMap(fsa_vec, dest, arc_map);
    }
  }
}

TEST(RmEpsilon, TestRemoveEpsilonsIterativeTropicalWithRandomTopSortedFsa) {
  for (int32_t i = 0; i != 1; ++i) {
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      int32_t min_num_fsas = 1;
      int32_t max_num_fsas = 1000;
      bool acyclic = true;
      // set max_symbol=10 so that we have a high probability
      // to create Fsas with epsilon arcs.
      int32_t max_symbol = 10;
      int32_t min_num_arcs = 0;
      int32_t max_num_arcs = 10000;
      FsaVec fsa_vec = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic,
                                    max_symbol, min_num_arcs, max_num_arcs);
      fsa_vec = fsa_vec.To(context);

      FsaVec dest;
      Ragged<int32_t> arc_map;
      RemoveEpsilonsIterativeTropical(fsa_vec, &dest, &arc_map);
      EXPECT_EQ(dest.NumAxes(), 3);
      EXPECT_EQ(arc_map.NumAxes(), 2);
      Array1<int32_t> properties;
      int32_t p;
      GetFsaVecBasicProperties(dest, &properties, &p);
      EXPECT_EQ(p & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
      bool log_semiring = false;
      float beam = std::numeric_limits<float>::infinity();
      fsa_vec = fsa_vec.To(GetCpuContext());
      dest = dest.To(GetCpuContext());
      EXPECT_TRUE(
          IsRandEquivalent(fsa_vec, dest, log_semiring, beam, true, 0.1));
      CheckArcMap(fsa_vec, dest, arc_map);
    }
  }
}

TEST(RmEpsilon, TestRemoveEpsilonsIterativeTropicalWithRandomNonTopSortedFsa) {
  for (int32_t i = 0; i != 1; ++i) {
    for (auto &context : {GetCpuContext(), GetCudaContext()}) {
      int32_t min_num_fsas = 1;
      int32_t max_num_fsas = 1000;
      bool acyclic = false;
      // set max_symbol=10 so that we have a high probability
      // to create Fsas with epsilon arcs.
      int32_t max_symbol = 10;
      int32_t min_num_arcs = 0;
      int32_t max_num_arcs = 10000;
      FsaVec fsa_vec = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic,
                                    max_symbol, min_num_arcs, max_num_arcs);
      // convert arcs' scores to negative as we don't allow positive epsilon
      // cycles in `RemoveEpsilonIterativeTropical`.
      Arc *fsa_vec_arcs_data = fsa_vec.values.Data();
      for (int32_t n = 0; n != fsa_vec.NumElements(); ++n) {
        Arc &cur_arc = fsa_vec_arcs_data[n];
        if (cur_arc.score > 0) cur_arc.score = -cur_arc.score;
      }
      fsa_vec = fsa_vec.To(context);

      FsaVec dest;
      Ragged<int32_t> arc_map;
      RemoveEpsilonsIterativeTropical(fsa_vec, &dest, &arc_map);
      EXPECT_EQ(dest.NumAxes(), 3);
      EXPECT_EQ(arc_map.NumAxes(), 2);
      Array1<int32_t> properties;
      int32_t p;
      GetFsaVecBasicProperties(dest, &properties, &p);
      EXPECT_EQ(p & kFsaPropertiesEpsilonFree, kFsaPropertiesEpsilonFree);
      fsa_vec = fsa_vec.To(GetCpuContext());
      dest = dest.To(GetCpuContext());
      EXPECT_TRUE(IsRandEquivalentUnweighted(fsa_vec, dest, true));
      CheckArcMap(fsa_vec, dest, arc_map);
      // TODO(haowen): how to check if the weights of sequences are correct?
    }
  }
}

}  // namespace k2
