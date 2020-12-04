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

#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/math.h"
#include "k2/csrc/rm_epsilon.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(RmEpsilon, ComputeEpsilonSubset) {
  // tests single FSA and also 2 copies of a single FSA.
  // for (auto &context : {GetCpuContext(), GetCudaContext()}) {
  for (auto &context : {GetCpuContext()}) {
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
    std::string s2 = R"(0 1 1 1
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

    FsaVec dest;
    Array1<int32_t> state_map, arc_map;
    ComputeEpsilonSubset(fsa_vec, &dest, &state_map, &arc_map);
    // TODO(haowen): test together with ComputeNonEpsilonSubset so we can assert
    // the result automatically
    K2_LOG(INFO) << dest;
    K2_LOG(INFO) << state_map;
    K2_LOG(INFO) << arc_map;
  }

  // TODO(haowen): add random tests
}

}  // namespace k2
