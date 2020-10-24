/**
 * @brief Unittest for host shim.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <string>

#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/test_utils.h"

namespace k2 {
TEST(HostShim, FsaToHostFsa) {
  std::string s = R"( 0 1 1 1
    1 2 2 2
    2 3 -1 0
    3
  )";
  Fsa fsa = FsaFromString(s);
  k2host::Fsa host_fsa = FsaToHostFsa(fsa);
  K2_LOG(INFO) << k2host::FsaToString(host_fsa);
  // TODO(fangjun): check the content of host_fsa
}

TEST(HostShim, FsaVecToHostFsa) {
  std::string s1 = R"( 0 1 1 1
    1 2 2 2
    2 3 -1 0
    3
  )";

  std::string s2 = R"( 0 1 1 10
    1 2 2 20
    2 3 -1 0
    3
  )";
  Fsa fsa1 = FsaFromString(s1);
  Fsa fsa2 = FsaFromString(s2);
  Fsa *fsa_array[] = {&fsa1, &fsa2};
  FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);

  k2host::Fsa host_fsa = FsaVecToHostFsa(fsa_vec, 0);
  K2_LOG(INFO) << k2host::FsaToString(host_fsa);
  // TODO(fangjun): check the content of host_fsa

  host_fsa = FsaVecToHostFsa(fsa_vec, 1);
  K2_LOG(INFO) << fsa_vec.values;
  K2_LOG(INFO) << k2host::FsaToString(host_fsa);
  // TODO(fangjun): check the content of host_fsa
}

TEST(HostShim, ShortestDistance) {
  std::string s = R"(0 4 1 1
    0 1 1 1
    1 2 1 2
    1 3 1 3
    2 7 1 4
    3 7 1 5
    4 6 1 2
    4 8 1 3
    5 9 -1 4
    6 9 -1 3
    7 9 -1 5
    8 9 -1 6
    9
  )";

  Fsa fsa = FsaFromString(s);
  Array1<int32_t> arc_indexes;
  double d = ShortestDistance(fsa, &arc_indexes);

  EXPECT_NEAR(d, 14, 1e-8);
  CheckArrayData(arc_indexes, std::vector<int32_t>{1, 3, 5, 10});
}

}  // namespace k2
