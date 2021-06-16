/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  Fsa fsa = FsaFromString(s, false);
  k2host::Fsa host_fsa = FsaToHostFsa(fsa);
  K2_LOG(INFO) << k2host::FsaToString(host_fsa);
  // TODO(fangjun): check the content of host_fsa
}


TEST(HostShim, IsRandEquivalent) {
  // check that empty FSAs with zero vs 2 states are equivalent.
  FsaVec f("[ [ [] [] ] [] [] ]"),
      g("[ [ [] [] ] [ [] [] ] [ [] [] ] ]");
  EXPECT_EQ(f.NumAxes(), 3);
  EXPECT_EQ(g.NumAxes(), 3);

  EXPECT_EQ(IsRandEquivalent(f, g, true), true);
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


}  // namespace k2
