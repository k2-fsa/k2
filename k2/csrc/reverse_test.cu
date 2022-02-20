/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey)
 *                2022  ASLP@NWPU          (authors: Hang Lyu)
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

#include <algorithm>
#include <random>

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/math.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(Reverse, SingleFsa) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 0 1 2
      0 1 1 1
      1 2 2 3
      1 2 3 4
      1 4 -1 3
      2 2 4 5
      2 3 4 6
      3 3 6 2
      3 4 -1 2
      4
    )";
    auto fsa = FsaFromString(s).To(c);

    Fsa reversed;
    Array1<int32_t> arc_map;
    Reverse(fsa, &reversed, &arc_map);

    // This initialization is implemented by overloadding the operator>> of
    // struct Ragged<T>.
    Fsa ref = Fsa("[ [ 0 1 0 3 0 3 0 2 ] [ 1 4 1 1 ] "
                  "  [ 2 1 2 3 2 1 3 4 2 2 4 5] [ 3 2 4 6 3 3 6 2 ] "
                  "  [ 4 4 1 2 4 5 -1 0 ] [ ] ]").To(c);
    Array1<int32_t> arc_map_ref(c, "[ 4 8 1 2 3 5 6 7 0 -1 ]");
    K2_CHECK(Equal(reversed, ref));
    K2_CHECK(Equal(arc_map, arc_map_ref));
  }
}

TEST(Reverse, RandomSingleFsa) {
  ContextPtr cpu = GetCpuContext();
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    bool acyclic = RandInt(0, 1);
    Fsa fsa = RandomFsa(acyclic).To(c);

    Fsa reversed;
    Array1<int32_t> arc_map;
    Reverse(fsa, &reversed, &arc_map);

    Array1<Arc> arcs = reversed.values.To(cpu),
                fsa_arcs = fsa.values.To(cpu);
    arc_map = arc_map.To(cpu);
    int32_t num_arcs = arcs.Dim();
    for (int32_t i = 0; i != num_arcs; ++i) {
      if (arc_map[i] != -1)
        EXPECT_EQ(arcs[i].score, fsa_arcs[arc_map[i]].score);
    }
  }
}

TEST(Reverse, FsaVec) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 1 1
      0 2 2 2
      1 3 -1 0
      3
    )";
    auto fsa1 = FsaFromString(s1);

    std::string s2 = R"(0 1 1 1
      1 3 -1 0
      2 1 2 2
      3
    )";
    auto fsa2 = FsaFromString(s2);

    std::string s3 = R"(0 1 1 1
      1 4 -1 0
      1 3 3 3
      2 1 2 2
      4
    )";
    auto fsa3 = FsaFromString(s3);

    Fsa *fsa_array[] = {&fsa1, &fsa2, &fsa3};
    FsaVec fsa_vec = CreateFsaVec(3, &fsa_array[0]).To(c);

    FsaVec reversed;
    Array1<int32_t> arc_map;
    Reverse(fsa_vec, &reversed, &arc_map);

    FsaVec ref = FsaVec("[ [ [ 0 1 0 0 ] [ 1 3 1 1 ] [ 2 3 2 2 ] [ 3 4 -1 0 ] "
                        "    [ ] ] "
                        "  [ [ 0 1 0 0 ] [ 1 3 1 1 1 2 2 2 ] [ ] [ 3 4 -1 0 ] "
                        "    [ ] ] "
                        "  [ [ 0 1 0 0 ] [ 1 4 1 1 1 2 2 2 ] [ ] "
                        "    [ 3 1 3 3 ] [ 4 5 -1 0 ] [ ] ] ]").To(c);
    Array1<int32_t> arc_map_ref(c, "[ 2 0 1 -1 4 3 5 -1 7 6 9 8 -1 ]");

    K2_CHECK(Equal(reversed, ref));
    K2_CHECK(Equal(arc_map, arc_map_ref));
  }
}

TEST(Reverse, EmptyFsaVec) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s1 = R"(0 1 1 1
      0 2 2 2
      1 3 -1 0
      3
    )";
    auto fsa1 = FsaFromString(s1);

    std::string s2 = R"( )";
    auto fsa2 = FsaFromString(s2);

    std::string s3 = R"(0 1 1 1
      1 4 -1 0
      1 3 3 3
      2 1 2 2
      4
    )";
    auto fsa3 = FsaFromString(s3);

    Fsa *fsa_array[] = {&fsa1, &fsa2, &fsa3};
    FsaVec fsa_vec = CreateFsaVec(3, &fsa_array[0]).To(c);

    FsaVec reversed;
    Array1<int32_t> arc_map;
    Reverse(fsa_vec, &reversed, &arc_map);

    FsaVec ref = FsaVec("[ [ [ 0 1 0 0 ] [ 1 3 1 1 ] [ 2 3 2 2 ] [ 3 4 -1 0 ] "
                        "    [ ] ] "
                        "  [ ]  "
                        "  [ [ 0 1 0 0 ] [ 1 4 1 1 1 2 2 2 ] [ ] "
                        "    [ 3 1 3 3 ] [ 4 5 -1 0 ] [ ] ] ]").To(c);
    Array1<int32_t> arc_map_ref(c, "[ 2 0 1 -1 4 3 6 5 -1 ]");

    K2_CHECK(Equal(reversed, ref));
    K2_CHECK(Equal(arc_map, arc_map_ref));
  }
}

TEST(Reverse, RandomFsaVec) {
  ContextPtr cpu = GetCpuContext();
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    bool acyclic = RandInt(0, 1);

    FsaVec fsa_vec = RandomFsaVec(1, 100, acyclic);
    fsa_vec = fsa_vec.To(c);

    FsaVec reversed;
    Array1<int32_t> arc_map;
    Reverse(fsa_vec, &reversed, &arc_map);

    Array1<Arc> arcs = reversed.values.To(cpu),
                fsa_arcs = fsa_vec.values.To(cpu);
    arc_map = arc_map.To(cpu);

    int32_t num_arcs = reversed.TotSize(2);
    for (int32_t i = 0; i != num_arcs; ++i) {
      if (arc_map[i] != -1)
        EXPECT_EQ(arcs[i].score, fsa_arcs[arc_map[i]].score);
    }
  }
}

}  // namespace k2
