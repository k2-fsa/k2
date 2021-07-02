/**
 * Copyright      2021  Xiaomi Corporation (authors: Wei Kang)
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

TEST(Connect, SingleFsa) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 2 1 1
      0 3 3 3
      1 4 5 5
      1 6 -1 0
      2 1 2 2
      3 1 4 4
      5 3 6 6
      6
    )";
    auto fsa = FsaFromString(s).To(c);
    int32_t gt = kFsaPropertiesMaybeAccessible |
                 kFsaPropertiesMaybeCoaccessible;
    int32_t p = GetFsaBasicProperties(fsa);
    EXPECT_NE(p & gt, gt);

    Fsa connected;
    Array1<int32_t> arc_map;
    Connect(fsa, &connected, &arc_map);

    Fsa ref = Fsa("[ [ 0 2 1 1 0 3 3 3 ] [ 1 4 -1 0 ] "
                  "  [ 2 1 2 2 ] [ 3 1 4 4 ] [ ] ]").To(c);
    Array1<int32_t> arc_map_ref(c, "[ 0 1 3 4 5 ]");
    K2_CHECK(Equal(connected, ref));
    K2_CHECK(Equal(arc_map, arc_map_ref));
    p = GetFsaBasicProperties(connected);
    EXPECT_EQ(p & gt, gt);
  }
}

TEST(Connect, CyclicFsa) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 1 1
      0 2 2 2
      1 2 3 3
      2 3 4 4
      2 4 5 5
      3 1 6 6
      3 6 -1 0
      5 2 7 7
      6
    )";
    auto fsa = FsaFromString(s).To(c);

    int32_t gt = kFsaPropertiesMaybeAccessible |
                 kFsaPropertiesMaybeCoaccessible;
    int32_t p = GetFsaBasicProperties(fsa);
    EXPECT_NE(p & gt, gt);

    Fsa connected;
    Array1<int32_t> arc_map;
    Connect(fsa, &connected, &arc_map);
    Fsa ref = Fsa("[ [ 0 1 1 1 0 2 2 2 ] [ 1 2 3 3 ] [ 2 3 4 4] "
                  "  [ 3 1 6 6 3 4 -1 0 ] [ ] ]").To(c);
    Array1<int32_t> arc_map_ref(c, "[ 0 1 2 3 5 6 ]");
    K2_CHECK(Equal(connected, ref));
    K2_CHECK(Equal(arc_map, arc_map_ref));
    p = GetFsaBasicProperties(connected);
    EXPECT_EQ(p & gt, gt);
  }
}

TEST(Connect, RandomSingleFsa) {
  ContextPtr cpu = GetCpuContext();
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    bool acyclic = RandInt(0, 1);
    Fsa fsa = RandomFsa(acyclic).To(c);
    int32_t gt = kFsaPropertiesMaybeAccessible |
                 kFsaPropertiesMaybeCoaccessible;

    Fsa connected;
    Array1<int32_t> arc_map;
    Connect(fsa, &connected, &arc_map);
    int32_t p = GetFsaBasicProperties(connected);
    EXPECT_EQ(p & gt, gt);

    Array1<Arc> arcs = connected.values.To(cpu),
                fsa_arcs = fsa.values.To(cpu);
    arc_map = arc_map.To(cpu);
    int32_t num_arcs = arcs.Dim();
    for (int32_t i = 0; i != num_arcs; ++i) {
      EXPECT_EQ(arcs[i].score, fsa_arcs[arc_map[i]].score);
    }
  }
}

TEST(Connect, FsaVec) {
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

    int32_t gt = kFsaPropertiesMaybeAccessible |
                 kFsaPropertiesMaybeCoaccessible;
    Array1<int32_t> properties;
    int32_t p;
    GetFsaVecBasicProperties(fsa_vec, &properties, &p);

    EXPECT_NE(p & gt, gt);
    EXPECT_NE(properties[0] & gt, gt);
    EXPECT_NE(properties[1] & gt, gt);
    EXPECT_NE(properties[2] & gt, gt);

    FsaVec connected;
    Array1<int32_t> arc_map;
    Connect(fsa_vec, &connected, &arc_map);
    FsaVec ref = FsaVec("[ [ [ 0 1 1 1 ] [ 1 2 -1 0 ] [ ] ] "
                        "  [ [ 0 1 1 1 ] [ 1 2 -1 0 ] [ ] ] "
                        "  [ [ 0 1 1 1 ] [ 1 2 -1 0 ] [ ] ] ]").To(c);
    Array1<int32_t> arc_map_ref(c, "[ 0 2 3 4 6 7 ]");
    K2_CHECK(Equal(connected, ref));
    K2_CHECK(Equal(arc_map, arc_map_ref));

    GetFsaVecBasicProperties(connected, &properties, &p);
    EXPECT_EQ(p & gt, gt);
    EXPECT_EQ(properties[0] & gt, gt);
    EXPECT_EQ(properties[1] & gt, gt);
    EXPECT_EQ(properties[2] & gt, gt);
  }
}

TEST(Connect, RandomFsaVec) {
  ContextPtr cpu = GetCpuContext();
  for (auto &c : {GetCpuContext(), GetCudaContext()}) {
    bool acyclic = RandInt(0, 1);

    FsaVec fsa_vec = RandomFsaVec(1, 100, acyclic);
    fsa_vec = fsa_vec.To(c);

    int32_t gt = kFsaPropertiesMaybeAccessible |
                 kFsaPropertiesMaybeCoaccessible;
    Array1<int32_t> properties;
    int32_t p;

    FsaVec connected;
    Array1<int32_t> arc_map;
    Connect(fsa_vec, &connected, &arc_map);

    GetFsaVecBasicProperties(connected, &properties, &p);

    EXPECT_EQ(p & gt, gt);
    properties = properties.To(cpu);
    int32_t num_fsas = fsa_vec.Dim0();
    for (int32_t i = 0; i != num_fsas; ++i) {
      EXPECT_EQ(properties[i] & gt, gt);
    }

    Array1<Arc> arcs = connected.values.To(cpu),
                fsa_arcs = fsa_vec.values.To(cpu);
    arc_map = arc_map.To(cpu);

    int32_t num_arcs = connected.TotSize(2);
    for (int32_t i = 0; i != num_arcs; ++i) {
      EXPECT_EQ(arcs[i].score, fsa_arcs[arc_map[i]].score);
    }
  }
}

}  // namespace k2
