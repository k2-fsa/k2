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

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

TEST(FsaPropertiesAsString, Empty) {
  auto s = FsaPropertiesAsString(0);
  EXPECT_EQ(s, "\"\"");
}

TEST(FsaPropertiesAsString, NonEmpty) {
  auto s = FsaPropertiesAsString(kFsaPropertiesValid);
  EXPECT_EQ(s, "\"Valid\"");

  s = FsaPropertiesAsString(kFsaPropertiesNonempty | kFsaPropertiesValid);
  EXPECT_EQ(s, "\"Valid|Nonempty\"");

  s = FsaPropertiesAsString(kFsaPropertiesTopSorted | kFsaPropertiesValid);
  EXPECT_EQ(s, "\"Valid|TopSorted\"");
}

TEST(FsaIO, FromAndToTensor) {
  // src_state dst_state label cost
  std::string s = R"(0 1 1 1
    0 2 2 2
    1 3 -1 1
    1 2 2 2
    2 3 -1 3
    3
  )";
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa = FsaFromString(s, false, 0, nullptr);
    fsa = fsa.To(context);
    Tensor tensor = FsaToTensor(fsa);
    int32_t num_arcs = fsa.values.Dim();
    ASSERT_EQ(tensor.NumAxes(), 2);
    EXPECT_EQ(tensor.GetDtype(), kInt32Dtype);
    EXPECT_EQ(tensor.Dim(0), num_arcs);
    EXPECT_EQ(tensor.Dim(1), 4);
    EXPECT_TRUE(tensor.IsContiguous());
    EXPECT_TRUE(IsCompatible(fsa, tensor));

    bool error = true;
    Fsa f = FsaFromTensor(tensor, &error);
    ASSERT_FALSE(error);
    EXPECT_TRUE(IsCompatible(f, tensor));
    f = f.To(GetCpuContext());  // for testing only
    EXPECT_EQ((f[{0, 0}]), (Arc{0, 1, 1, 1.f}));
    EXPECT_EQ((f[{0, 1}]), (Arc{0, 2, 2, 2.f}));
    EXPECT_EQ((f[{0, 2}]), (Arc{1, 3, -1, 1.f}));
    EXPECT_EQ((f[{0, 3}]), (Arc{1, 2, 2, 2.f}));
    EXPECT_EQ((f[{0, 4}]), (Arc{2, 3, -1, 3.f}));
  }
}

TEST(FsaVecIO, FromAndToTensor) {
  // src_state dst_state label cost
  std::string s1 = R"(0 1 1 1
    0 2 2 2
    1 3 -1 1
    1 2 2 2
    2 3 -1 3
    3
  )";

  std::string s2 = R"(0 1 1 1.5
    1 2 2 2.5
    2 3 -1 3.5
    3
  )";
  for (auto &context : {GetCpuContext(), GetCudaContext()}) {
    Fsa fsa1 = FsaFromString(s1);
    Fsa fsa2 = FsaFromString(s2);

    Fsa *fsa_array[] = {&fsa1, &fsa2};
    FsaVec fsa_vec = CreateFsaVec(2, &fsa_array[0]);
    fsa_vec = fsa_vec.To(context);

    Tensor tensor = FsaVecToTensor(fsa_vec);
    EXPECT_EQ(tensor.NumAxes(), 1);
    EXPECT_EQ(tensor.GetDtype(), kInt32Dtype);
    EXPECT_TRUE(IsCompatible(fsa_vec, tensor));

    bool error = true;
    FsaVec f = FsaVecFromTensor(tensor, &error);
    ASSERT_FALSE(error);
    EXPECT_TRUE(IsCompatible(f, tensor));
    f = f.To(GetCpuContext());  // for testing

    EXPECT_EQ((f[{0, 0, 0}]), (Arc{0, 1, 1, 1.f}));
    EXPECT_EQ((f[{0, 0, 1}]), (Arc{0, 2, 2, 2.f}));
    EXPECT_EQ((f[{0, 0, 2}]), (Arc{1, 3, -1, 1.f}));
    EXPECT_EQ((f[{0, 0, 3}]), (Arc{1, 2, 2, 2.f}));
    EXPECT_EQ((f[{0, 0, 4}]), (Arc{2, 3, -1, 3.f}));

    EXPECT_EQ((f[{1, 0, 0}]), (Arc{0, 1, 1, 1.5f}));
    EXPECT_EQ((f[{1, 0, 1}]), (Arc{1, 2, 2, 2.5f}));
    EXPECT_EQ((f[{1, 0, 2}]), (Arc{2, 3, -1, 3.5f}));
  }
}

}  // namespace k2
