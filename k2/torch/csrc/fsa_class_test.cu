/**
 * Copyright (c)  2021  Xiaomi Corporation (authors: Wei Kang)
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

#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

TEST(FsaClassTest, FromUnaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 2 10
        0 1 1 20
        1 2 -1 30
        2)";

    auto device = DeviceFromContext(c);
    Fsa fsa = FsaFromString(s).To(c);
    FsaClass src = FsaClass(fsa);

    src.SetAttr(
        "float_attr",
        torch::IValue(torch::tensor(
            {0.1, 0.2, 0.3}, torch::dtype(torch::kFloat32).device(device))));

    src.SetAttr("int_attr",
                torch::IValue(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    Ragged<int32_t> ragged_attr(c, "[[1 2 3] [5 6] []]");

    src.SetAttr("ragged_attr", ToIValue(ragged_attr));

    Array1<int32_t> arc_map;
    Ragged<Arc> arcs;
    ArcSort(src.fsa, &arcs, &arc_map);
    auto dest = FsaClass::FromUnaryFunctionTensor(
        src, arcs, Array1ToTorch<int32_t>(arc_map));

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").toTensor(),
        torch::tensor({0.2, 0.1, 0.3},
                      torch::dtype(torch::kFloat32).device(device))));

    EXPECT_TRUE(torch::allclose(
        dest.Scores(),
        torch::tensor({20, 10, 30},
                      torch::dtype(torch::kFloat32).device(device))));

    EXPECT_TRUE(torch::equal(
        dest.GetAttr("int_attr").toTensor(),
        torch::tensor({2, 1, 3}, torch::dtype(torch::kInt32).device(device))));

    Ragged<int32_t> expected_ragged_attr =
        Ragged<int32_t>(c, "[[5 6] [1 2 3] []]");

    EXPECT_TRUE(
        Equal(ToRaggedInt(dest.GetAttr("ragged_attr")), expected_ragged_attr));
  }
}

TEST(FsaClassTest, Attributes) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = DeviceFromContext(c);
    std::string s = R"(0 1 2 10
        0 1 1 20
        1 2 -1 30
        2)";
    Fsa fsa = FsaFromString(s).To(c);
    FsaClass src = FsaClass(fsa);

    // test scores
    EXPECT_TRUE(torch::equal(
        src.Scores(),
        torch::tensor({10, 20, 30},
                      torch::dtype(torch::kFloat32).device(device))));
    torch::Tensor scores =
        torch::tensor({1, 2, 3}, torch::dtype(torch::kFloat32).device(device));
    src.SetScores(scores);
    EXPECT_TRUE(torch::equal(src.Scores(), scores));

    // test labels
    EXPECT_TRUE(torch::equal(
        src.Labels(),
        torch::tensor({2, 1, -1}, torch::dtype(torch::kInt32).device(device))));
    torch::Tensor labels =
        torch::tensor({20, 10, -1}, torch::dtype(torch::kInt32).device(device));
    src.SetLabels(labels);
    EXPECT_TRUE(torch::equal(src.Labels(), labels));

    // test tensor attribute
    torch::Tensor tensor_int =
        torch::tensor({1, 2, 3}, torch::dtype(torch::kInt32).device(device));
    src.SetAttr("tensor_int", torch::IValue(tensor_int));
    torch::Tensor tensor_float =
        torch::tensor({1, 2, 3}, torch::dtype(torch::kFloat32).device(device));
    src.SetAttr("tensor_float", torch::IValue(tensor_float));

    EXPECT_TRUE(torch::equal(src.GetAttr("tensor_int").toTensor(), tensor_int));
    EXPECT_TRUE(
        torch::allclose(src.GetAttr("tensor_float").toTensor(), tensor_float));
    src.DeleteAttr("tensor_int");
    EXPECT_FALSE(src.HasAttr("tensor_int"));

    // test ragged attribute
    auto ragged_int = Ragged<int32_t>(c, "[[1, 2], [3], [4]]");
    src.SetAttr("ragged_int", ToIValue(ragged_int));

    EXPECT_TRUE(Equal(ToRaggedInt(src.GetAttr("ragged_int")), ragged_int));
    src.DeleteAttr("ragged_int");
    EXPECT_FALSE(src.HasAttr("ragged_int"));
  }
}

}  // namespace k2
