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

TEST(FsaClassTest, FromUnaryFunctionRagged) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = DeviceFromContext(c);
    std::string s = R"(0 1 0 0
        0 1 1 0
        1 2 -1 0
        2)";
    torch::Tensor scores =
        torch::tensor({1, 2, 3}, torch::dtype(torch::kFloat32).device(device));

    Fsa fsa = FsaFromString(s).To(c);
    FsaClass src = FsaClass(fsa);
    src.SetScores(scores);

    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3}, torch::dtype(torch::kFloat32).device(device));

    src.SetAttr("float_attr", torch::IValue(float_attr));

    src.SetAttr("int_attr",
                torch::IValue(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    auto ragged_int = Ragged<int32_t>(c, "[[10 20] [30 40 50] [60 70]]");
    src.SetAttr("ragged_attr", ToIValue(ragged_int));

    Ragged<int32_t> arc_map;
    Ragged<Arc> arcs;
    RemoveEpsilon(src.fsa, src.Properties(), &arcs, &arc_map);

    FsaClass dest = FsaClass::FromUnaryFunctionRagged(src, arcs, arc_map);

    Ragged<int32_t> expected_arc_map = Ragged<int32_t>(c, "[[1] [0 2] [2]]");

    EXPECT_TRUE(Equal(arc_map, expected_arc_map));

    Ragged<int32_t> expected_int_attr = Ragged<int32_t>(c, "[[2] [1 3] [3]]");
    EXPECT_TRUE(
        Equal(ToRaggedInt(dest.GetAttr("int_attr")), expected_int_attr));

    Ragged<int32_t> expected_ragged_attr =
        Ragged<int32_t>(c, "[[30 40 50] [10 20 60 70] [60 70]]");

    EXPECT_TRUE(
        Equal(ToRaggedInt(dest.GetAttr("ragged_attr")), expected_ragged_attr));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").toTensor());
    expected_float_attr[0] = float_attr[1];
    expected_float_attr[1] = float_attr[0] + float_attr[2];
    expected_float_attr[2] = float_attr[2];

    EXPECT_TRUE(torch::allclose(dest.GetAttr("float_attr").toTensor(),
                                expected_float_attr));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = scores[1];
    expected_scores[1] = scores[0] + scores[2];
    expected_scores[2] = scores[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));
  }
}

TEST(FsaClassTest, FromUnaryFunctionRaggedWithEmptyList) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = DeviceFromContext(c);
    std::string s = R"(0 1 0 0
        0 1 1 0
        1 2 -1 0
        2)";

    torch::Tensor scores =
        torch::tensor({1, 2, 3}, torch::dtype(torch::kFloat32).device(device));

    Fsa fsa = FsaFromString(s).To(c);
    FsaClass src = FsaClass(fsa);
    src.SetScores(scores);

    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3}, torch::dtype(torch::kFloat32).device(device));

    src.SetAttr("float_attr", torch::IValue(float_attr));

    src.SetAttr("int_attr",
                torch::IValue(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    auto ragged_attr_int = Ragged<int32_t>(c, "[[10 20] [30 40 50] [60 70]]");
    src.SetAttr("ragged_attr", ToIValue(ragged_attr_int));

    Ragged<int32_t> arc_map;
    Ragged<Arc> arcs;
    RemoveEpsilonAndAddSelfLoops(src.fsa, src.Properties(), &arcs, &arc_map);

    FsaClass dest = FsaClass::FromUnaryFunctionRagged(src, arcs, arc_map);

    auto expected_arc_map = Ragged<int32_t>(c, "[[] [1] [0 2] [] [2]]");

    EXPECT_TRUE(Equal(arc_map, expected_arc_map));

    auto expected_int_attr = Ragged<int32_t>(c, "[[] [2] [1 3] [] [3]]");

    EXPECT_TRUE(
        Equal(ToRaggedInt(dest.GetAttr("int_attr")), expected_int_attr));

    auto expected_ragged_attr =
        Ragged<int32_t>(c, "[[] [30 40 50] [10 20 60 70] [] [60 70]]");

    EXPECT_TRUE(
        Equal(ToRaggedInt(dest.GetAttr("ragged_attr")), expected_ragged_attr));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").toTensor());
    expected_float_attr[0] = 0;
    expected_float_attr[1] = float_attr[1];
    expected_float_attr[2] = float_attr[0] + float_attr[2];
    expected_float_attr[3] = 0;
    expected_float_attr[4] = float_attr[2];

    EXPECT_TRUE(torch::allclose(dest.GetAttr("float_attr").toTensor(),
                                expected_float_attr));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = 0;
    expected_scores[1] = scores[1];
    expected_scores[2] = scores[0] + scores[2];
    expected_scores[3] = 0;
    expected_scores[4] = scores[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));
  }
}

TEST(FsaClassTest, FromBinaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = DeviceFromContext(c);
    std::string s1 = R"(0 1 0 0.1
        0 1 1 0.2
        1 1 2 0.3
        1 2 -1 0.4
        2)";
    std::vector<Fsa *> a_fsas;
    Fsa a_fsa_tmp = FsaFromString(s1);
    a_fsas.emplace_back(&a_fsa_tmp);
    FsaVec fsa_vec_a = CreateFsaVec(a_fsas.size(), a_fsas.data()).To(c);
    FsaClass a_fsa = FsaClass(fsa_vec_a);

    torch::Tensor scores_a = torch::tensor(
        {0.1, 0.2, 0.3, 0.4}, torch::dtype(torch::kFloat32).device(device));

    torch::Tensor a_float_attr = torch::tensor(
        {1, 2, 3, 4}, torch::dtype(torch::kFloat32).device(device));

    a_fsa.SetAttr("float_attr_a", torch::IValue(a_float_attr));

    a_fsa.SetAttr("float_attr", torch::IValue(a_float_attr.clone()));

    auto ragged_attr_a_int =
        Ragged<int32_t>(c, "[[10 20] [30 40 50] [60 70] [80]]");
    a_fsa.SetAttr("ragged_attr_a", ToIValue(ragged_attr_a_int));

    std::string s2 = R"(0 1 1 1
        0 1 2 2
        1 2 -1 3
        2)";

    std::vector<Fsa *> b_fsas;
    Fsa b_fsa_tmp = FsaFromString(s2);
    b_fsas.emplace_back(&b_fsa_tmp);
    FsaVec fsa_vec_b = CreateFsaVec(b_fsas.size(), b_fsas.data()).To(c);
    FsaClass b_fsa = FsaClass(fsa_vec_b);

    torch::Tensor scores_b =
        torch::tensor({1, 2, 3}, torch::dtype(torch::kFloat32).device(device));

    torch::Tensor b_float_attr = torch::tensor(
        {0.1, 0.2, 0.3}, torch::dtype(torch::kFloat32).device(device));

    b_fsa.SetAttr("float_attr", torch::IValue(b_float_attr.clone()));
    b_fsa.SetAttr("float_attr_b", torch::IValue(b_float_attr));

    auto ragged_attr_b_int = Ragged<int32_t>(c, "[[10 20] [30 40 50] [60 70]]");
    b_fsa.SetAttr("ragged_attr_b", ToIValue(ragged_attr_b_int));

    Array1<int32_t> a_arc_map_raw;
    Array1<int32_t> b_arc_map_raw;
    Array1<int32_t> b_to_a_map(c, "[0]");
    Ragged<Arc> arcs = IntersectDevice(a_fsa.fsa, a_fsa.Properties(), b_fsa.fsa,
                                       b_fsa.Properties(), b_to_a_map,
                                       &a_arc_map_raw, &b_arc_map_raw, true);

    torch::Tensor a_arc_map = Array1ToTorch<int32_t>(a_arc_map_raw);
    torch::Tensor b_arc_map = Array1ToTorch<int32_t>(b_arc_map_raw);

    FsaClass dest = FsaClass::FromBinaryFunctionTensor(a_fsa, b_fsa, arcs,
                                                       a_arc_map, b_arc_map);

    torch::Tensor expected_a_arc_map =
        torch::tensor({1, 3}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor expected_b_arc_map =
        torch::tensor({0, 2}, torch::dtype(torch::kInt32).device(device));

    EXPECT_TRUE(torch::equal(a_arc_map, expected_a_arc_map));
    EXPECT_TRUE(torch::equal(b_arc_map, expected_b_arc_map));

    auto expected_ragged_attr_a = Ragged<int32_t>(c, "[[30 40 50] [80]]");
    auto expected_ragged_attr_b = Ragged<int32_t>(c, "[[10 20] [60 70]]");

    EXPECT_TRUE(Equal(ToRaggedInt(dest.GetAttr("ragged_attr_a")),
                      expected_ragged_attr_a));
    EXPECT_TRUE(Equal(ToRaggedInt(dest.GetAttr("ragged_attr_b")),
                      expected_ragged_attr_b));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").toTensor());
    expected_float_attr[0] = a_float_attr[1] + b_float_attr[0];
    expected_float_attr[1] = a_float_attr[3] + b_float_attr[2];

    EXPECT_TRUE(torch::allclose(dest.GetAttr("float_attr").toTensor(),
                                expected_float_attr));

    torch::Tensor expected_float_attr_a =
        torch::empty_like(dest.GetAttr("float_attr_a").toTensor());
    expected_float_attr_a[0] = a_float_attr[1];
    expected_float_attr_a[1] = a_float_attr[3];

    EXPECT_TRUE(torch::allclose(dest.GetAttr("float_attr_a").toTensor(),
                                expected_float_attr_a));

    torch::Tensor expected_float_attr_b =
        torch::empty_like(dest.GetAttr("float_attr_b").toTensor());
    expected_float_attr_b[0] = b_float_attr[0];
    expected_float_attr_b[1] = b_float_attr[2];

    EXPECT_TRUE(torch::allclose(dest.GetAttr("float_attr_b").toTensor(),
                                expected_float_attr_b));
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
