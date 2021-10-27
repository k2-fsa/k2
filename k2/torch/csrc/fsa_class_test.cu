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
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/ragged_any.h"
#include "k2/torch/csrc/torch_utils.h"

namespace k2 {

TEST(FsaClassTest, FromUnaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 2 10
        0 1 1 20
        1 2 -1 30
        2)";

    auto device = GetDevice(c);
    FsaClass src = FsaClass(s).To(device);
    src.SetRequiresGrad(true);

    src.SetAttr(
        "float_attr",
        torch::IValue(torch::tensor(
            {0.1, 0.2, 0.3},
            torch::dtype(torch::kFloat32).device(device).requires_grad(true))));

    src.SetAttr("int_attr",
                torch::IValue(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    RaggedAny ragged_attr("[[1 2 3] [5 6] []]", torch::kInt32, device);

    src.SetAttr("ragged_attr", ToIValue(ragged_attr));

    src.SetAttr("attr1", torch::IValue("src"));
    src.SetAttr("attr2", torch::IValue(10));

    Array1<int32_t> arc_map;
    Ragged<Arc> arcs;
    ArcSort(src.fsa, &arcs, &arc_map);
    auto dest =
        FsaClass::FromUnaryFunctionTensor(src, arcs, ToTorch<int32_t>(arc_map));

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

    RaggedAny expected_ragged_attr =
        RaggedAny("[[5 6] [1 2 3] []]", torch::kInt32, device);

    EXPECT_EQ(ToRaggedAny(dest.GetAttr("ragged_attr")).ToString(true),
              expected_ragged_attr.ToString(true));

    EXPECT_EQ(dest.GetAttr("attr1").toStringRef(),
              src.GetAttr("attr1").toStringRef());

    EXPECT_EQ(dest.GetAttr("attr2").toInt(), src.GetAttr("attr2").toInt());

    torch::Tensor scale = torch::tensor(
        {10, 20, 30}, torch::dtype(torch::kFloat32).device(device));

    torch::Tensor sum_attr =
        (dest.GetAttr("float_attr").toTensor() * scale).sum();
    torch::Tensor sum_score = (dest.Scores() * scale).sum();

    sum_attr.backward();
    sum_score.backward();

    torch::Tensor expected_grad = torch::tensor(
        {20, 10, 30}, torch::dtype(torch::kFloat32).device(device));

    EXPECT_TRUE(torch::allclose(src.GetAttr("float_attr").toTensor().grad(),
                                expected_grad));

    EXPECT_TRUE(torch::allclose(src.Scores().grad(), expected_grad));
  }
}

TEST(FsaClassTest, FromUnaryFunctionRagged) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s = R"(0 1 0 0
        0 1 1 0
        1 2 -1 0
        2)";
    torch::Tensor scores = torch::tensor(
        {1, 2, 3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor scores_copy = scores.detach().clone().requires_grad_(true);

    FsaClass src = FsaClass(s).To(device);
    src.SetScores(scores);
    src.SetAttr("attr1", torch::IValue("hello"));
    src.SetAttr("attr2", torch::IValue(10));

    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    src.SetAttr(
        "float_attr",
        torch::IValue(float_attr.detach().clone().requires_grad_(true)));

    src.SetAttr("int_attr",
                torch::IValue(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    auto ragged_attr_any =
        RaggedAny("[[10 20] [30 40 50] [60 70]]", torch::kInt32, device);
    src.SetAttr("ragged_attr", ToIValue(ragged_attr_any));

    Ragged<int32_t> arc_map;
    Ragged<Arc> arcs;
    RemoveEpsilon(src.fsa, src.Properties(), &arcs, &arc_map);

    FsaClass dest = FsaClass::FromUnaryFunctionRagged(src, arcs, arc_map);

    EXPECT_EQ(dest.GetAttr("attr1").toStringRef(),
              src.GetAttr("attr1").toStringRef());

    EXPECT_EQ(dest.GetAttr("attr2").toInt(), src.GetAttr("attr2").toInt());

    RaggedAny expected_arc_map =
        RaggedAny("[[1] [0 2] [2]]", torch::kInt32, device);

    EXPECT_EQ(RaggedAny(arc_map.Generic()).ToString(true),
              expected_arc_map.ToString(true));

    RaggedAny expected_int_attr =
        RaggedAny("[[2] [1 3] [3]]", torch::kInt32, device);
    EXPECT_EQ(ToRaggedAny(dest.GetAttr("int_attr")).ToString(true),
              expected_int_attr.ToString(true));

    RaggedAny expected_ragged_attr =
        RaggedAny("[[30 40 50] [10 20 60 70] [60 70]]", torch::kInt32, device);

    EXPECT_EQ(ToRaggedAny(dest.GetAttr("ragged_attr")).ToString(true),
              expected_ragged_attr.ToString(true));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").toTensor());
    expected_float_attr[0] = float_attr[1];
    expected_float_attr[1] = float_attr[0] + float_attr[2];
    expected_float_attr[2] = float_attr[2];

    EXPECT_TRUE(torch::allclose(dest.GetAttr("float_attr").toTensor(),
                                expected_float_attr));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = scores_copy[1];
    expected_scores[1] = scores_copy[0] + scores_copy[2];
    expected_scores[2] = scores_copy[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scale = torch::tensor({10, 20, 30}).to(float_attr);

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").toTensor() * scale).sum();

    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    float_attr_sum.backward();
    expected_float_attr_sum.backward();

    EXPECT_TRUE(torch::allclose(src.GetAttr("float_attr").toTensor().grad(),
                                float_attr.grad()));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    scores_sum.backward();
    expected_scores_sum.backward();

    EXPECT_TRUE(torch::allclose(scores.grad(), scores_copy.grad()));
  }
}

TEST(FsaClassTest, FromUnaryFunctionRaggedWithEmptyList) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s = R"(0 1 0 0
        0 1 1 0
        1 2 -1 0
        2)";

    torch::Tensor scores = torch::tensor(
        {1, 2, 3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor scores_copy = scores.detach().clone().requires_grad_(true);

    FsaClass src = FsaClass(s).To(device);
    src.SetScores(scores);
    src.SetAttr("attr1", torch::IValue("hello"));
    src.SetAttr("attr2", torch::IValue(10));

    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    src.SetAttr(
        "float_attr",
        torch::IValue(float_attr.detach().clone().requires_grad_(true)));

    src.SetAttr("int_attr",
                torch::IValue(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    auto ragged_attr_any =
        RaggedAny("[[10 20] [30 40 50] [60 70]]", torch::kInt32, device);
    src.SetAttr("ragged_attr", ToIValue(ragged_attr_any));

    Ragged<int32_t> arc_map;
    Ragged<Arc> arcs;
    RemoveEpsilonAndAddSelfLoops(src.fsa, src.Properties(), &arcs, &arc_map);

    FsaClass dest = FsaClass::FromUnaryFunctionRagged(src, arcs, arc_map);

    EXPECT_EQ(dest.GetAttr("attr1").toStringRef(),
              src.GetAttr("attr1").toStringRef());

    EXPECT_EQ(dest.GetAttr("attr2").toInt(), src.GetAttr("attr2").toInt());

    RaggedAny expected_arc_map =
        RaggedAny("[[] [1] [0 2] [] [2]]", torch::kInt32, device);

    EXPECT_EQ(RaggedAny(arc_map.Generic()).ToString(true),
              expected_arc_map.ToString(true));

    RaggedAny expected_int_attr =
        RaggedAny("[[] [2] [1 3] [] [3]]", torch::kInt32, device);
    EXPECT_EQ(ToRaggedAny(dest.GetAttr("int_attr")).ToString(true),
              expected_int_attr.ToString(true));

    RaggedAny expected_ragged_attr = RaggedAny(
        "[[] [30 40 50] [10 20 60 70] [] [60 70]]", torch::kInt32, device);

    EXPECT_EQ(ToRaggedAny(dest.GetAttr("ragged_attr")).ToString(true),
              expected_ragged_attr.ToString(true));

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
    expected_scores[1] = scores_copy[1];
    expected_scores[2] = scores_copy[0] + scores_copy[2];
    expected_scores[3] = 0;
    expected_scores[4] = scores_copy[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scale = torch::tensor({10, 20, 30, 40, 50}).to(float_attr);

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").toTensor() * scale).sum();

    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    float_attr_sum.backward();
    expected_float_attr_sum.backward();

    EXPECT_TRUE(torch::allclose(src.GetAttr("float_attr").toTensor().grad(),
                                float_attr.grad()));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    scores_sum.backward();
    expected_scores_sum.backward();

    EXPECT_TRUE(torch::allclose(scores.grad(), scores_copy.grad()));
  }
}

TEST(FsaClassTest, FromBinaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s1 = R"(0 1 0 0.1
        0 1 1 0.2
        1 1 2 0.3
        1 2 -1 0.4
        2)";
    std::vector<FsaClass> a_fsas;
    a_fsas.emplace_back(FsaClass(s1));
    FsaClass a_fsa = FsaClass::CreateFsaVec(a_fsas).To(device);
    a_fsa.SetRequiresGrad(true);

    a_fsa.SetAttr("attr1", torch::IValue("hello"));

    torch::Tensor scores_a = torch::tensor(
        {0.1, 0.2, 0.3, 0.4},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor a_float_attr = torch::tensor(
        {1, 2, 3, 4},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor a_float_attr_copy =
        a_float_attr.detach().clone().requires_grad_(true);

    a_fsa.SetAttr(
        "float_attr",
        torch::IValue(a_float_attr.detach().clone().requires_grad_(true)));

    a_fsa.SetAttr(
        "float_attr_a",
        torch::IValue(a_float_attr.detach().clone().requires_grad_(true)));

    auto ragged_attr_a_any =
        RaggedAny("[[10 20] [30 40 50] [60 70] [80]]", torch::kInt32, device);
    a_fsa.SetAttr("ragged_attr_a", ToIValue(ragged_attr_a_any));

    std::string s2 = R"(0 1 1 1
        0 1 2 2
        1 2 -1 3
        2)";

    std::vector<FsaClass> b_fsas;
    b_fsas.emplace_back(FsaClass(s2));
    FsaClass b_fsa = FsaClass::CreateFsaVec(b_fsas).To(device);
    b_fsa.SetRequiresGrad(true);

    // this attribute will not be propagated to the final fsa,
    // because there is already an attribute named `attr1` in a_fsa.
    b_fsa.SetAttr("attr1", torch::IValue("hello2"));
    b_fsa.SetAttr("attr2", torch::IValue("k2"));

    torch::Tensor scores_b = torch::tensor(
        {1, 2, 3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor b_float_attr = torch::tensor(
        {0.1, 0.2, 0.3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor b_float_attr_copy =
        b_float_attr.detach().clone().requires_grad_(true);

    b_fsa.SetAttr(
        "float_attr",
        torch::IValue(b_float_attr.detach().clone().requires_grad_(true)));
    b_fsa.SetAttr(
        "float_attr_b",
        torch::IValue(b_float_attr.detach().clone().requires_grad_(true)));

    auto ragged_attr_b_any =
        RaggedAny("[[10 20] [30 40 50] [60 70]]", torch::kInt32, device);
    b_fsa.SetAttr("ragged_attr_b", ToIValue(ragged_attr_b_any));

    Array1<int32_t> a_arc_map_raw;
    Array1<int32_t> b_arc_map_raw;
    Array1<int32_t> b_to_a_map(c, "[0]");
    Ragged<Arc> arcs = IntersectDevice(a_fsa.fsa, a_fsa.Properties(), b_fsa.fsa,
                                       b_fsa.Properties(), b_to_a_map,
                                       &a_arc_map_raw, &b_arc_map_raw, true);

    torch::Tensor a_arc_map = ToTorch<int32_t>(a_arc_map_raw);
    torch::Tensor b_arc_map = ToTorch<int32_t>(b_arc_map_raw);

    FsaClass dest = FsaClass::FromBinaryFunctionTensor(a_fsa, b_fsa, arcs,
                                                       a_arc_map, b_arc_map);

    torch::Tensor expected_a_arc_map =
        torch::tensor({1, 3}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor expected_b_arc_map =
        torch::tensor({0, 2}, torch::dtype(torch::kInt32).device(device));

    EXPECT_TRUE(torch::equal(a_arc_map, expected_a_arc_map));
    EXPECT_TRUE(torch::equal(b_arc_map, expected_b_arc_map));

    EXPECT_EQ(dest.GetAttr("attr1").toStringRef(),
              a_fsa.GetAttr("attr1").toStringRef());
    EXPECT_EQ(dest.GetAttr("attr2").toStringRef(),
              b_fsa.GetAttr("attr2").toStringRef());

    RaggedAny expected_ragged_attr_a =
        RaggedAny("[[30 40 50] [80]]", torch::kInt32, device);
    RaggedAny expected_ragged_attr_b =
        RaggedAny("[[10 20] [60 70]]", torch::kInt32, device);

    EXPECT_EQ(ToRaggedAny(dest.GetAttr("ragged_attr_a")).ToString(true),
              expected_ragged_attr_a.ToString(true));
    EXPECT_EQ(ToRaggedAny(dest.GetAttr("ragged_attr_b")).ToString(true),
              expected_ragged_attr_b.ToString(true));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").toTensor());
    expected_float_attr[0] = a_float_attr_copy[1] + b_float_attr_copy[0];
    expected_float_attr[1] = a_float_attr_copy[3] + b_float_attr_copy[2];

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

    torch::Tensor scale =
        torch::tensor({10, 20}, torch::dtype(torch::kFloat32).device(device));

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").toTensor() * scale).sum();
    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    torch::Tensor float_attr_a_sum =
        (dest.GetAttr("float_attr_a").toTensor() * scale).sum();
    torch::Tensor expected_float_attr_a_sum =
        (expected_float_attr_a * scale).sum();

    torch::Tensor float_attr_b_sum =
        (dest.GetAttr("float_attr_b").toTensor() * scale).sum();
    torch::Tensor expected_float_attr_b_sum =
        (expected_float_attr_b * scale).sum();

    float_attr_sum.backward();
    expected_float_attr_sum.backward();
    float_attr_a_sum.backward();
    expected_float_attr_a_sum.backward();
    float_attr_b_sum.backward();
    expected_float_attr_b_sum.backward();

    EXPECT_TRUE(torch::allclose(a_fsa.GetAttr("float_attr").toTensor().grad(),
                                a_float_attr_copy.grad()));
    EXPECT_TRUE(torch::allclose(b_fsa.GetAttr("float_attr").toTensor().grad(),
                                b_float_attr_copy.grad()));

    EXPECT_TRUE(torch::allclose(a_fsa.GetAttr("float_attr_a").toTensor().grad(),
                                a_float_attr.grad()));
    EXPECT_TRUE(torch::allclose(b_fsa.GetAttr("float_attr_b").toTensor().grad(),
                                b_float_attr.grad()));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = scores_a[1] + scores_b[0];
    expected_scores[1] = scores_a[3] + scores_b[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    scores_sum.backward();
    expected_scores_sum.backward();

    EXPECT_TRUE(torch::allclose(a_fsa.Scores().grad(), scores_a.grad()));
    EXPECT_TRUE(torch::allclose(b_fsa.Scores().grad(), scores_b.grad()));
  }
}

TEST(FsaClassTest, Attributes) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s = R"(0 1 2 10
        0 1 1 20
        1 2 -1 30
        2)";
    FsaClass src = FsaClass(s).To(device);

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
    RaggedAny ragged_int =
        RaggedAny("[[1, 2], [3], [4]]", torch::kInt32, device);
    src.SetAttr("ragged_int", ToIValue(ragged_int));
    RaggedAny ragged_float =
        RaggedAny("[[1, 2], [3], [4]]", torch::kFloat32, device);
    src.SetAttr("ragged_float", ToIValue(ragged_float));

    EXPECT_EQ(ToRaggedAny(src.GetAttr("ragged_int")).ToString(),
              ragged_int.ToString());
    EXPECT_EQ(ToRaggedAny(src.GetAttr("ragged_float")).ToString(),
              ragged_float.ToString());
    src.DeleteAttr("ragged_int");
    EXPECT_FALSE(src.HasAttr("ragged_int"));

    // test other attribute
    src.SetAttr("int", torch::IValue(10));
    src.SetAttr("str", torch::IValue("k2"));
    EXPECT_EQ(src.GetAttr("int").toInt(), 10);
    EXPECT_EQ(src.GetAttr("str").toStringRef(), "k2");
    src.DeleteAttr("int");
    EXPECT_FALSE(src.HasAttr("int"));

    // test filler
    src.SetAttr("tensor_int_filler", torch::IValue(10));
    src.SetAttr("tensor_float_filler", torch::IValue(10.0));

    EXPECT_EQ(src.GetFiller("tensor_int").toInt(), 10);
    EXPECT_EQ(src.GetFiller("tensor_float").toDouble(), 10.0);
    EXPECT_EQ(src.GetFiller("none").toInt(), 0);
    src.DeleteAttr("tensor_int_filler");
    EXPECT_EQ(src.GetFiller("tensor_int").toInt(), 0);
  }
}

}  // namespace k2
