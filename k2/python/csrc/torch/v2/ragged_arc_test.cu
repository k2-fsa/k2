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
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"
#include "pybind11/embed.h"

namespace k2 {

TEST(RaggedArcTest, FromUnaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 2 10
        0 1 1 20
        1 2 -1 30
        2)";

    auto device = GetDevice(c);
    RaggedArc src = RaggedArc(s).To(device);
    src.SetRequiresGrad(true);

    src.SetAttr(
        "float_attr",
        py::cast(torch::tensor(
            {0.1, 0.2, 0.3},
            torch::dtype(torch::kFloat32).device(device).requires_grad(true))));

    src.SetAttr("int_attr",
                py::cast(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    src.SetAttr("ragged_attr",
                py::cast(RaggedAny("[[1 2 3] [5 6] []]",
                                   py::cast(torch::kInt32), device)));

    src.SetAttr("attr1", py::str("src"));
    src.SetAttr("attr2", py::str("fsa"));

    Array1<int32_t> arc_map;
    Ragged<Arc> arcs;
    ArcSort(src.fsa, &arcs, &arc_map);
    auto dest = RaggedArc::FromUnaryFunctionTensor(src, arcs,
                                                   ToTorch<int32_t>(arc_map));

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").cast<torch::Tensor>(),
        torch::tensor({0.2, 0.1, 0.3},
                      torch::dtype(torch::kFloat32).device(device))));

    EXPECT_TRUE(torch::allclose(
        dest.Scores(),
        torch::tensor({20, 10, 30},
                      torch::dtype(torch::kFloat32).device(device))));

    EXPECT_TRUE(torch::equal(
        dest.GetAttr("int_attr").cast<torch::Tensor>(),
        torch::tensor({2, 1, 3}, torch::dtype(torch::kInt32).device(device))));

    RaggedAny expected_ragged_attr =
        RaggedAny("[[5 6] [1 2 3] []]", py::cast(torch::kInt32), device);
    EXPECT_EQ(dest.GetAttr("ragged_attr").cast<RaggedAny>().ToString(true),
              expected_ragged_attr.ToString(true));

    EXPECT_EQ(dest.GetAttr("attr1").cast<std::string>(),
              src.GetAttr("attr1").cast<std::string>());

    EXPECT_EQ(dest.GetAttr("attr2").cast<std::string>(),
              src.GetAttr("attr2").cast<std::string>());

    torch::Tensor scale = torch::tensor(
        {10, 20, 30}, torch::dtype(torch::kFloat32).device(device));

    torch::Tensor sum_attr =
        (dest.GetAttr("float_attr").cast<torch::Tensor>() * scale).sum();
    torch::Tensor sum_score = (dest.Scores() * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({sum_attr}, {});
      torch::autograd::backward({sum_score}, {});
    }

    torch::Tensor expected_grad = torch::tensor(
        {20, 10, 30}, torch::dtype(torch::kFloat32).device(device));

    EXPECT_TRUE(torch::allclose(
        src.GetAttr("float_attr").cast<torch::Tensor>().grad(), expected_grad));

    EXPECT_TRUE(torch::allclose(src.Scores().grad(), expected_grad));
  }
}

TEST(RaggedArcTest, FromUnaryFunctionRagged) {
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

    RaggedArc src = RaggedArc(s).To(device);
    src.SetScores(scores);
    src.SetAttr("attr1", py::str("hello"));
    src.SetAttr("attr2", py::str("k2"));

    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    src.SetAttr("float_attr",
                py::cast(float_attr.detach().clone().requires_grad_(true)));

    src.SetAttr("int_attr",
                py::cast(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    src.SetAttr("ragged_attr",
                py::cast(RaggedAny("[[10 20] [30 40 50] [60 70]]",
                                   py::cast(torch::kInt32), device)));

    Ragged<int32_t> arc_map;
    Ragged<Arc> arcs;
    RemoveEpsilon(src.fsa, src.Properties(), &arcs, &arc_map);

    RaggedArc dest = RaggedArc::FromUnaryFunctionRagged(src, arcs, arc_map);

    EXPECT_EQ(dest.GetAttr("attr1").cast<std::string>(),
              src.GetAttr("attr1").cast<std::string>());

    EXPECT_EQ(dest.GetAttr("attr2").cast<std::string>(),
              src.GetAttr("attr2").cast<std::string>());

    RaggedAny expected_arc_map =
        RaggedAny("[[1] [0 2] [2]]", py::cast(torch::kInt32), device);

    EXPECT_EQ(RaggedAny(arc_map.Generic()).ToString(true),
              expected_arc_map.ToString(true));

    RaggedAny expected_int_attr =
        RaggedAny("[[2] [1 3] [3]]", py::cast(torch::kInt32), device);
    EXPECT_EQ(dest.GetAttr("int_attr").cast<RaggedAny>().ToString(true),
              expected_int_attr.ToString(true));

    RaggedAny expected_ragged_attr = RaggedAny(
        "[[30 40 50] [10 20 60 70] [60 70]]", py::cast(torch::kInt32), device);

    EXPECT_EQ(dest.GetAttr("ragged_attr").cast<RaggedAny>().ToString(true),
              expected_ragged_attr.ToString(true));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").cast<torch::Tensor>());
    expected_float_attr[0] = float_attr[1];
    expected_float_attr[1] = float_attr[0] + float_attr[2];
    expected_float_attr[2] = float_attr[2];

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").cast<torch::Tensor>(), expected_float_attr));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = scores_copy[1];
    expected_scores[1] = scores_copy[0] + scores_copy[2];
    expected_scores[2] = scores_copy[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scale = torch::tensor({10, 20, 30}).to(float_attr);

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").cast<torch::Tensor>() * scale).sum();

    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({float_attr_sum}, {});
      torch::autograd::backward({expected_float_attr_sum}, {});
    }

    EXPECT_TRUE(
        torch::allclose(src.GetAttr("float_attr").cast<torch::Tensor>().grad(),
                        float_attr.grad()));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({scores_sum}, {});
      torch::autograd::backward({expected_scores_sum}, {});
    }

    EXPECT_TRUE(torch::allclose(scores.grad(), scores_copy.grad()));
  }
}

TEST(RaggedArcTest, FromUnaryFunctionRaggedWithEmptyList) {
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

    RaggedArc src = RaggedArc(s).To(device);
    src.SetScores(scores);
    src.SetAttr("attr1", py::str("hello"));
    src.SetAttr("attr2", py::str("k2"));

    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    src.SetAttr("float_attr",
                py::cast(float_attr.detach().clone().requires_grad_(true)));

    src.SetAttr("int_attr",
                py::cast(torch::tensor(
                    {1, 2, 3}, torch::dtype(torch::kInt32).device(device))));

    src.SetAttr("ragged_attr",
                py::cast(RaggedAny("[[10 20] [30 40 50] [60 70]]",
                                   py::cast(torch::kInt32), device)));

    Ragged<int32_t> arc_map;
    Ragged<Arc> arcs;
    RemoveEpsilonAndAddSelfLoops(src.fsa, src.Properties(), &arcs, &arc_map);

    RaggedArc dest = RaggedArc::FromUnaryFunctionRagged(src, arcs, arc_map);

    EXPECT_EQ(dest.GetAttr("attr1").cast<std::string>(),
              src.GetAttr("attr1").cast<std::string>());

    EXPECT_EQ(dest.GetAttr("attr2").cast<std::string>(),
              src.GetAttr("attr2").cast<std::string>());

    RaggedAny expected_arc_map =
        RaggedAny("[[] [1] [0 2] [] [2]]", py::cast(torch::kInt32), device);

    EXPECT_EQ(RaggedAny(arc_map.Generic()).ToString(true),
              expected_arc_map.ToString(true));

    RaggedAny expected_int_attr =
        RaggedAny("[[] [2] [1 3] [] [3]]", py::cast(torch::kInt32), device);
    EXPECT_EQ(dest.GetAttr("int_attr").cast<RaggedAny>().ToString(true),
              expected_int_attr.ToString(true));

    RaggedAny expected_ragged_attr =
        RaggedAny("[[] [30 40 50] [10 20 60 70] [] [60 70]]",
                  py::cast(torch::kInt32), device);

    EXPECT_EQ(dest.GetAttr("ragged_attr").cast<RaggedAny>().ToString(true),
              expected_ragged_attr.ToString(true));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").cast<torch::Tensor>());
    expected_float_attr[0] = 0;
    expected_float_attr[1] = float_attr[1];
    expected_float_attr[2] = float_attr[0] + float_attr[2];
    expected_float_attr[3] = 0;
    expected_float_attr[4] = float_attr[2];

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").cast<torch::Tensor>(), expected_float_attr));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = 0;
    expected_scores[1] = scores_copy[1];
    expected_scores[2] = scores_copy[0] + scores_copy[2];
    expected_scores[3] = 0;
    expected_scores[4] = scores_copy[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scale = torch::tensor({10, 20, 30, 40, 50}).to(float_attr);

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").cast<torch::Tensor>() * scale).sum();

    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({float_attr_sum}, {});
      torch::autograd::backward({expected_float_attr_sum}, {});
    }

    EXPECT_TRUE(
        torch::allclose(src.GetAttr("float_attr").cast<torch::Tensor>().grad(),
                        float_attr.grad()));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({scores_sum}, {});
      torch::autograd::backward({expected_scores_sum}, {});
    }

    EXPECT_TRUE(torch::allclose(scores.grad(), scores_copy.grad()));
  }
}

TEST(RaggedArcTest, FromBinaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s1 = R"(0 1 0 0.1
        0 1 1 0.2
        1 1 2 0.3
        1 2 -1 0.4
        2)";
    std::vector<RaggedArc> a_fsas;
    a_fsas.emplace_back(RaggedArc(s1));
    RaggedArc a_fsa = RaggedArc::CreateFsaVec(a_fsas).To(device);
    a_fsa.SetRequiresGrad(true);

    a_fsa.SetAttr("attr1", py::str("hello"));

    torch::Tensor scores_a = torch::tensor(
        {0.1, 0.2, 0.3, 0.4},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor a_float_attr = torch::tensor(
        {1, 2, 3, 4},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor a_float_attr_copy =
        a_float_attr.detach().clone().requires_grad_(true);

    a_fsa.SetAttr("float_attr",
                  py::cast(a_float_attr.detach().clone().requires_grad_(true)));

    a_fsa.SetAttr("float_attr_a",
                  py::cast(a_float_attr.detach().clone().requires_grad_(true)));

    a_fsa.SetAttr("ragged_attr_a",
                  py::cast(RaggedAny("[[10 20] [30 40 50] [60 70] [80]]",
                                     py::cast(torch::kInt32), device)));

    std::string s2 = R"(0 1 1 1
        0 1 2 2
        1 2 -1 3
        2)";

    std::vector<RaggedArc> b_fsas;
    b_fsas.emplace_back(RaggedArc(s2));
    RaggedArc b_fsa = RaggedArc::CreateFsaVec(b_fsas).To(device);
    b_fsa.SetRequiresGrad(true);

    // this attribute will not be propagated to the final fsa,
    // because there is already an attribute named `attr1` in a_fsa.
    b_fsa.SetAttr("attr1", py::str("hello2"));
    b_fsa.SetAttr("attr2", py::str("k2"));

    torch::Tensor scores_b = torch::tensor(
        {1, 2, 3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor b_float_attr = torch::tensor(
        {0.1, 0.2, 0.3},
        torch::dtype(torch::kFloat32).device(device).requires_grad(true));

    torch::Tensor b_float_attr_copy =
        b_float_attr.detach().clone().requires_grad_(true);

    b_fsa.SetAttr("float_attr",
                  py::cast(b_float_attr.detach().clone().requires_grad_(true)));
    b_fsa.SetAttr("float_attr_b",
                  py::cast(b_float_attr.detach().clone().requires_grad_(true)));

    b_fsa.SetAttr("ragged_attr_b",
                  py::cast(RaggedAny("[[10 20] [30 40 50] [60 70]]",
                                     py::cast(torch::kInt32), device)));

    Array1<int32_t> a_arc_map_raw;
    Array1<int32_t> b_arc_map_raw;
    Array1<int32_t> b_to_a_map(c, "[0]");
    Ragged<Arc> arcs = IntersectDevice(a_fsa.fsa, a_fsa.Properties(), b_fsa.fsa,
                                       b_fsa.Properties(), b_to_a_map,
                                       &a_arc_map_raw, &b_arc_map_raw, true);

    torch::Tensor a_arc_map = ToTorch<int32_t>(a_arc_map_raw);
    torch::Tensor b_arc_map = ToTorch<int32_t>(b_arc_map_raw);

    RaggedArc dest = RaggedArc::FromBinaryFunctionTensor(a_fsa, b_fsa, arcs,
                                                         a_arc_map, b_arc_map);

    torch::Tensor expected_a_arc_map =
        torch::tensor({1, 3}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor expected_b_arc_map =
        torch::tensor({0, 2}, torch::dtype(torch::kInt32).device(device));

    EXPECT_TRUE(torch::equal(a_arc_map, expected_a_arc_map));
    EXPECT_TRUE(torch::equal(b_arc_map, expected_b_arc_map));

    EXPECT_EQ(dest.GetAttr("attr1").cast<std::string>(),
              a_fsa.GetAttr("attr1").cast<std::string>());
    EXPECT_EQ(dest.GetAttr("attr2").cast<std::string>(),
              b_fsa.GetAttr("attr2").cast<std::string>());

    RaggedAny expected_ragged_attr_a =
        RaggedAny("[[30 40 50] [80]]", py::cast(torch::kInt32), device);
    RaggedAny expected_ragged_attr_b =
        RaggedAny("[[10 20] [60 70]]", py::cast(torch::kInt32), device);

    EXPECT_EQ(dest.GetAttr("ragged_attr_a").cast<RaggedAny>().ToString(true),
              expected_ragged_attr_a.ToString(true));
    EXPECT_EQ(dest.GetAttr("ragged_attr_b").cast<RaggedAny>().ToString(true),
              expected_ragged_attr_b.ToString(true));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").cast<torch::Tensor>());
    expected_float_attr[0] = a_float_attr_copy[1] + b_float_attr_copy[0];
    expected_float_attr[1] = a_float_attr_copy[3] + b_float_attr_copy[2];

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").cast<torch::Tensor>(), expected_float_attr));

    torch::Tensor expected_float_attr_a =
        torch::empty_like(dest.GetAttr("float_attr_a").cast<torch::Tensor>());
    expected_float_attr_a[0] = a_float_attr[1];
    expected_float_attr_a[1] = a_float_attr[3];

    EXPECT_TRUE(
        torch::allclose(dest.GetAttr("float_attr_a").cast<torch::Tensor>(),
                        expected_float_attr_a));

    torch::Tensor expected_float_attr_b =
        torch::empty_like(dest.GetAttr("float_attr_b").cast<torch::Tensor>());
    expected_float_attr_b[0] = b_float_attr[0];
    expected_float_attr_b[1] = b_float_attr[2];

    EXPECT_TRUE(
        torch::allclose(dest.GetAttr("float_attr_b").cast<torch::Tensor>(),
                        expected_float_attr_b));

    torch::Tensor scale =
        torch::tensor({10, 20}, torch::dtype(torch::kFloat32).device(device));

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").cast<torch::Tensor>() * scale).sum();
    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    torch::Tensor float_attr_a_sum =
        (dest.GetAttr("float_attr_a").cast<torch::Tensor>() * scale).sum();
    torch::Tensor expected_float_attr_a_sum =
        (expected_float_attr_a * scale).sum();

    torch::Tensor float_attr_b_sum =
        (dest.GetAttr("float_attr_b").cast<torch::Tensor>() * scale).sum();
    torch::Tensor expected_float_attr_b_sum =
        (expected_float_attr_b * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({float_attr_sum}, {});
      torch::autograd::backward({expected_float_attr_sum}, {});
      torch::autograd::backward({float_attr_a_sum}, {});
      torch::autograd::backward({expected_float_attr_a_sum}, {});
      torch::autograd::backward({float_attr_b_sum}, {});
      torch::autograd::backward({expected_float_attr_b_sum}, {});
    }

    EXPECT_TRUE(torch::allclose(
        a_fsa.GetAttr("float_attr").cast<torch::Tensor>().grad(),
        a_float_attr_copy.grad()));
    EXPECT_TRUE(torch::allclose(
        b_fsa.GetAttr("float_attr").cast<torch::Tensor>().grad(),
        b_float_attr_copy.grad()));

    EXPECT_TRUE(torch::allclose(
        a_fsa.GetAttr("float_attr_a").cast<torch::Tensor>().grad(),
        a_float_attr.grad()));
    EXPECT_TRUE(torch::allclose(
        b_fsa.GetAttr("float_attr_b").cast<torch::Tensor>().grad(),
        b_float_attr.grad()));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = scores_a[1] + scores_b[0];
    expected_scores[1] = scores_a[3] + scores_b[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({scores_sum}, {});
      torch::autograd::backward({expected_scores_sum}, {});
    }

    EXPECT_TRUE(torch::allclose(a_fsa.Scores().grad(), scores_a.grad()));
    EXPECT_TRUE(torch::allclose(b_fsa.Scores().grad(), scores_b.grad()));
  }
}

}  // namespace k2

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  py::scoped_interpreter guard{};
  py::module_::import("torch");
  py::module_::import("_k2");
  return RUN_ALL_TESTS();
}
