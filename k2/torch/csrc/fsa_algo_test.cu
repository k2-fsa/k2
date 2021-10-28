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
#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/torch_utils.h"

namespace k2 {

TEST(FsaAlgoTest, AddEpsilonSelfLoopsSingle) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s = R"(
        0 1 1 0.1
        0 2 1 0.2
        1 3 2 0.3
        2 3 3 0.4
        3 4 -1 0.5
        4)";
    FsaClass fsa = FsaClass(s).To(device);
    fsa.SetRequiresGrad(true);
    FsaClass new_fsa = fsa.AddEpsilonSelfLoops();

    EXPECT_TRUE(torch::equal(
        new_fsa.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        torch::tensor({{0, 0, 0},
                       {0, 1, 1},
                       {0, 2, 1},
                       {1, 1, 0},
                       {1, 3, 2},
                       {2, 2, 0},
                       {2, 3, 3},
                       {3, 3, 0},
                       {3, 4, -1}},
                      torch::dtype(torch::kInt32).device(device))));
    EXPECT_TRUE(torch::allclose(
        new_fsa.Scores(),
        torch::tensor({0.0, 0.1, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5},
                      torch::dtype(torch::kFloat32).device(device))));

    torch::Tensor scale =
        torch::arange(new_fsa.Scores().numel(), torch::device(device));
    torch::Tensor scores_sum = (new_fsa.Scores() * scale).sum();
    torch::autograd::backward({scores_sum}, {});

    EXPECT_TRUE(torch::allclose(
        fsa.Scores().grad(),
        torch::tensor({1.0, 2.0, 4.0, 6.0, 8.0},
                      torch::dtype(torch::kFloat32).device(device))));
  }
}

TEST(FsaAlgoTest, AddEpsilonSelfLoopsVector) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::string s1 = R"(0 1 1 0.1
        0 2 1 0.2
        1 3 2 0.3
        2 3 3 0.4
        3 4 -1 0.5
        4)";
    std::string s2 = R"(0 1 1 0.1
        0 2 2 0.2
        1 2 3 0.3
        2 3 -1 0.4
        3)";
    FsaClass fsa1 = FsaClass(s1).To(device);
    FsaClass fsa2 = FsaClass(s2).To(device);
    fsa1.SetRequiresGrad(true);
    fsa2.SetRequiresGrad(true);
    std::vector<FsaClass> fsas;
    fsas.emplace_back(fsa1);
    fsas.emplace_back(fsa2);
    FsaClass fsa_vec = FsaClass::CreateFsaVec(fsas);
    FsaClass new_fsa_vec = fsa_vec.AddEpsilonSelfLoops();
    EXPECT_TRUE(torch::equal(
        new_fsa_vec.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        torch::tensor({{0, 0, 0},
                       {0, 1, 1},
                       {0, 2, 1},
                       {1, 1, 0},
                       {1, 3, 2},
                       {2, 2, 0},
                       {2, 3, 3},
                       {3, 3, 0},
                       {3, 4, -1},
                       {0, 0, 0},
                       {0, 1, 1},
                       {0, 2, 2},
                       {1, 1, 0},
                       {1, 2, 3},
                       {2, 2, 0},
                       {2, 3, -1}},
                      torch::dtype(torch::kInt32).device(device))));
    EXPECT_TRUE(torch::allclose(
        new_fsa_vec.Scores(),
        torch::tensor({0.0, 0.1, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5, 0.0, 0.1,
                       0.2, 0.0, 0.3, 0.0, 0.4},
                      torch::dtype(torch::kFloat32).device(device))));
    torch::Tensor scale =
        torch::arange(new_fsa_vec.Scores().numel(), torch::device(device));

    torch::Tensor scores_sum = (new_fsa_vec.Scores() * scale).sum();
    torch::autograd::backward({scores_sum}, {});

    EXPECT_TRUE(torch::allclose(
        fsa1.Scores().grad(),
        torch::tensor({1.0, 2.0, 4.0, 6.0, 8.0},
                      torch::dtype(torch::kFloat32).device(device))));
    EXPECT_TRUE(torch::allclose(
        fsa2.Scores().grad(),
        torch::tensor({10.0, 11.0, 13.0, 15.0},
                      torch::dtype(torch::kFloat32).device(device))));
  }
}

TEST(FsaAlgoTest, Connect) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 1 0.1
        0 2 2 0.2
        1 4 -1 0.3
        3 4 -1 0.4
        4)";
    auto device = GetDevice(c);
    FsaClass fsa = FsaClass(s).To(device);
    fsa.SetRequiresGrad(true);
    FsaClass connected_fsa = fsa.Connect();
    torch::Tensor loss = connected_fsa.Scores().sum();
    torch::autograd::backward({loss}, {});
    EXPECT_TRUE(torch::allclose(
        fsa.Scores().grad(),
        torch::tensor({1, 0, 1, 0},
                      torch::dtype(torch::kFloat32).device(device))));
    std::string expected_str = "0 1 1 0.1\n1 2 -1 0.3\n2\n";

    EXPECT_EQ(connected_fsa.ToStringSimple(), expected_str);
  }
}

TEST(FsaAlgoTest, Topsort) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    // arc 0: 0 -> 1, weight 1
    // arc 1: 0 -> 2, weight 2
    // arc 2: 1 -> 3, weight 3
    // arc 3: 2 -> 1, weight 4
    // the shortest path is 0 -> 1 -> 3, weight is 4
    // That is, (arc 0) -> (arc 2)
    std::string s = R"(0 1 1 1
        0 2 2 2
        1 3 -1 3
        2 1 3 4
        3)";
    FsaClass fsa = FsaClass(s).To(device);
    fsa.SetRequiresGrad(true);
    auto sorted_fsa = fsa.TopSort();
    // the shortest path in the sorted fsa is(arc 0)->(arc 3)
    torch::Tensor loss = (sorted_fsa.Scores()[0] + sorted_fsa.Scores()[3]) / 2;
    torch::autograd::backward({loss}, {});
    torch::Tensor expected = torch::tensor(
        {0.5, 0.0, 0.5, 0.0}, torch::dtype(torch::kFloat32).device(device));
    EXPECT_TRUE(torch::allclose(fsa.Scores().grad(), expected));
  }
}

}  // namespace k2
