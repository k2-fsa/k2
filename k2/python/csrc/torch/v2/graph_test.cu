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

#include "gtest/gtest.h"
#include "k2/csrc/test_utils.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/graph.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"
#include "pybind11/embed.h"

namespace k2 {

TEST(GraphTest, CtcTopo) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    // Test standard topology
    RaggedArc topo = CtcTopo(3, device, false);

    torch::Tensor expected_arcs =
        torch::tensor({{0, 0, 0}, {0, 1, 1}, {0, 2, 2}, {0, 3, 3}, {0, 4, -1},
                       {1, 0, 0}, {1, 1, 1}, {1, 2, 2}, {1, 3, 3}, {1, 4, -1},
                       {2, 0, 0}, {2, 1, 1}, {2, 2, 2}, {2, 3, 3}, {2, 4, -1},
                       {3, 0, 0}, {3, 1, 1}, {3, 2, 2}, {3, 3, 3}, {3, 4, -1}},
                      torch::dtype(torch::kInt32).device(device));

    torch::Tensor aux_label_ref = torch::tensor(
        {0, 1, 2, 3, -1, 0, 0, 2, 3, -1, 0, 1, 0, 3, -1, 0, 1, 2, 0, -1},
        torch::dtype(torch::kInt32).device(device));

    EXPECT_TRUE(
        torch::allclose(topo.Scores(), torch::zeros_like(topo.Scores())));

    EXPECT_TRUE(torch::equal(
        topo.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        expected_arcs));

    EXPECT_TRUE(torch::equal(topo.GetAttr("aux_labels").cast<torch::Tensor>(),
                             aux_label_ref));

    // Test simplified topology
    topo = CtcTopo(3, device, true);
    expected_arcs = torch::tensor({{0, 0, 0},
                                   {0, 0, 1},
                                   {0, 0, 2},
                                   {0, 0, 3},
                                   {0, 1, 1},
                                   {0, 2, 2},
                                   {0, 3, 3},
                                   {0, 4, -1},
                                   {1, 1, 1},
                                   {1, 0, 1},
                                   {2, 2, 2},
                                   {2, 0, 2},
                                   {3, 3, 3},
                                   {3, 0, 3}},
                                  torch::dtype(torch::kInt32).device(device));
    aux_label_ref = torch::tensor({0, 1, 2, 3, 1, 2, 3, -1, 0, 0, 0, 0, 0, 0},
                                  torch::dtype(torch::kInt32).device(device));
    EXPECT_TRUE(
        torch::allclose(topo.Scores(), torch::zeros_like(topo.Scores())));

    EXPECT_TRUE(torch::equal(
        topo.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        expected_arcs));

    EXPECT_TRUE(torch::equal(topo.GetAttr("aux_labels").cast<torch::Tensor>(),
                             aux_label_ref));
  }
}

TEST(GraphTest, CtcGraphs) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    RaggedAny symbols("[ [ 1 2 2 3 ] [ 1 2 3 ] ]", py::cast(torch::kInt32),
                      device);
    // test standard
    RaggedArc graph = CtcGraphs(symbols, false);
    torch::Tensor graph_ref = torch::tensor(
        {{0, 0, 0},  {0, 1, 1}, {1, 2, 0},  {1, 1, 1}, {1, 3, 2}, {2, 2, 0},
         {2, 3, 2},  {3, 4, 0}, {3, 3, 2},  {4, 4, 0}, {4, 5, 2}, {5, 6, 0},
         {5, 5, 2},  {5, 7, 3}, {6, 6, 0},  {6, 7, 3}, {7, 8, 0}, {7, 7, 3},
         {7, 9, -1}, {8, 8, 0}, {8, 9, -1},  // fsa 0
         {0, 0, 0},  {0, 1, 1}, {1, 2, 0},   // fsa 1
         {1, 1, 1},  {1, 3, 2}, {2, 2, 0},  {2, 3, 2}, {3, 4, 0}, {3, 3, 2},
         {3, 5, 3},  {4, 4, 0}, {4, 5, 3},  {5, 6, 0}, {5, 5, 3}, {5, 7, -1},
         {6, 6, 0},  {6, 7, -1}},
        torch::dtype(torch::kInt32).device(device));

    torch::Tensor aux_labels_ref =
        torch::tensor({0, 1, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0,
                       0, 0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0},
                      torch::dtype(torch::kInt32).device(device));
    EXPECT_TRUE(
        torch::allclose(graph.Scores(), torch::zeros_like(graph.Scores())));

    EXPECT_TRUE(torch::equal(
        graph.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        graph_ref));

    EXPECT_TRUE(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                             aux_labels_ref));

    std::vector<std::vector<int32_t>> symbols2({{1, 2, 2, 3}, {1, 2, 3}});

    graph = CtcGraphs(symbols2, device, false);
    EXPECT_TRUE(
        torch::allclose(graph.Scores(), torch::zeros_like(graph.Scores())));

    EXPECT_TRUE(torch::equal(
        graph.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        graph_ref));
    EXPECT_TRUE(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                             aux_labels_ref));

    // test modified
    graph = CtcGraphs(symbols, true);
    graph_ref = torch::tensor(
        {{0, 0, 0},  {0, 1, 1},  {1, 2, 0}, {1, 1, 1},  {1, 3, 2}, {2, 2, 0},
         {2, 3, 2},  {3, 4, 0},  {3, 3, 2}, {3, 5, 2},  {4, 4, 0}, {4, 5, 2},
         {5, 6, 0},  {5, 5, 2},  {5, 7, 3}, {6, 6, 0},  {6, 7, 3}, {7, 8, 0},
         {7, 7, 3},  {7, 9, -1}, {8, 8, 0}, {8, 9, -1},  // fsa 0
         {0, 0, 0},  {0, 1, 1},                          // fsa 1
         {1, 2, 0},  {1, 1, 1},  {1, 3, 2}, {2, 2, 0},  {2, 3, 2}, {3, 4, 0},
         {3, 3, 2},  {3, 5, 3},  {4, 4, 0}, {4, 5, 3},  {5, 6, 0}, {5, 5, 3},
         {5, 7, -1}, {6, 6, 0},  {6, 7, -1}},
        torch::dtype(torch::kInt32).device(device));

    aux_labels_ref = torch::tensor(
        {0, 1, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0},
        torch::dtype(torch::kInt32).device(device));

    EXPECT_TRUE(
        torch::allclose(graph.Scores(), torch::zeros_like(graph.Scores())));

    EXPECT_TRUE(torch::equal(
        graph.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        graph_ref));
    EXPECT_TRUE(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                             aux_labels_ref));

    graph = CtcGraphs(symbols2, device, true);
    EXPECT_TRUE(
        torch::allclose(graph.Scores(), torch::zeros_like(graph.Scores())));

    EXPECT_TRUE(torch::equal(
        graph.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        graph_ref));
    EXPECT_TRUE(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                             aux_labels_ref));
  }
}

TEST(GraphTest, LinearFsaSingle) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::vector<int32_t> labels({2, 5, 8});
    RaggedArc fsa = LinearFsa(labels, device);
    EXPECT_TRUE(torch::allclose(fsa.Scores(), torch::zeros_like(fsa.Scores())));
    EXPECT_TRUE(torch::equal(
        fsa.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        torch::tensor({{0, 1, 2}, {1, 2, 5}, {2, 3, 8}, {3, 4, -1}},
                      torch::dtype(torch::kInt32).device(device))));
  }
}

TEST(GraphTest, LinearFsaVec) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    std::vector<std::vector<int32_t>> labels({{1, 3, 5}, {2, 6}, {8, 7, 9}});
    RaggedArc fsa = LinearFsa(labels, device);
    torch::Tensor expected_arcs =
        torch::tensor({{0, 1, 1},  // fsa 0
                       {1, 2, 3},
                       {2, 3, 5},
                       {3, 4, -1},
                       {0, 1, 2},  // fsa 1
                       {1, 2, 6},
                       {2, 3, -1},
                       {0, 1, 8},  // fsa 2
                       {1, 2, 7},
                       {2, 3, 9},
                       {3, 4, -1}},
                      torch::dtype(torch::kInt32).device(device));
    EXPECT_TRUE(torch::allclose(fsa.Scores(), torch::zeros_like(fsa.Scores())));
    EXPECT_TRUE(torch::equal(
        fsa.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        expected_arcs));

    RaggedAny labels_any("[[1 3 5] [2 6] [8 7 9]]", py::cast(torch::kInt32),
                         device);

    fsa = LinearFsa(labels_any);
    EXPECT_TRUE(torch::allclose(fsa.Scores(), torch::zeros_like(fsa.Scores())));
    EXPECT_TRUE(torch::equal(
        fsa.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        expected_arcs));
  }
}

TEST(GraphTest, LevenshteinGraphs) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    auto device = GetDevice(c);
    float ins_del_score = -0.51;
    RaggedAny symbols("[[1 2 3] [] [4 5 6]]", py::cast(torch::kInt32), device);
    RaggedArc graph = LevenshteinGraphs(symbols, ins_del_score);
    torch::Tensor expected_arcs = torch::tensor(
        {
            {0, 0, 0}, {0, 1, 0},  {0, 1, 1}, {1, 1, 0}, {1, 2, 0},  {1, 2, 2},
            {2, 2, 0}, {2, 3, 0},  {2, 3, 3}, {3, 3, 0}, {3, 4, -1},  // fsa 0
            {0, 0, 0}, {0, 1, -1},                                    // fsa 1
            {0, 0, 0}, {0, 1, 0},  {0, 1, 4}, {1, 1, 0}, {1, 2, 0},  {1, 2, 5},
            {2, 2, 0}, {2, 3, 0},  {2, 3, 6}, {3, 3, 0}, {3, 4, -1}  // fsa 2
        },
        torch::dtype(torch::kInt32).device(device));

    torch::Tensor expected_scores =
        torch::tensor({-0.51, -0.5,  0.0, -0.51, -0.5, 0.0,   -0.51, -0.5,
                       0.0,   -0.51, 0.0, -0.51, 0.0,  -0.51, -0.5,  0.0,
                       -0.51, -0.5,  0.0, -0.51, -0.5, 0.0,   -0.51, 0.0},
                      torch::dtype(torch::kFloat32).device(device));

    torch::Tensor expected_aux_labels =
        torch::tensor({0,  1, 1, 0, 2, 2, 0, 3, 3, 0, -1, 0,
                       -1, 0, 4, 4, 0, 5, 5, 0, 6, 6, 0,  -1},
                      torch::dtype(torch::kInt32).device(device));

    torch::Tensor expected_value_offset =
        torch::tensor({-0.01, 0.0,   0.0, -0.01, 0.0, 0.0,   -0.01, 0.0,
                       0.0,   -0.01, 0.0, -0.01, 0.0, -0.01, 0.0,   0.0,
                       -0.01, 0.0,   0.0, -0.01, 0.0, 0.0,   -0.01, 0.0},
                      torch::dtype(torch::kFloat32).device(device));

    EXPECT_TRUE(torch::allclose(graph.Scores(), expected_scores));
    EXPECT_TRUE(torch::equal(
        graph.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        expected_arcs));
    EXPECT_TRUE(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                             expected_aux_labels));
    EXPECT_TRUE(
        torch::allclose(graph.GetAttr("__ins_del_score_offset_internal_attr_")
                            .cast<torch::Tensor>(),
                        expected_value_offset));

    // test std::vector as the argument
    std::vector<std::vector<int32_t>> symbols2({{1, 2, 3}, {}, {4, 5, 6}});
    graph = LevenshteinGraphs(symbols2, ins_del_score, device);

    EXPECT_TRUE(torch::allclose(graph.Scores(), expected_scores));
    EXPECT_TRUE(torch::equal(
        graph.Arcs().index(
            {"...", torch::indexing::Slice(torch::indexing::None, 3)}),
        expected_arcs));
    EXPECT_TRUE(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                             expected_aux_labels));
    EXPECT_TRUE(
        torch::allclose(graph.GetAttr("__ins_del_score_offset_internal_attr_")
                            .cast<torch::Tensor>(),
                        expected_value_offset));
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
