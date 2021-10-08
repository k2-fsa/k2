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
    Ragged<Arc> topo_ref(
        c,
        "[ [ 0 0 0 0 0 1 1 0 0 2 2 0 0 3 3 0 0 4 -1 0 ] "
        "  [ 1 0 0 0 1 1 1 0 1 2 2 0 1 3 3 0 1 4 -1 0 ] "
        "  [ 2 0 0 0 2 1 1 0 2 2 2 0 2 3 3 0 2 4 -1 0 ] "
        "  [ 3 0 0 0 3 1 1 0 3 2 2 0 3 3 3 0 3 4 -1 0 ] [ ] ]");
    torch::Tensor aux_label_ref = torch::tensor(
        {0, 1, 2, 3, -1, 0, 0, 2, 3, -1, 0, 1, 0, 3, -1, 0, 1, 2, 0, -1},
        torch::dtype(torch::kInt32).device(device));
    K2_CHECK(Equal(topo.fsa, topo_ref));
    K2_CHECK(torch::equal(topo.GetAttr("aux_labels").cast<torch::Tensor>(),
                          aux_label_ref));

    // Test simplified topology
    topo = CtcTopo(3, device, true);
    topo_ref = Ragged<Arc>(c,
                           "[ [ 0 0 0 0 0 0 1 0 0 0 2 0 0 0 3 0 "
                           "    0 1 1 0 0 2 2 0 0 3 3 0 0 4 -1 0 ]"
                           "  [ 1 1 1 0 1 0 1 0 ]"
                           "  [ 2 2 2 0 2 0 2 0 ]"
                           "  [ 3 3 3 0 3 0 3 0 ] [ ] ]");
    aux_label_ref = torch::tensor({0, 1, 2, 3, 1, 2, 3, -1, 0, 0, 0, 0, 0, 0},
                                  torch::dtype(torch::kInt32).device(device));
    K2_CHECK(Equal(topo.fsa, topo_ref));
    K2_CHECK(torch::equal(topo.GetAttr("aux_labels").cast<torch::Tensor>(),
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
    Ragged<Arc> graph_ref(
        c,
        "[ [ [ 0 0 0 0 0 1 1 0 ] [ 1 2 0 0 1 1 1 0 1 3 2 0 ] "
        "    [ 2 2 0 0 2 3 2 0 ] [ 3 4 0 0 3 3 2 0 ] "
        "    [ 4 4 0 0 4 5 2 0 ] [ 5 6 0 0 5 5 2 0 5 7 3 0 ] "
        "    [ 6 6 0 0 6 7 3 0 ] [ 7 8 0 0 7 7 3 0 7 9 -1 0 ] "
        "    [ 8 8 0 0 8 9 -1 0 ] [ ] ] "
        "  [ [ 0 0 0 0 0 1 1 0 ] [ 1 2 0 0 1 1 1 0 1 3 2 0 ] "
        "    [ 2 2 0 0 2 3 2 0 ] [ 3 4 0 0 3 3 2 0 3 5 3 0 ] "
        "    [ 4 4 0 0 4 5 3 0 ] [ 5 6 0 0 5 5 3 0 5 7 -1 0 ] "
        "    [ 6 6 0 0 6 7 -1 0 ] [ ] ] ]");
    torch::Tensor aux_labels_ref =
        torch::tensor({0, 1, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0,
                       0, 0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0},
                      torch::dtype(torch::kInt32).device(device));
    K2_CHECK(Equal(graph.fsa, graph_ref));
    K2_CHECK(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                          aux_labels_ref));
    // test modified
    graph = CtcGraphs(symbols, true);
    graph_ref =
        Ragged<Arc>(c,
                    "[ [ [ 0 0 0 0 0 1 1 0 ] [ 1 2 0 0 1 1 1 0 1 3 2 0 ] "
                    "    [ 2 2 0 0 2 3 2 0 ] [ 3 4 0 0 3 3 2 0 3 5 2 0] "
                    "    [ 4 4 0 0 4 5 2 0 ] [ 5 6 0 0 5 5 2 0 5 7 3 0 ] "
                    "    [ 6 6 0 0 6 7 3 0 ] [ 7 8 0 0 7 7 3 0 7 9 -1 0 ] "
                    "    [ 8 8 0 0 8 9 -1 0 ] [ ] ] "
                    "  [ [ 0 0 0 0 0 1 1 0 ] [ 1 2 0 0 1 1 1 0 1 3 2 0 ] "
                    "    [ 2 2 0 0 2 3 2 0 ] [ 3 4 0 0 3 3 2 0 3 5 3 0 ] "
                    "    [ 4 4 0 0 4 5 3 0 ] [ 5 6 0 0 5 5 3 0 5 7 -1 0 ] "
                    "    [ 6 6 0 0 6 7 -1 0 ] [ ] ] ]");
    aux_labels_ref = torch::tensor(
        {0, 1, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0},
        torch::dtype(torch::kInt32).device(device));

    K2_CHECK(Equal(graph.fsa, graph_ref));
    K2_CHECK(torch::equal(graph.GetAttr("aux_labels").cast<torch::Tensor>(),
                          aux_labels_ref));
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
