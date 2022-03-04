/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei kang)
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
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/rnnt_decode.h"

namespace k2 {
namespace rnnt_decoding {
TEST(RnntDecodeStream, CreateRnntDecodeStream) {
  for (const auto &c : {GetCpuContext(), GetCudaContext()}) {
    auto graph = std::make_shared<Fsa>(TrivialGraph(c, 5));
    auto stream = CreateStream(graph);
    K2_CHECK(Equal(*graph, *(stream->graph)));
    K2_CHECK(Equal(stream->states, Ragged<int64_t>(c, "[[0]]")));
    K2_CHECK(Equal(stream->scores, Ragged<double>(c, "[[0]]")));
    K2_CHECK_EQ(stream->num_graph_states, graph->Dim0());
  }
}

// This test does not do any checking, just to make sure it runs normally.
TEST(RnntDecodingStreams, Basic) {
  for (const auto &c : {GetCpuContext(), GetCudaContext()}) {
    int32_t vocab_size = 6;
    auto config =
        RnntDecodingConfig(vocab_size, 2 /*decoder_history_len*/, 8.0f /*beam*/,
                           2 /*max_states*/, 3 /*max_contexts*/);

    auto trivial_graph = std::make_shared<Fsa>(TrivialGraph(c, 5));
    Array1<int32_t> aux_label;
    auto ctc_topo = std::make_shared<Fsa>(CtcTopo(c, 5, false, &aux_label));
    auto ctc_topo_modified =
        std::make_shared<Fsa>(CtcTopo(c, 5, true, &aux_label));
    std::vector<std::shared_ptr<Fsa>> graphs(
        {trivial_graph, ctc_topo, ctc_topo_modified});

    int32_t num_streams = 3;
    std::vector<std::shared_ptr<RnntDecodingStream>> streams_vec(num_streams);
    for (int32_t i = 0; i < num_streams; ++i) {
      streams_vec[i] = CreateStream(graphs[RandInt(0,2)]);
    }
    auto streams = RnntDecodingStreams(streams_vec, config);

    K2_LOG(INFO) << "states : " << streams.states_;
    K2_LOG(INFO) << "scores : " << streams.scores_;
    K2_LOG(INFO) << "num_graph_states : " << streams.num_graph_states_;

    float mean = 5, std = 3;
    RaggedShape context_shape;
    Array2<int32_t> context;

    int32_t steps = 5;

    for (int32_t i = 0; i < steps; ++i) {
      streams.GetContexts(&context_shape, &context);
      K2_LOG(INFO) << "context_shape : " << context_shape;
      K2_LOG(INFO) << "context : " << context;
      auto logprobs = RandGaussianArray2<float>(c, context_shape.NumElements(),
                                                vocab_size, mean, std);
      K2_LOG(INFO) << "logprobs : " << logprobs;
      streams.Advance(logprobs);
      K2_LOG(INFO) << "states : " << streams.states_;
      K2_LOG(INFO) << "scores : " << streams.scores_;
    }
    Array1<int32_t> out_map;
    FsaVec ofsa;
    streams.FormatOutput(&ofsa, &out_map);
    K2_LOG(INFO) << "ofsa : " << ofsa;
    K2_LOG(INFO) << "out map : " << out_map;
    std::vector<Fsa> fsas;
    Unstack(ofsa, 0, &fsas);
    K2_LOG(INFO) << FsaToString(fsas[0]);
  }
}

}  // namespace rnnt_decoding
}  // namespace k2
