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
#include "k2/csrc/macros.h"
#include "k2/csrc/rnnt_decode.h"

namespace k2 {
namespace rnnt_decoding {
TEST(RnntDecodeStream, CreateRnntDecodeStream) {
  for (const auto &c : {GetCpuContext(), GetCudaContext()}) {
    Array1<int32_t> aux_labels;
    auto graph = std::make_shared<Fsa>(TrivialGraph(c, 5, &aux_labels));
    auto stream = CreateStream(graph);
    K2_CHECK(Equal(*graph, *(stream->graph)));
    K2_CHECK(Equal(stream->states, Ragged<int64_t>(c, "[[0]]")));
    K2_CHECK(Equal(stream->scores, Ragged<double>(c, "[[0]]")));
    K2_CHECK_EQ(stream->num_graph_states, graph->Dim0());
  }
}

static void ApplyLog(Ragged<float> &probs) {
  float *probs_data = probs.values.Data();
  K2_EVAL(
      probs.Context(), probs.NumElements(), lambda_set_log,
      (int32_t i) { probs_data[i] = logf(probs_data[i]); });
}

// This test does not do any checking, just to make sure it runs normally.
TEST(RnntDecodingStreams, Basic) {
  for (auto c : {GetCpuContext(), GetCudaContext()}) {
    int32_t vocab_size = 6;
    auto config =
        RnntDecodingConfig(vocab_size, 2 /*decoder_history_len*/, 5.0f /*beam*/,
                           8 /*max_states*/, 4 /*max_contexts*/);

    Array1<int32_t> aux_labels;
    auto trivial_graph = std::make_shared<Fsa>(TrivialGraph(c, 5, &aux_labels));
    auto ctc_topo = std::make_shared<Fsa>(CtcTopo(c, 5, false, &aux_labels));
    auto ctc_topo_modified =
        std::make_shared<Fsa>(CtcTopo(c, 5, true, &aux_labels));
    std::vector<std::shared_ptr<Fsa>> graphs(
        {trivial_graph, ctc_topo, ctc_topo_modified});

    int32_t num_streams = 3;
    std::vector<std::shared_ptr<RnntDecodingStream>> streams_vec(num_streams);
    for (int32_t i = 0; i < num_streams; ++i) {
      streams_vec[i] = CreateStream(graphs[RandInt(0, 2)]);
    }
    auto streams = RnntDecodingStreams(streams_vec, config);

    RaggedShape context_shape;
    Array2<int32_t> context;

    int32_t steps = 5;

    for (int32_t i = 0; i < steps; ++i) {
      streams.GetContexts(&context_shape, &context);

      auto probs = Ragged<float>(
          RegularRaggedShape(c, context_shape.NumElements(), vocab_size),
          RandUniformArray1<float>(c, context_shape.NumElements() * vocab_size,
                                   0, 1));

      probs = NormalizePerSublist<float>(probs, false /*use_log*/);

      ApplyLog(probs);

      auto logprobs =
          Array2<float>(probs.values, context_shape.NumElements(), vocab_size);

      streams.Advance(logprobs);
    }
    streams.TerminateAndFlushToStreams();

    for (auto num_frames : {std::vector<int32_t>(num_streams, steps),
                             std::vector<int32_t>({2, 5, 4})}) {
      Array1<int32_t> out_map;
      FsaVec ofsa;
      streams.FormatOutput(num_frames, true/*allow_partial*/, &ofsa, &out_map);
      Array1<int32_t> properties;
      int32_t property;
      GetFsaVecBasicProperties(ofsa, &properties, &property);
      if (!(property & kFsaPropertiesValid)) {
        std::vector<Fsa> fsas;
        Unstack(ofsa, 0, &fsas);
        auto cpu_properties = properties.To(GetCpuContext());
        for (int32_t i = 0; i < cpu_properties.Dim(); ++i) {
          K2_LOG(INFO) << (cpu_properties[i] & kFsaPropertiesValid);
          K2_LOG(INFO) << FsaToString(fsas[i]);
        }
      }
      K2_CHECK(property & kFsaPropertiesValid);
    }
  }
}
}  // namespace rnnt_decoding

}  // namespace k2
