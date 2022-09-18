/**
 * @brief python wrappers for rnnt_decode.h
 *
 * @copyright
 * Copyright      2022  Xiaomi Corp.       (authors: Wei Kang)
 *
 * @copyright
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

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/rnnt_decode.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/rnnt_decode.h"

namespace k2 {
static void PybindRnntDecodingConfig(py::module &m) {
  using PyClass = rnnt_decoding::RnntDecodingConfig;
  py::class_<PyClass> config(m, "RnntDecodingConfig");
  config.def(py::init<int32_t, int32_t, double, int32_t, int32_t>(),
             py::arg("vocab_size"), py::arg("decoder_history_len"),
             py::arg("beam"), py::arg("max_states"), py::arg("max_contexts"),
             R"(
             Construct a RnntDecodingConfig object, it contains the parameters
             needed by rnnt decoding.

             Args:
               vocab_size:
                 It indicates how many symbols we are using, equals the
                 largest-symbol plus one.
               decoder_history_len:
                 The number of symbols of history the
                 decoder takes; will normally be one or two
                 ("stateless decoder"), our RNN-T decoding setup does not
                 support unlimited decoder context such as with LSTMs.
               beam:
                 `beam` imposes a limit on the score of a state, relative to the
                 best-scoring state on the same frame.  E.g. 10.
               max_states:
                 `max_states` is a limit on the number of distinct states that
                 we allow per frame, per stream; the number of states will not
                 be allowed to exceed this limit.
               max_contexts:
                 `max_contexts` is a limit on the number of distinct contexts
                 that we allow per frame, per stream; the number of contexts
                 will not be allowed to exceed this limit.
             )");

  config.def_readwrite("vocab_size", &PyClass::vocab_size)
      .def_readwrite("decoder_history_len", &PyClass::decoder_history_len)
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("max_states", &PyClass::max_states)
      .def_readwrite("max_contexts", &PyClass::max_contexts);

  config.def("__str__", [](const PyClass &self) -> std::string {
    std::ostringstream os;
    os << "RnntDecodingConfig : {\n"
       << "  vocab_size : " << self.vocab_size << "\n"
       << "  decoder_history_len : " << self.decoder_history_len << "\n"
       << "  beam : " << self.beam << "\n"
       << "  max_states : " << self.max_states << "\n"
       << "  max_contexts : " << self.max_contexts << "\n"
       << "}";
    return os.str();
  });
}

static void PybindRnntDecodingStream(py::module &m) {
  using PyClass = rnnt_decoding::RnntDecodingStream;
  py::class_<PyClass, std::shared_ptr<PyClass>> stream(m, "RnntDecodingStream");

  stream.def("__str__", [](const PyClass &self) -> std::string {
    std::ostringstream os;
    os << "RnntDecodingStream : {\n"
       << "  num graph states : " << self.graph->Dim0() << "\n"
       << "  num graph arcs : " << self.graph->NumElements() << "\n"
       << "  num contexts : " << self.states.Dim0() << "\n"
       << "  num states : " << self.states.NumElements() << "\n"
       << "  num prev frames : " << self.prev_frames.size() << "\n"
       << "}";
    return os.str();
  });

  m.def("create_rnnt_decoding_stream",
        [](Fsa &graph) -> std::shared_ptr<PyClass> {
          DeviceGuard guard(graph.Context());
          return rnnt_decoding::CreateStream(std::make_shared<Fsa>(graph));
        });
}

static void PybindRnntDecodingStreams(py::module &m) {
  using PyClass = rnnt_decoding::RnntDecodingStreams;
  py::class_<PyClass> streams(m, "RnntDecodingStreams");

  streams.def(py::init(
      [](std::vector<std::shared_ptr<rnnt_decoding::RnntDecodingStream>> &srcs,
         const rnnt_decoding::RnntDecodingConfig &config)
          -> std::unique_ptr<PyClass> {
        K2_CHECK_GE(srcs.size(), 1);
        DeviceGuard guard(srcs[0]->graph->Context());
        return std::make_unique<PyClass>(srcs, config);
      }));

  streams.def("advance", [](PyClass &self, torch::Tensor logprobs) -> void {
    DeviceGuard guard(self.Context());
    logprobs = logprobs.to(torch::kFloat);
    Array2<float> logprobs_array = FromTorch<float>(logprobs, Array2Tag{});
    self.Advance(logprobs_array);
  });

  streams.def("get_contexts",
              [](PyClass &self) -> std::pair<RaggedShape, torch::Tensor> {
                DeviceGuard guard(self.Context());
                RaggedShape shape;
                Array2<int32_t> contexts;
                self.GetContexts(&shape, &contexts);
                torch::Tensor contexts_tensor = ToTorch<int32_t>(contexts);
                return std::make_pair(shape, contexts_tensor);
              });

  streams.def("terminate_and_flush_to_streams", [](PyClass &self) -> void {
    DeviceGuard guard(self.Context());
    self.TerminateAndFlushToStreams();
  });

  streams.def("format_output",
              [](PyClass &self, std::vector<int32_t> &num_frames,
                 bool allow_partial) -> std::pair<FsaVec, torch::Tensor> {
                DeviceGuard guard(self.Context());
                FsaVec ofsa;
                Array1<int32_t> out_map;
                self.FormatOutput(num_frames, allow_partial, &ofsa, &out_map);
                torch::Tensor out_map_tensor = ToTorch<int32_t>(out_map);
                return std::make_pair(ofsa, out_map_tensor);
              });
}

}  // namespace k2

void PybindRnntDecode(py::module &m) {
  k2::PybindRnntDecodingConfig(m);
  k2::PybindRnntDecodingStream(m);
  k2::PybindRnntDecodingStreams(m);
}
