/**
 * @brief wraps nbest code.
 *
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
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

#include <tuple>

#include "k2/csrc/context.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nbest.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/nbest.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

static void PybindGetBestMatchingStats(py::module &m) {
  m.def(
      "get_best_matching_stats",
      [](RaggedAny &ragged, torch::Tensor scores, torch::Tensor counts,
         int32_t eos, int32_t min_token, int32_t max_token,
         int32_t max_order) -> std::tuple<torch::Tensor, torch::Tensor,
                                          torch::Tensor, torch::Tensor> {
        DeviceGuard guard(ragged.any.Context());
        Ragged<int32_t> tokens = ragged.any.Specialize<int32_t>();
        Array1<float> scores_array = FromTorch<float>(scores);
        Array1<int32_t> counts_array = FromTorch<int32_t>(counts);
        Array1<float> mean, var;
        Array1<int32_t> counts_out, ngram_order;
        GetBestMatchingStats(tokens, scores_array, counts_array, eos, min_token,
                             max_token, max_order, &mean, &var, &counts_out,
                             &ngram_order);
        return std::make_tuple(ToTorch(mean), ToTorch(var), ToTorch(counts_out),
                               ToTorch(ngram_order));
      },
      py::arg("tokens"), py::arg("scores"), py::arg("counts"), py::arg("eos"),
      py::arg("min_token"), py::arg("max_token"), py::arg("max_order"));
}

}  // namespace k2

void PybindNbest(py::module &m) { k2::PybindGetBestMatchingStats(m); }
