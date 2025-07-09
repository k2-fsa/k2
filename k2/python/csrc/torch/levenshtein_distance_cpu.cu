/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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

#include <algorithm>

#include "k2/python/csrc/torch/levenshtein_distance.h"

namespace k2 {

torch::Tensor LevenshteinDistanceCpu(
    torch::Tensor px, torch::Tensor py,
    torch::optional<torch::Tensor> opt_boundary) {
  TORCH_CHECK(px.dim() == 2, "px must be 2-dimensional");
  TORCH_CHECK(py.dim() == 2, "py must be 2-dimensional.");
  TORCH_CHECK(px.device().is_cpu() && py.device().is_cpu(),
              "inputs must be CPU tensors");
  TORCH_CHECK(px.dtype() == torch::kInt32 && py.dtype() == torch::kInt32,
              "The dtype of inputs must be kInt32");

  auto opts = torch::TensorOptions().dtype(px.dtype()).device(px.device());

  const int B = px.size(0), S = px.size(1), U = py.size(1);
  TORCH_CHECK(B == py.size(0), "px and py must have same batch size");

  auto boundary = opt_boundary.value_or(
      torch::tensor({0, 0, S, U},
                    torch::dtype(torch::kInt64).device(torch::kCPU))
          .reshape({1, 4})
          .expand({B, 4}));
  TORCH_CHECK(boundary.dim() == 2, "boundary must be 2-dimensional.");
  TORCH_CHECK(boundary.size(0) == B && boundary.size(1) == 4);
  TORCH_CHECK(boundary.device().is_cpu() && boundary.dtype() == torch::kInt64);

  torch::Tensor ans = torch::empty({B, S + 1, U + 1}, opts);

  auto px_a = px.accessor<int32_t, 2>(), py_a = py.accessor<int32_t, 2>();
  auto boundary_a = boundary.accessor<int64_t, 2>();
  auto ans_a = ans.accessor<int32_t, 3>();

  for (int b = 0; b < B; b++) {
    int s_begin = boundary_a[b][0];
    int u_begin = boundary_a[b][1];
    int s_end = boundary_a[b][2];
    int u_end = boundary_a[b][3];
    ans_a[b][s_begin][u_begin] = 0;

    for (int s = s_begin + 1; s <= s_end; ++s)
      ans_a[b][s][u_begin] = s - s_begin;
    for (int u = u_begin + 1; u <= u_end; ++u)
      ans_a[b][s_begin][u] = u - u_begin;

    for (int s = s_begin + 1; s <= s_end; ++s) {
      for (int u = u_begin + 1; u <= u_end; ++u) {
        int cost = px_a[b][s - 1] == py_a[b][u - 1] ? 0 : 1;
        ans_a[b][s][u] =
            min(min(ans_a[b][s - 1][u] + 1, ans_a[b][s][u - 1] + 1),
                ans_a[b][s - 1][u - 1] + cost);
      }
    }
  }
  return ans;
}

}  // namespace k2
