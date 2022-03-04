/**
 * @copyright
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey)
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

#include "k2/csrc/utils.h"  // for LogAdd
#include "k2/python/csrc/torch/mutual_information.h"

namespace k2 {

// forward of mutual_information.  See """... """ comment of
// `mutual_information_recursion` in
// in k2/python/k2/mutual_information.py for documentation of the
// behavior of this function.

// px: of shape [B, S, T+1] if !modified, else [B, S, T]  <-- work out
// `modified` from this.
// py: of shape [B, S+1, T]
// boundary: of shape [B, 4], containing (s_begin, t_begin, s_end, t_end)
//  defaulting to (0, 0, S, T).
// p: of shape (S+1, T+1)
// Computes the recursion:
// if !modified:
//             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
//                                p[b,s,t-1] + py[b,s,t-1])
// if modified:
//             p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
//                                p[b,s,t-1] + py[b,s,t-1])

// .. treating out-of-range elements as -infinity and with special cases:
// p[b, s_begin, t_begin] = 0.0
//
// and this function returns a tensor of shape (B,) consisting of elements
//  p[b, s_end, t_end]
torch::Tensor MutualInformationCpu(torch::Tensor px, torch::Tensor py,
                                   torch::optional<torch::Tensor> opt_boundary,
                                   torch::Tensor p) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(
      px.device().is_cpu() && py.device().is_cpu() && p.device().is_cpu(),
      "inputs must be CPU tensors");

  bool modified = (px.size(2) == py.size(2));

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0), S = px.size(1), T = py.size(2);
  TORCH_CHECK(px.size(2) == (modified ? T : T + 1));
  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);

  auto boundary = opt_boundary.value_or(
      torch::tensor({0, 0, S, T},
                    torch::dtype(torch::kInt64).device(torch::kCPU))
          .reshape({1, 4})
          .expand({B, 4}));
  TORCH_CHECK(boundary.dim() == 2, "boundary must be 2-dimensional.");
  TORCH_CHECK(boundary.size(0) == B && boundary.size(1) == 4);
  TORCH_CHECK(boundary.device().is_cpu() && boundary.dtype() == torch::kInt64);

  torch::Tensor ans = torch::empty({B}, opts);

  AT_DISPATCH_FLOATING_TYPES(
      px.scalar_type(), "mutual_information_cpu_loop", ([&] {
        auto px_a = px.accessor<scalar_t, 3>(),
             py_a = py.accessor<scalar_t, 3>(), p_a = p.accessor<scalar_t, 3>();
        auto boundary_a = boundary.accessor<int64_t, 2>();
        auto ans_a = ans.accessor<scalar_t, 1>();

        int t_offset = (modified ? -1 : 0);
        for (int b = 0; b < B; b++) {
          int s_begin = boundary_a[b][0];
          int t_begin = boundary_a[b][1];
          int s_end = boundary_a[b][2];
          int t_end = boundary_a[b][3];
          p_a[b][s_begin][t_begin] = 0.0;
          if (modified) {
            for (int s = s_begin + 1; s <= s_end; ++s)
              p_a[b][s][t_begin] = -std::numeric_limits<scalar_t>::infinity();
          } else {
            // note: t_offset = 0 so don't need t_begin + t_offset below.
            for (int s = s_begin + 1; s <= s_end; ++s)
              p_a[b][s][t_begin] =
                  p_a[b][s - 1][t_begin] + px_a[b][s - 1][t_begin];
          }
          for (int t = t_begin + 1; t <= t_end; ++t)
            p_a[b][s_begin][t] =
                p_a[b][s_begin][t - 1] + py_a[b][s_begin][t - 1];
          for (int s = s_begin + 1; s <= s_end; ++s) {
            scalar_t p_s_t1 = p_a[b][s][t_begin];
            for (int t = t_begin + 1; t <= t_end; ++t) {
              // The following statement is a small optimization of:
              // p_a[b][s][t] = LogAdd(
              //    p_a[b][s - 1][t + t_offset] + px_a[b][s -1][t + t_offset],
              //    p_a[b][s][t - 1] + py_a[b][s][t - 1]);
              // .. which obtains p_a[b][s][t - 1] from a register.
              p_a[b][s][t] = p_s_t1 = LogAdd<scalar_t>()(
                  p_a[b][s - 1][t + t_offset] + px_a[b][s - 1][t + t_offset],
                  p_s_t1 + py_a[b][s][t - 1]);
            }
          }
          ans_a[b] = p_a[b][s_end][t_end];
        }
      }));
  return ans;
}

// backward of mutual_information.  Returns (px_grad, py_grad).
// p corresponds to what we computed in the forward pass.
std::vector<torch::Tensor> MutualInformationBackwardCpu(
    torch::Tensor px, torch::Tensor py,
    torch::optional<torch::Tensor> opt_boundary, torch::Tensor p,
    torch::Tensor ans_grad) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(ans_grad.dim() == 1, "ans_grad must be 1-dimensional.");

  bool modified = (px.size(2) == py.size(2));

  TORCH_CHECK(px.device().is_cpu() && py.device().is_cpu() &&
                  p.device().is_cpu() && ans_grad.device().is_cpu(),
              "inputs must be CPU tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0), S = px.size(1), T = py.size(2);
  TORCH_CHECK(px.size(2) == (modified ? T : T + 1));
  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);

  auto boundary = opt_boundary.value_or(
      torch::tensor({0, 0, S, T},
                    torch::dtype(torch::kInt64).device(torch::kCPU))
          .reshape({1, 4})
          .expand({B, 4}));
  TORCH_CHECK(boundary.dim() == 2, "boundary must be 2-dimensional.");
  TORCH_CHECK(boundary.size(0) == B && boundary.size(1) == 4);
  TORCH_CHECK(boundary.device().is_cpu() && boundary.dtype() == torch::kInt64);

  bool has_boundary = opt_boundary.has_value();
  int T1 = T + (modified ? 0 : 1);
  torch::Tensor p_grad = torch::zeros({B, S + 1, T + 1}, opts),
                px_grad = (has_boundary ? torch::zeros({B, S, T1}, opts)
                                        : torch::empty({B, S, T1}, opts)),
                py_grad = (has_boundary ? torch::zeros({B, S + 1, T}, opts)
                                        : torch::empty({B, S + 1, T}, opts));

  AT_DISPATCH_FLOATING_TYPES(
      px.scalar_type(), "mutual_information_cpu_backward_loop", ([&] {
        auto px_a = px.accessor<scalar_t, 3>(), p_a = p.accessor<scalar_t, 3>(),
             p_grad_a = p_grad.accessor<scalar_t, 3>(),
             px_grad_a = px_grad.accessor<scalar_t, 3>(),
             py_grad_a = py_grad.accessor<scalar_t, 3>();

        auto ans_grad_a = ans_grad.accessor<scalar_t, 1>();
        auto boundary_a = boundary.accessor<int64_t, 2>();
        int t_offset = (modified ? -1 : 0);

        for (int b = 0; b < B; b++) {
          int s_begin = boundary_a[b][0];
          int t_begin = boundary_a[b][1];
          int s_end = boundary_a[b][2];
          int t_end = boundary_a[b][3];
          // Backprop for: ans_a[b] = p_a[b][s_end][t_end];
          p_grad_a[b][s_end][t_end] = ans_grad_a[b];

          for (int s = s_end; s > s_begin; --s) {
            for (int t = t_end; t > t_begin; --t) {
              // The s,t indexes correspond to
              // The statement we are backpropagating here is:
              // p_a[b][s][t] = LogAdd(
              //    p_a[b][s - 1][t + t_offset] + px_a[b][s - 1][t + t_offset],
              //    p_a[b][s][t - 1] + py_a[b][s][t - 1]);
              // .. which obtains p_a[b][s][t - 1] from a register.
              scalar_t term1 = p_a[b][s - 1][t + t_offset] +
                               px_a[b][s - 1][t + t_offset],
                       // term2 = p_a[b][s][t - 1] + py_a[b][s][t - 1], <-- not
                       // actually needed..
                  total = p_a[b][s][t];
              if (total - total != 0) total = 0;
              scalar_t term1_deriv = exp(term1 - total),
                       term2_deriv = 1.0 - term1_deriv,
                       grad = p_grad_a[b][s][t];
              scalar_t term1_grad, term2_grad;
              if (term1_deriv - term1_deriv == 0.0) {
                term1_grad = term1_deriv * grad;
                term2_grad = term2_deriv * grad;
              } else {
                // could happen if total == -inf
                term1_grad = term2_grad = 0.0;
              }
              px_grad_a[b][s - 1][t + t_offset] = term1_grad;
              p_grad_a[b][s - 1][t + t_offset] = term1_grad;
              py_grad_a[b][s][t - 1] = term2_grad;
              p_grad_a[b][s][t - 1] += term2_grad;
            }
          }
          for (int t = t_end; t > t_begin; --t) {
            // Backprop for:
            // p_a[b][s_begin][t] =
            //     p_a[b][s_begin][t - 1] + py_a[b][s_begin][t - 1];
            scalar_t this_p_grad = p_grad_a[b][s_begin][t];
            p_grad_a[b][s_begin][t - 1] += this_p_grad;
            py_grad_a[b][s_begin][t - 1] = this_p_grad;
          }
          if (!modified) {
            for (int s = s_end; s > s_begin; --s) {
              // Backprop for:
              // p_a[b][s][t_begin] =
              //    p_a[b][s - 1][t_begin] + px_a[b][s - 1][t_begin];
              scalar_t this_p_grad = p_grad_a[b][s][t_begin];
              p_grad_a[b][s - 1][t_begin] += this_p_grad;
              px_grad_a[b][s - 1][t_begin] = this_p_grad;
            }
          }  // else these were all -infinity's and there is nothing to
             // backprop.
          // There is no backprop for:
          // p_a[b][s_begin][t_begin] = 0.0;
          // .. but we can use this for a check, that the grad at the beginning
          // of the sequence is equal to the grad at the end of the sequence.
          if (ans_grad_a[b] != 0.0) {
            float grad_ratio = p_grad_a[b][s_begin][t_begin] / ans_grad_a[b];
            if (fabs(grad_ratio - 1.0) > 0.01) {
              K2_LOG(WARNING)
                  << "Warning: mutual_information backprop: expected these "
                  << "numbers to be the same:"
                  << static_cast<float>(p_grad_a[b][s_begin][t_begin]) << " vs "
                  << static_cast<float>(ans_grad_a[b]);
            }
          }
        }
      }));

  return std::vector<torch::Tensor>({px_grad, py_grad});
}
}  // namespace k2
