/**
 * @copyright
 * Copyright      2023  Xiaomi Corp.  (authors: Zengwei Yao, Fangjun Kuang)
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_SWOOSH_H_
#define K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_SWOOSH_H_

#include <utility>

#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch.h"

namespace k2 {

/**

swoosh(x) = log(1 + exp(x - kShift)) - kCoeff * x - kOffset

For swooshL:
  - kShift=4, kCoeff=0.08, kOffset=0.035
  swooshL(x) = log(1 + exp(x-4)) - 0.08x - 0.035

For swooshR:
  - kShift=1, kCoeff=0.08, kOffset=0.313261687
  swooshR(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687
 */
template <typename SwooshConstants>
class SwooshFunction
    : public torch::autograd::Function<SwooshFunction<SwooshConstants>> {
 public:
  static constexpr float kShift = SwooshConstants::kShift;
  static constexpr float kCoeff = SwooshConstants::kCoeff;
  static constexpr float kOffset = SwooshConstants::kOffset;

  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor x, float dropout_prob) {
    bool requires_grad = x.requires_grad();
    auto x_dtype = x.dtype();

    x = x.to(torch::kFloat32);

    if (!requires_grad) {
      torch::Tensor zero = torch::zeros({1}, x.options());
      return torch::logaddexp(zero, x - kShift) - kCoeff * x - kOffset;
    }

    x = x.contiguous();

    torch::Tensor y = torch::empty_like(x).contiguous();

    // for dropout
    torch::Tensor r;
    if (dropout_prob != 0.0f) {
      r = torch::rand_like(x).contiguous();
    }

    // for uint8 quantization
    torch::Tensor r2 = torch::rand_like(x).contiguous();

    auto context = GetContext(x);
    DeviceGuard guard(context);

    const float *x_data = x.data_ptr<float>();
    const float *r_data = r.data_ptr<float>();
    const float *r2_data = r2.data_ptr<float>();

    torch::Tensor g = torch::empty(x.sizes(), torch::kByte).to(x.device());

    float *y_data = x.data_ptr<float>();
    uint8_t *g_data = g.data_ptr<uint8_t>();

    K2_EVAL(
        context, x.numel(), lambda_compute_swoosh_forward, (int32_t i)->void {
          float xi = x_data[i];
          float yi = xi;  // will be the swoosh output
          float gi = 1;   // will be the gradient of log(1+exp(x-kShift))
          if (xi < 10) {
            float e = expf(xi - kShift);
            gi = e / (1.0f + e);
            float l = log1pf(e);
            yi = l - kCoeff * xi - kOffset;
          }
          // else just keep yi = xi, g=1 if xi >= 10
          if (dropout_prob != 0.0f) {
            float ri = r_data[i];
            if (ri < dropout_prob) {
              yi = 0;
              //  gi currently represents swoosh'(x) + kCoeff, so
              //  this value corresponds to swoosh'(x) = 0
              gi = kCoeff;
            }
          }

          y_data[i] = yi;
          unsigned char g_int =
              (unsigned char)(gi * (255.0f / 1.005f) + r2_data[i]);
          g_data[i] = g_int;
        });

    ctx->saved_data["dropout_prob"] = dropout_prob;
    ctx->save_for_backward({g});

    return y;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list y_grad) {
    float dropout_prob = ctx->saved_data["dropout_prob"].toDouble();
    auto saved = ctx->get_saved_variables();
    torch::Tensor g = saved[0];

    // please follow ./normalize.h and other header files in this folder
    return {};
  }
};

void PybindSwoosh(py::module &m);

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_AUTOGRAD_SWOOSH_H_
