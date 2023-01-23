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

#include <type_traits>
#include "k2/python/csrc/torch/v2/autograd/swoosh.h"

namespace k2 {

struct SwooshLConstants {
  static constexpr float kShift = 4;
  static constexpr float kCoeff = 0.08;
  static constexpr float kOffset = 0.035;
};

struct SwooshRConstants {
  static constexpr float kShift = 1;
  static constexpr float kCoeff = 0.08;
  static constexpr float kOffset = 0.313261687;
};

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

    torch::Tensor r_total = torch::rand(x.numel() * 2, x.options()).contiguous();
    // for dropout
    const float *r_data = r_total.data_ptr<float>();
    // for uint8 quantization
    const float *r2_data = r_data + x.numel();

    ContextPtr context = GetContext(x);

    const float *x_data = x.data_ptr<float>();

    auto opts = torch::TensorOptions()
                    .dtype(torch::kByte)
                    .device(x.device());
    torch::Tensor g = torch::empty(x.sizes(), opts).contiguous();

    float *y_data = y.data_ptr<float>();
    uint8_t *g_data = g.data_ptr<uint8_t>();

    float shift = kShift;
    float coeff = kCoeff;
    float offset = kOffset;

    K2_EVAL(
        context, x.numel(), lambda_compute_swoosh_forward, (int32_t i)->void {
          float xi = x_data[i];
          float yi = xi;  // will be the swoosh output
          float gi = 1;   // will be the gradient of log(1+exp(x-kShift))
          float xi_offset = xi - shift;
          float e = expf(xi_offset);
          // gi = e / (1.0f + e);
          gi = 1.0f - 1.0f / (1.0f + e);
          float l = log1pf(e);
          if (isinf(l)) {
            l = xi_offset;
          }
          yi = l - coeff * xi - offset;
          if (dropout_prob != 0.0f) {
            float ri = r_data[i];
            if (ri < dropout_prob) {
              yi = 0;
              //  gi currently represents swoosh'(x) + kCoeff, so
              //  this value corresponds to swoosh'(x) = 0
              gi = coeff;
            } else {
              yi *= 1.0f / (1.0f - dropout_prob);
            }
          }

          y_data[i] = yi;
          uint8_t g_int =
              (uint8_t)(gi * (255.0f / 1.005f) + r2_data[i]);
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
    const uint8_t *g_data = g.data_ptr<uint8_t>();

    torch::Tensor out_grad = y_grad[0];
    out_grad = out_grad.contiguous();

    int32_t stride = out_grad.stride(-1);
    const float *out_grad_data = out_grad.data_ptr<float>();

    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(g.device());
    torch::Tensor in_grad = torch::empty(g.sizes(), opts).contiguous();
    float *in_grad_data = in_grad.data_ptr<float>();

    ContextPtr context = GetContext(out_grad);
    float coeff = kCoeff;
    K2_EVAL(
        context, g.numel(), lambda_compute_swoosh_backward, (int32_t i)->void {
          float oi = out_grad_data[i * stride];
          float gi = g_data[i];  // cast uint8_t to float
          float ii = 1;
          float fi = 1;  // functional grad

          fi = ((gi * (1.005f / 255.0f)) - coeff) / (1.0f - dropout_prob);
          ii = oi * fi;
          in_grad_data[i] = ii;
        });

    return {
      in_grad,  // x
      torch::Tensor() //  dropout_prob
    };
  }
};

using SwooshLFunction = SwooshFunction<SwooshLConstants>;
using SwooshRFunction = SwooshFunction<SwooshRConstants>;

template class SwooshFunction<SwooshLConstants>;
template class SwooshFunction<SwooshRConstants>;

void PybindSwoosh(py::module &m) {
  m.def(
      "swoosh_l",
      [](torch::Tensor x, float dropout_prob) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshLFunction::apply(x, dropout_prob);
      },
      py::arg("x"), py::arg("dropout_prob"));

  m.def(
      "swoosh_r",
      [](torch::Tensor x, float dropout_prob) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshRFunction::apply(x, dropout_prob);
      },
      py::arg("x"), py::arg("dropout_prob"));
}

}  // namespace k2
