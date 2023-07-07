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

static constexpr const char *kSwooshLDoc = R"doc(
Compute ``swoosh_l(x) = log(1 + exp(x-4)) - 0.08x - 0.035``,
and optionally apply dropout.
If x.requires_grad is True, it returns ``dropout(swoosh_l(l))``.
In order to reduce momory, the function derivative ``swoosh_l'(x)``
is encoded into 8-bits.
If x.requires_grad is False, it returns ``swoosh_l(x)``.

Args:
  x:
    A Tensor.
  dropout_prob:
    A float number. The default value is 0.
)doc";

static constexpr const char *kSwooshRDoc = R"doc(
Compute ``swoosh_r(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687``,
and optionally apply dropout.
If x.requires_grad is True, it returns ``dropout(swoosh_r(l))``.
In order to reduce momory, the function derivative ``swoosh_r'(x)``
is encoded into 8-bits.
If x.requires_grad is False, it returns ``swoosh_r(x)``.

Args:
  x:
    A Tensor.
  dropout_prob:
    A float number. The default value is 0.
)doc";

static constexpr const char *kSwooshLForwardDoc = R"doc(
Compute ``swoosh_l(x) = log(1 + exp(x-4)) - 0.08x - 0.035``.

Args:
  x:
    A Tensor.
)doc";

static constexpr const char *kSwooshRForwardDoc = R"doc(
Compute ``swoosh_r(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687``.

Args:
  x:
    A Tensor.
)doc";

static constexpr const char *kSwooshLForwardAndDerivDoc = R"doc(
Compute ``swoosh_l(x) = log(1 + exp(x-4)) - 0.08x - 0.035``,
and also the derivative ``swoosh_l'(x) = 0.92 - 1 / (1 + exp(x-4))``.

Note:

  .. math::

    \text{swoosh_l}'(x) &= -0.08 + \exp(x-4) / (1 + \exp(x-4)) \\
                        &= -0.08 + (1 -  1 / (1 + \exp(x-4))) \\
                        &= 0.92 - 1 / (1 + \exp(x-4))

  ``1 + exp(x-4)`` might be infinity, but ``1 / (1 + exp(x-4))`` will
  be 0 in that case. This is partly why we rearranged the expression above, to
  avoid infinity / infinity = nan.

Args:
  x:
    A Tensor.
)doc";

static constexpr const char *kSwooshRForwardAndDerivDoc = R"doc(
Compute ``swoosh_r(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687``,
and also the derivative ``swoosh_r'(x) = 0.92 - 1 / (1 + exp(x-1))``.

Note:

  .. math::

    \text{swoosh_r}'(x) &= -0.08 + \exp(x-1) / (1 + \exp(x-1)) \\
                        &= -0.08 + (1 -  1 / (1 + \exp(x-1))) \\
                        &= 0.92 - 1 / (1 + \exp(x-1))

  ``1 + exp(x-1)`` might be infinity, but ``1 / (1 + exp(x-1))`` will
  be 0 in that case. This is partly why we rearranged the expression above, to
  avoid infinity / infinity = nan.

Args:
  x:
    A Tensor.
)doc";

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
    auto context = GetContext(x);
    DeviceGuard guard(context);

    bool requires_grad = x.requires_grad();

    x = x.to(torch::kFloat32);

    if (!requires_grad) {
      torch::Tensor zero = torch::zeros({1}, x.options());
      return torch::logaddexp(zero, x - kShift) - kCoeff * x - kOffset;
    }

    x = x.contiguous();

    torch::Tensor y = torch::empty_like(x).contiguous();

    // for dropout
    torch::Tensor r;
    const float *r_data = nullptr;
    if (dropout_prob != 0.0f) {
      r = torch::rand_like(x).contiguous();
      r_data = r.data_ptr<float>();
    }

    // for uint8 quantization
    torch::Tensor r2 = torch::rand(x.numel(), x.options()).contiguous();
    const float *r2_data = r2.data_ptr<float>();

    const float *x_data = x.data_ptr<float>();

    auto opts = torch::TensorOptions().dtype(torch::kByte).device(x.device());
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
          uint8_t g_int = (uint8_t)(gi * (255.0f / 1.005f) + r2_data[i]);
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

    auto context = GetContext(out_grad);
    DeviceGuard guard(context);

    auto opts =
        torch::TensorOptions().dtype(torch::kFloat32).device(g.device());
    torch::Tensor in_grad = torch::empty(g.sizes(), opts).contiguous();
    float *in_grad_data = in_grad.data_ptr<float>();

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
        in_grad,         // x
        torch::Tensor()  //  dropout_prob
    };
  }
};

using SwooshLFunction = SwooshFunction<SwooshLConstants>;
using SwooshRFunction = SwooshFunction<SwooshRConstants>;

template class SwooshFunction<SwooshLConstants>;
template class SwooshFunction<SwooshRConstants>;

// simple version of SwooshL that does not redefine the backprop, used in
// ActivationDropoutAndLinearFunction.
template <typename SwooshConstants>
torch::Tensor SwooshForward(torch::Tensor x) {
  auto context = GetContext(x);
  DeviceGuard guard(context);

  static constexpr float kShift = SwooshConstants::kShift;
  static constexpr float kCoeff = SwooshConstants::kCoeff;
  static constexpr float kOffset = SwooshConstants::kOffset;

  x = x.to(torch::kFloat32).contiguous();
  const float *x_data = x.data_ptr<float>();

  torch::Tensor y = torch::empty_like(x).contiguous();
  float *y_data = y.data_ptr<float>();

  float shift = kShift;
  float coeff = kCoeff;
  float offset = kOffset;

  K2_EVAL(
      context, x.numel(), lambda_swoosh_forward, (int32_t i)->void {
        float xi = x_data[i];
        float yi = xi;  // will be the swoosh output
        float xi_offset = xi - shift;
        float e = expf(xi_offset);
        float log_sum = log1pf(e);
        if (isinf(log_sum)) {
          log_sum = xi_offset;
        }
        yi = log_sum - coeff * xi - offset;
        y_data[i] = yi;
      });

  return y;
}

// swooshl(x) = log(1 + exp(x-4)) - 0.08 * x - 0.035
// x_deriv = -0.08 + exp(x-4) / (1 + exp(x-4))
//         = -0.08 + (1 -  1 / (1 + exp(x-4)))
//         = 0.92 - 1 / (1 + exp(x-4))
// note: 1 + exp(x_offset) might be infinity, but 1 / (1 + exp(x_offset)) will
// be 0 in that case.  This is partly why we rearranged the expression above, to
// avoid infinity / infinity = nan.
template <typename SwooshConstants>
std::pair<torch::Tensor, torch::Tensor> SwooshForwardAndDeriv(
    torch::Tensor x) {
  auto context = GetContext(x);
  DeviceGuard guard(context);

  static constexpr float kShift = SwooshConstants::kShift;
  static constexpr float kCoeff = SwooshConstants::kCoeff;
  static constexpr float kOffset = SwooshConstants::kOffset;

  x = x.to(torch::kFloat32).contiguous();
  const float *x_data = x.data_ptr<float>();

  torch::Tensor y = torch::empty_like(x).contiguous();
  float *y_data = y.data_ptr<float>();

  torch::Tensor deriv = torch::empty_like(x).contiguous();
  float *d_data = deriv.data_ptr<float>();

  float shift = kShift;
  float coeff = kCoeff;
  float offset = kOffset;

  K2_EVAL(
      context, x.numel(), lambda_swoosh_forward_and_deriv, (int32_t i)->void {
        float xi = x_data[i];
        float yi = xi;  // will be the swoosh output
        float xi_offset = xi - shift;
        float denom = 1.0f + expf(xi_offset);
        float inv_denom = 1.0f / denom;
        float di = 0.92f - inv_denom;  // deriv
        float log_denom = logf(denom);
        if (isinf(log_denom)) {
          log_denom = xi_offset;
        }
        yi = log_denom - coeff * xi - offset;
        y_data[i] = yi;
        d_data[i] = di;
      });

  return {y, deriv};
}

void PybindSwoosh(py::module &m) {
  m.def(
      "swoosh_l",
      [](torch::Tensor x, float dropout_prob) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshLFunction::apply(x, dropout_prob);
      },
      py::arg("x"), py::arg("dropout_prob") = 0.0f, kSwooshLDoc);

  m.def(
      "swoosh_r",
      [](torch::Tensor x, float dropout_prob) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshRFunction::apply(x, dropout_prob);
      },
      py::arg("x"), py::arg("dropout_prob") = 0.0f, kSwooshRDoc);

  m.def(
      "swoosh_l_forward",
      [](torch::Tensor x) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshForward<SwooshLConstants>(x);
      },
      py::arg("x"), kSwooshLForwardDoc);

  m.def(
      "swoosh_r_forward",
      [](torch::Tensor x) -> torch::Tensor {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshForward<SwooshRConstants>(x);
      },
      py::arg("x"), kSwooshRForwardDoc);

  m.def(
      "swoosh_l_forward_and_deriv",
      [](torch::Tensor x) -> std::pair<torch::Tensor, torch::Tensor> {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshForwardAndDeriv<SwooshLConstants>(x);
      },
      py::arg("x"), kSwooshLForwardAndDerivDoc);

  m.def(
      "swoosh_r_forward_and_deriv",
      [](torch::Tensor x) -> std::pair<torch::Tensor, torch::Tensor> {
        auto context = GetContext(x);
        DeviceGuard guard(context);
        return SwooshForwardAndDeriv<SwooshRConstants>(x);
      },
      py::arg("x"), kSwooshRForwardAndDerivDoc);
}

}  // namespace k2
