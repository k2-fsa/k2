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
    A Tensor. If its dtype is float32, float16 or bfloat16, the returned
    tensor and the gradient have the same dtype (computation is done in
    float32); otherwise x is converted to float32 first.
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
    A Tensor. If its dtype is float32, float16 or bfloat16, the returned
    tensor and the gradient have the same dtype (computation is done in
    float32); otherwise x is converted to float32 first.
  dropout_prob:
    A float number. The default value is 0.
)doc";

static constexpr const char *kSwooshLForwardDoc = R"doc(
Compute ``swoosh_l(x) = log(1 + exp(x-4)) - 0.08x - 0.035``.

Args:
  x:
    A Tensor. If its dtype is float32, float16 or bfloat16, the returned
    tensor(s) have the same dtype (computation is done in float32);
    otherwise x is converted to float32 first.
)doc";

static constexpr const char *kSwooshRForwardDoc = R"doc(
Compute ``swoosh_r(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687``.

Args:
  x:
    A Tensor. If its dtype is float32, float16 or bfloat16, the returned
    tensor(s) have the same dtype (computation is done in float32);
    otherwise x is converted to float32 first.
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
    A Tensor. If its dtype is float32, float16 or bfloat16, the returned
    tensor(s) have the same dtype (computation is done in float32);
    otherwise x is converted to float32 first.
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
    A Tensor. If its dtype is float32, float16 or bfloat16, the returned
    tensor(s) have the same dtype (computation is done in float32);
    otherwise x is converted to float32 first.
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
// Floating point types supported by the swoosh kernels.  The computation is
// always done in float32 registers; inputs are read and outputs written in
// their native dtype, so that half/bfloat16 training (autocast) does not need
// extra cast kernels and float32 temporaries.
static bool IsSwooshSupportedDtype(torch::ScalarType t) {
  return t == torch::kFloat || t == torch::kHalf || t == torch::kBFloat16;
}

#define K2_SWOOSH_DISPATCH(scalar_type, T, ...)                          \
  do {                                                                   \
    switch (scalar_type) {                                               \
      case torch::kFloat: {                                              \
        using T = float;                                                 \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case torch::kHalf: {                                               \
        using T = at::Half;                                              \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case torch::kBFloat16: {                                           \
        using T = at::BFloat16;                                          \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      default:                                                           \
        K2_LOG(FATAL) << "Unsupported dtype for swoosh: " << scalar_type; \
    }                                                                    \
  } while (0)

// Returns x if it has a supported floating point dtype, else x converted to
// float32 (e.g. for float64 or integer inputs, which keep the old behavior of
// returning float32).
static torch::Tensor SwooshInput(torch::Tensor x) {
  if (!IsSwooshSupportedDtype(x.scalar_type())) x = x.to(torch::kFloat32);
  return x.contiguous();
}

template <typename SwooshConstants, typename T>
static void SwooshForwardWithGradKernel(ContextPtr context, int32_t n,
                                        const T *x_data, const float *r_data,
                                        const float *r2_data,
                                        float dropout_prob, T *y_data,
                                        uint8_t *g_data) {
  float shift = SwooshConstants::kShift;
  float coeff = SwooshConstants::kCoeff;
  float offset = SwooshConstants::kOffset;
  K2_EVAL(
      context, n, lambda_compute_swoosh_forward, (int32_t i)->void {
        float xi = static_cast<float>(x_data[i]);
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

        y_data[i] = static_cast<T>(yi);
        uint8_t g_int = (uint8_t)(gi * (255.0f / 1.005f) + r2_data[i]);
        g_data[i] = g_int;
      });
}

template <typename SwooshConstants, typename T>
static void SwooshBackwardKernel(ContextPtr context, int32_t n,
                                 const T *out_grad_data,
                                 const uint8_t *g_data, float dropout_prob,
                                 T *in_grad_data) {
  float coeff = SwooshConstants::kCoeff;
  K2_EVAL(
      context, n, lambda_compute_swoosh_backward, (int32_t i)->void {
        float oi = static_cast<float>(out_grad_data[i]);
        float gi = g_data[i];  // cast uint8_t to float
        // functional grad
        float fi = ((gi * (1.005f / 255.0f)) - coeff) / (1.0f - dropout_prob);
        in_grad_data[i] = static_cast<T>(oi * fi);
      });
}

template <typename SwooshConstants, typename T>
static void SwooshForwardKernel(ContextPtr context, int32_t n,
                                const T *x_data, T *y_data) {
  float shift = SwooshConstants::kShift;
  float coeff = SwooshConstants::kCoeff;
  float offset = SwooshConstants::kOffset;
  K2_EVAL(
      context, n, lambda_swoosh_forward, (int32_t i)->void {
        float xi = static_cast<float>(x_data[i]);
        float xi_offset = xi - shift;
        float e = expf(xi_offset);
        float log_sum = log1pf(e);
        if (isinf(log_sum)) {
          log_sum = xi_offset;
        }
        y_data[i] = static_cast<T>(log_sum - coeff * xi - offset);
      });
}

template <typename SwooshConstants, typename T>
static void SwooshForwardAndDerivKernel(ContextPtr context, int32_t n,
                                        const T *x_data, T *y_data,
                                        T *d_data) {
  float shift = SwooshConstants::kShift;
  float coeff = SwooshConstants::kCoeff;
  float offset = SwooshConstants::kOffset;
  K2_EVAL(
      context, n, lambda_swoosh_forward_and_deriv, (int32_t i)->void {
        float xi = static_cast<float>(x_data[i]);
        float xi_offset = xi - shift;
        float denom = 1.0f + expf(xi_offset);
        float inv_denom = 1.0f / denom;
        float di = (1.0f - coeff) - inv_denom;  // deriv
        float log_denom = logf(denom);
        if (isinf(log_denom)) {
          log_denom = xi_offset;
        }
        y_data[i] = static_cast<T>(log_denom - coeff * xi - offset);
        d_data[i] = static_cast<T>(di);
      });
}

/**

swoosh(x) = log(1 + exp(x - kShift)) - kCoeff * x - kOffset

For swooshL:
  - kShift=4, kCoeff=0.08, kOffset=0.035
  swooshL(x) = log(1 + exp(x-4)) - 0.08x - 0.035

For swooshR:
  - kShift=1, kCoeff=0.08, kOffset=0.313261687
  swooshR(x) = log(1 + exp(x-1)) - 0.08x - 0.313261687

The output (and the gradient w.r.t. x) has the same dtype as x if x is
float32, float16 or bfloat16 (computation is done in float32); other dtypes
are converted to float32 first.
 */
template <typename SwooshConstants>
torch::Tensor SwooshForward(torch::Tensor x);

template <typename SwooshConstants>
class SwooshFunction
    : public torch::autograd::Function<SwooshFunction<SwooshConstants>> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor x, float dropout_prob) {
    auto context = GetContext(x);
    DeviceGuard guard(context);

    bool requires_grad = x.requires_grad();

    if (!requires_grad) {
      return SwooshForward<SwooshConstants>(x);
    }

    x = SwooshInput(x);

    torch::Tensor y = torch::empty_like(x).contiguous();

    auto float_opts = x.options().dtype(torch::kFloat32);
    // for dropout
    torch::Tensor r;
    const float *r_data = nullptr;
    if (dropout_prob != 0.0f) {
      r = torch::rand(x.sizes(), float_opts).contiguous();
      r_data = r.data_ptr<float>();
    }

    // for uint8 quantization
    torch::Tensor r2 = torch::rand(x.numel(), float_opts).contiguous();
    const float *r2_data = r2.data_ptr<float>();

    auto opts = torch::TensorOptions().dtype(torch::kByte).device(x.device());
    torch::Tensor g = torch::empty(x.sizes(), opts).contiguous();
    uint8_t *g_data = g.data_ptr<uint8_t>();

    int32_t n = x.numel();
    K2_SWOOSH_DISPATCH(x.scalar_type(), T,
                       SwooshForwardWithGradKernel<SwooshConstants, T>(
                           context, n, x.data_ptr<T>(), r_data, r2_data,
                           dropout_prob, y.data_ptr<T>(), g_data));

    ctx->saved_data["dropout_prob"] = dropout_prob;
    ctx->saved_data["x_dtype"] = x.scalar_type();
    ctx->save_for_backward({g});

    return y;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list y_grad) {
    float dropout_prob = ctx->saved_data["dropout_prob"].toDouble();
    auto x_dtype = ctx->saved_data["x_dtype"].toScalarType();
    auto saved = ctx->get_saved_variables();

    torch::Tensor g = saved[0];
    const uint8_t *g_data = g.data_ptr<uint8_t>();

    // The gradient is returned in the dtype of x (autograd would cast it to
    // that dtype anyway).
    torch::Tensor out_grad = y_grad[0].to(x_dtype).contiguous();

    auto context = GetContext(out_grad);
    DeviceGuard guard(context);

    torch::Tensor in_grad = torch::empty(g.sizes(), out_grad.options());

    int32_t n = g.numel();
    K2_SWOOSH_DISPATCH(x_dtype, T,
                       SwooshBackwardKernel<SwooshConstants, T>(
                           context, n, out_grad.data_ptr<T>(), g_data,
                           dropout_prob, in_grad.data_ptr<T>()));

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

  x = SwooshInput(x);
  torch::Tensor y = torch::empty_like(x).contiguous();

  int32_t n = x.numel();
  K2_SWOOSH_DISPATCH(x.scalar_type(), T,
                     SwooshForwardKernel<SwooshConstants, T>(
                         context, n, x.data_ptr<T>(), y.data_ptr<T>()));
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

  x = SwooshInput(x);
  torch::Tensor y = torch::empty_like(x).contiguous();
  torch::Tensor deriv = torch::empty_like(x).contiguous();

  int32_t n = x.numel();
  K2_SWOOSH_DISPATCH(
      x.scalar_type(), T,
      SwooshForwardAndDerivKernel<SwooshConstants, T>(
          context, n, x.data_ptr<T>(), y.data_ptr<T>(), deriv.data_ptr<T>()));

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
