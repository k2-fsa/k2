#!/usr/bin/env python3
#
# Copyright      2023  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R swoosh_l_test_py

import unittest

import k2
import torch


class SwooshLFunction(torch.autograd.Function):
    """
    swoosh(x) =  log(1 + exp(x-4)) - 0.08*x - 0.035
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        requires_grad = x.requires_grad
        x_dtype = x.dtype

        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        coeff = -0.08

        with torch.cuda.amp.autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035

                if not requires_grad:
                    return y
                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = coeff
                ceil = 1.0 + coeff + 0.005

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                    grad
                )
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.float16)
                return y

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor) -> torch.Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.

        coeff = -0.08
        floor = coeff
        ceil = 1.0 + coeff + 0.005
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshL(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swoosh-L activation."""
        if torch.jit.is_scripting():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035
        return SwooshLFunction.apply(x)


class TestSwooshL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test(self):
        for device in self.devices:
            for shape in [(10,), (2, 3), (4, 5, 6), (7, 8, 9, 10)]:
                torch_x = torch.rand(*shape).to(device)  # .requires_grad_(True)
                k2_x = torch_x.detach().clone()

                torch_swoosh_l = SwooshL()
                torch_y = torch_swoosh_l(torch_x)

                k2_y = k2.swoosh_l(x=k2_x, dropout_prob=0.0)
                assert torch.allclose(torch_y, k2_y), (torch_y - k2_y).abs().max()
                print(torch_y.sum(), k2_y.sum())


if __name__ == "__main__":
    torch.manual_seed(20230116)
    unittest.main()
