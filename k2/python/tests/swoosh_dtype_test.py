#!/usr/bin/env python3
#
# Copyright      2026  (authors: k2 contributors)
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

# Tests that the swoosh ops keep the dtype of their input for
# float32/float16/bfloat16 (computing internally in float32), and convert
# other dtypes (float64) to float32 as before.
#
# To run this single test, use
#
#  python3 k2/python/tests/swoosh_dtype_test.py

import unittest

import k2
import torch

SHIFT = {"l": 4.0, "r": 1.0}
OFFSET = {"l": 0.035, "r": 0.313261687}


def ref_forward(x: torch.Tensor, kind: str) -> torch.Tensor:
    x = x.double()
    return torch.nn.functional.softplus(x - SHIFT[kind]) - 0.08 * x - OFFSET[kind]


def ref_deriv(x: torch.Tensor, kind: str) -> torch.Tensor:
    return torch.sigmoid(x.double() - SHIFT[kind]) - 0.08


class TestSwooshDtype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))

    def _tol(self, dtype):
        return {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]

    def test_forward(self):
        for device in self.devices:
            for kind in ("l", "r"):
                fwd = getattr(k2, f"swoosh_{kind}_forward")
                fwd_deriv = getattr(k2, f"swoosh_{kind}_forward_and_deriv")
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    x = (torch.randn(3, 1000, device=device) * 5).to(dtype)
                    tol = self._tol(dtype)
                    y = fwd(x)
                    assert y.dtype == dtype, (device, kind, dtype, y.dtype)
                    expected = ref_forward(x, kind)
                    err = (y.double() - expected).abs().max().item()
                    assert err < tol * (1 + expected.abs().max().item()), (
                        device, kind, dtype, err)

                    y2, d = fwd_deriv(x)
                    assert y2.dtype == dtype and d.dtype == dtype
                    # forward uses log1pf(e), forward_and_deriv logf(1 + e)
                    # (as upstream), so they agree only up to rounding.
                    assert torch.allclose(y2.double(), y.double(), atol=tol * 4, rtol=tol), (
                        device, kind, dtype, (y2.double() - y.double()).abs().max())
                    err = (d.double() - ref_deriv(x, kind)).abs().max().item()
                    assert err < tol, (device, kind, dtype, err)

    def test_autograd(self):
        for device in self.devices:
            for kind in ("l", "r"):
                func = getattr(k2, f"swoosh_{kind}")
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    x = (torch.randn(2000, device=device) * 5).to(dtype)
                    x.requires_grad_(True)
                    y = func(x)
                    assert y.dtype == dtype, (device, kind, dtype, y.dtype)
                    g = torch.rand_like(y)
                    (x_grad,) = torch.autograd.grad(y, x, g)
                    assert x_grad.dtype == dtype
                    expected = g.double() * ref_deriv(x.detach(), kind)
                    # the derivative is stored in 8 bits.
                    err = (x_grad.double() - expected).abs().max().item()
                    assert err < 0.01 + self._tol(dtype), (device, kind, dtype, err)

                    # without requires_grad
                    y_nograd = func(x.detach())
                    assert y_nograd.dtype == dtype
                    assert torch.allclose(
                        y_nograd.double(), y.detach().double(),
                        atol=self._tol(dtype) * 10)

    def test_dropout(self):
        for device in self.devices:
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(10000, device=device).to(dtype).requires_grad_(True)
                y = k2.swoosh_l(x, dropout_prob=0.5)
                assert y.dtype == dtype
                frac_zero = (y == 0).double().mean().item()
                assert 0.45 < frac_zero < 0.55, frac_zero
                (x_grad,) = torch.autograd.grad(y, x, torch.ones_like(y))
                assert x_grad.dtype == dtype
                # dropped elements have zero gradient
                assert (x_grad[y == 0].double().abs() < 0.01).all()

    def test_float64_falls_back_to_float32(self):
        for device in self.devices:
            x = torch.randn(10, device=device, dtype=torch.float64)
            assert k2.swoosh_l_forward(x).dtype == torch.float32
            assert k2.swoosh_r_forward_and_deriv(x)[0].dtype == torch.float32


if __name__ == "__main__":
    torch.manual_seed(20261121)
    unittest.main()
