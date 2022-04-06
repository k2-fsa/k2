#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation   (authors: Daniel Povey,
#                                                     Wei Kang)
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
#  ctest --verbose -R mutual_information_test_py

import random
import unittest

import k2
import torch


# Caution: this will fail occasionally due to cutoffs not being quite large
# enough. As long as it passes most of the time, it's OK.
class TestMutualInformation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))
        cls.dtypes = [torch.float32, torch.float64]

    def test_mutual_information_basic(self):
        for _iter in range(100):
            (B, S, T) = (
                random.randint(1, 10),
                random.randint(1, 16),
                random.randint(1, 500),
            )
            random_px = random.random() < 0.2
            random_py = random.random() < 0.2
            random_boundary = random.random() < 0.7
            big_px = random.random() < 0.2
            big_py = random.random() < 0.2

            modified = random.random() < 0.5

            if modified and T < S:
                T = S + random.randint(0, 30)

            for dtype in self.dtypes:
                for device in self.devices:
                    if random_boundary:

                        def get_boundary_row():
                            this_S = random.randint(
                                0, S
                            )  # allow empty sequence
                            this_T = random.randint(
                                this_S if modified else 1, T
                            )
                            s_begin = random.randint(0, S - this_S)
                            t_begin = random.randint(0, T - this_T)
                            s_end = s_begin + this_S
                            t_end = t_begin + this_T
                            return [s_begin, t_begin, s_end, t_end]

                        if device == torch.device("cpu"):
                            boundary = torch.tensor(
                                [get_boundary_row() for _ in range(B)],
                                dtype=torch.int64,
                                device=device,
                            )
                        else:
                            boundary = boundary.to(device)
                    else:
                        # Use default boundary, but either specified directly
                        # or not.
                        if random.random() < 0.5:
                            boundary = (
                                torch.tensor([0, 0, S, T], dtype=torch.int64)
                                .unsqueeze(0)
                                .expand(B, 4)
                                .to(device)
                            )
                        else:
                            boundary = None

                    if device == torch.device("cpu"):
                        if random_px:
                            # log of an odds ratio
                            px = torch.randn(
                                B, S, T + (0 if modified else 1), dtype=dtype
                            ).to(device)
                            if S > 1 and not random_boundary and not modified:
                                px[:, :, -1:] = float("-inf")
                        else:
                            # log of an odds ratio
                            px = torch.zeros(
                                B, S, T + (0 if modified else 1), dtype=dtype
                            ).to(device)
                        # px and py get exponentiated, and then multiplied
                        # together up to 32 times (BLOCK_SIZE in the CUDA code),
                        # so 15 is actually a big number that could lead to
                        # overflow.
                        if big_px:
                            px += 15.0
                        if random_py:
                            # log of an odds ratio
                            py = torch.randn(B, S + 1, T, dtype=dtype).to(
                                device
                            )
                        else:
                            # log of an odds ratio
                            py = torch.zeros(B, S + 1, T, dtype=dtype).to(
                                device
                            )
                        if big_py:
                            py += 15.0

                    else:
                        px = px.to(device).detach()
                        py = py.to(device).detach()
                    px.requires_grad = True
                    py.requires_grad = True

                    m = k2.mutual_information_recursion(px, py, boundary)

                    m2 = k2.joint_mutual_information_recursion(
                        (px,), (py,), boundary
                    )

                    m3 = k2.joint_mutual_information_recursion(
                        (px * 0.5, px * 0.5), (py * 0.5, py * 0.5), boundary
                    )

                    # it is supposed to be identical only after
                    # summing over dim 0, corresponding to the
                    # sequence dim
                    m3 = m3.sum(dim=0)

                    assert torch.allclose(m, m2)
                    assert torch.allclose(m, m3)

                    # the loop this is in checks that the CPU and CUDA versions
                    # give the same derivative;
                    # by randomizing which of m, m2 or m3 we backprop, we also
                    # ensure that the joint version of the code gives the same
                    # derivative as the regular version
                    scale = 3
                    if random.random() < 0.5:
                        (m.sum() * scale).backward()
                    elif random.random() < 0.5:
                        (m2.sum() * scale).backward()
                    else:
                        (m3.sum() * scale).backward()

                    if device == torch.device("cpu"):
                        expected_px_grad = px.grad
                        expected_py_grad = py.grad
                        expected_m = m
                    assert torch.allclose(
                        px.grad,
                        expected_px_grad.to(device),
                        atol=1.0e-02,
                        rtol=1.0e-02,
                    )
                    assert torch.allclose(
                        py.grad,
                        expected_py_grad.to(device),
                        atol=1.0e-02,
                        rtol=1.0e-02,
                    )
                    assert torch.allclose(
                        m, expected_m.to(device), atol=1.0e-02, rtol=1.0e-02
                    )

    def test_mutual_information_deriv(self):
        for _iter in range(100):
            (B, S, T) = (
                random.randint(1, 100),
                random.randint(1, 200),
                random.randint(1, 200),
            )
            random_px = random.random() < 0.2
            random_py = random.random() < 0.2
            random_boundary = random.random() < 0.7
            big_px = random.random() < 0.2
            big_py = random.random() < 0.2

            modified = random.random() < 0.5

            if modified and T < S:
                T = S + random.randint(0, 30)

            for dtype in self.dtypes:
                for device in self.devices:

                    if random_boundary:

                        def get_boundary_row():
                            this_S = random.randint(1, S)
                            this_T = random.randint(
                                this_S if modified else 1, T
                            )
                            s_begin = random.randint(0, S - this_S)
                            t_begin = random.randint(0, T - this_T)
                            s_end = s_begin + this_S
                            t_end = t_begin + this_T
                            return [s_begin, t_begin, s_end, t_end]

                        if device == torch.device("cpu"):
                            boundary = torch.tensor(
                                [get_boundary_row() for _ in range(B)],
                                dtype=torch.int64,
                                device=device,
                            )
                        else:
                            boundary = boundary.to(device)
                    else:
                        # Use default boundary, but either specified directly
                        # or not.
                        if random.random() < 0.5:
                            boundary = (
                                torch.tensor([0, 0, S, T], dtype=torch.int64)
                                .unsqueeze(0)
                                .expand(B, 4)
                                .to(device)
                            )
                        else:
                            boundary = None

                    T1 = T + (0 if modified else 1)
                    if device == torch.device("cpu"):
                        if random_px:
                            # log of an odds ratio
                            px = torch.randn(B, S, T1, dtype=dtype).to(device)
                        else:
                            # log of an odds ratio
                            px = torch.zeros(B, S, T1, dtype=dtype).to(device)
                        # px and py get exponentiated, and then multiplied
                        # together up to 32 times (BLOCK_SIZE in the CUDA code),
                        # so 15 is actually a big number that could lead to
                        # overflow.
                        if big_px:
                            px += 15.0
                        if random_py:
                            # log of an odds ratio
                            py = torch.randn(B, S + 1, T, dtype=dtype).to(
                                device
                            )
                        else:
                            # log of an odds ratio
                            py = torch.zeros(B, S + 1, T, dtype=dtype).to(
                                device
                            )
                        if big_py:
                            py += 15.0
                    else:
                        px = px.to(device).detach()
                        py = py.to(device).detach()
                    px.requires_grad = True
                    py.requires_grad = True

                    m = k2.mutual_information_recursion(px, py, boundary)

                    m_grad = torch.randn(B, dtype=dtype, device=device)
                    m.backward(gradient=m_grad)
                    delta = 1.0e-04
                    delta_px = delta * torch.randn_like(px)
                    m2 = k2.mutual_information_recursion(
                        px + delta_px, py, boundary
                    )
                    delta_m = m2 - m
                    observed_delta = (delta_m * m_grad).sum().to("cpu")
                    predicted_delta = (delta_px * px.grad).sum().to("cpu")

                    atol = 1.0e-01
                    rtol = atol

                    assert torch.allclose(
                        observed_delta, predicted_delta, atol=atol, rtol=rtol
                    ), (observed_delta, predicted_delta)

                    delta_py = delta * torch.randn_like(py)
                    m2 = k2.mutual_information_recursion(
                        px, py + delta_py, boundary
                    )
                    delta_m = m2 - m
                    observed_delta = (delta_m * m_grad).sum().to("cpu")
                    predicted_delta = (delta_py * py.grad).sum().to("cpu")

                    assert torch.allclose(
                        observed_delta, predicted_delta, atol=atol, rtol=rtol
                    )


if __name__ == "__main__":
    unittest.main()
