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

import unittest

import k2
import random
import torch


# Caution: this will fail occasionally due to cutoffs not being quite large enough.
# As long as it passes most of the time, it's OK.
class TestMutualInformation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_mutual_information_basic(self):
        for _iter in range(100):
            (B, S, T) = (random.randint(1, 10), random.randint(1, 16),
                         random.randint(1, 500))
            random_px = (random.random() < 0.2)
            random_py = (random.random() < 0.2)
            random_boundary = (random.random() < 0.7)
            big_px = (random.random() < 0.2)
            big_py = (random.random() < 0.2)

            print(
                f"B, S, T = {B}, {S}, {T}, random_px={random_px}, random_py={random_py}, big_px={big_px}, big_py={big_py}, random_boundary={random_boundary}"
            )
            for dtype in [torch.float32, torch.float64]:
                px_grads = []
                py_grads = []
                m_vals = []
                for device in [torch.device('cpu'), torch.device('cuda:0')]:
                    print("dtype = ", dtype, ", device = ", device)

                    if random_boundary:

                        def get_boundary_row():
                            s_begin = random.randint(0, S - 1)
                            t_begin = random.randint(0, T - 1)
                            # allow empty sequence
                            s_end = random.randint(s_begin, S)
                            # allow empty sequence
                            t_end = random.randint(t_begin, T)
                            return [s_begin, t_begin, s_end, t_end]

                        if device == torch.device('cpu'):
                            boundary = torch.tensor(
                                [get_boundary_row() for _ in range(B)],
                                dtype=torch.int64,
                                device=device)
                        else:
                            boundary = boundary.to(device)
                    else:
                        # Use default boundary, but either specified directly or not.
                        if random.random() < 0.5:
                            boundary = torch.tensor(
                                [0, 0, S, T],
                                dtype=torch.int64).unsqueeze(0).expand(
                                    B, 4).to(device)
                        else:
                            boundary = None

                    if device == torch.device('cpu'):
                        if random_px:
                            px = torch.randn(B, S, T + 1, dtype=dtype).to(
                                device)  # log of an odds ratio
                            if S > 1 and not random_boundary:
                                px[:, :, -1:] = float('-inf')
                        else:
                            px = torch.zeros(B, S, T + 1, dtype=dtype).to(
                                device)  # log of an odds ratio
                        # px and py get exponentiated, and then multiplied together up to
                        # 32 times (BLOCK_SIZE in the CUDA code), so 15 is actually a big number that
                        # could lead to overflow.
                        if big_px:
                            px += 15.0
                        if random_py:
                            py = torch.randn(B, S + 1, T, dtype=dtype).to(
                                device)  # log of an odds ratio
                        else:
                            py = torch.zeros(B, S + 1, T, dtype=dtype).to(
                                device)  # log of an odds ratio
                        if big_py:
                            py += 15.0

                    else:
                        px = px.to(device).detach()
                        py = py.to(device).detach()
                    px.requires_grad = True
                    py.requires_grad = True

                    m, grad = k2.mutual_information_recursion(px, py, boundary)

                    m2 = k2.joint_mutual_information_recursion((px,), (py,),
                                                               boundary)

                    m3 = k2.joint_mutual_information_recursion(
                        (px * 0.5, px * 0.5), (py * 0.5, py * 0.5), boundary)

                    print("m3, before sum, = ", m3)

                    # it is supposed to be identical only after
                    # summing over dim 0, corresponding to the
                    # sequence dim
                    m3 = m3.sum(dim=0)

                    print("m = ", m, ", size = ", m.shape)
                    print("m2 = ", m2, ", size = ", m2.shape)
                    print("m3 = ", m3, ", size = ", m3.shape)

                    assert torch.allclose(m, m2)
                    assert torch.allclose(m, m3)

                    # the loop this is in checks that the CPU and CUDA versions give the same
                    # derivative; by randomizing which of m, m2 or m3 we backprop, we also
                    # ensure that the joint version of the code gives the same derivative
                    # as the regular version
                    scale = 3
                    if random.random() < 0.5:
                        (m.sum() * scale).backward()
                    elif random.random() < 0.5:
                        (m2.sum() * scale).backward()
                    else:
                        (m3.sum() * scale).backward()

                    px_grads.append(px.grad.to('cpu'))
                    py_grads.append(py.grad.to('cpu'))
                    m_vals.append(m.to('cpu'))
                if not torch.allclose(
                        m_vals[0], m_vals[1], atol=1.0e-02, rtol=1.0e-02):
                    print(
                        f"m_vals differed CPU vs CUDA: {m_vals[0]} vs. {m_vals[1]}"
                    )
                    assert 0
                if not torch.allclose(
                        px_grads[0], px_grads[1], atol=1.0e-02, rtol=1.0e-02):
                    print(
                        f"px_grads differed CPU vs CUDA: {px_grads[0]} vs. {px_grads[1]}"
                    )
                    assert 0
                if not torch.allclose(
                        py_grads[0], py_grads[1], atol=1.0e-02, rtol=1.0e-02):
                    print(
                        f"py_grads differed CPU vs CUDA: {py_grads[0]} vs. {py_grads[1]}"
                    )
                    assert 0

    def test_mutual_information_deriv(self):
        for _iter in range(100):
            (B, S, T) = (random.randint(1, 10), random.randint(1, 200),
                         random.randint(1, 200))
            random_px = (random.random() < 0.2)
            random_py = (random.random() < 0.2)
            random_boundary = (random.random() < 0.7)
            big_px = (random.random() < 0.2)
            big_py = (random.random() < 0.2)

            print(
                f"B, S, T = {B}, {S}, {T}, random_px={random_px}, random_py={random_py}, big_px={big_px}, big_py={big_py}, random_boundary={random_boundary}"
            )
            for dtype in [torch.float32, torch.float64]:
                for device in [torch.device('cpu'), torch.device('cuda:0')]:
                    print("dtype = ", dtype, ", device = ", device)

                    if random_boundary:

                        def get_boundary_row():
                            s_begin = random.randint(0, S - 1)
                            t_begin = random.randint(0, T - 1)
                            s_end = random.randint(s_begin + 1, S)
                            t_end = random.randint(t_begin + 1, T)
                            return [s_begin, t_begin, s_end, t_end]

                        if device == torch.device('cpu'):
                            boundary = torch.tensor(
                                [get_boundary_row() for _ in range(B)],
                                dtype=torch.int64,
                                device=device)
                        else:
                            boundary = boundary.to(device)
                    else:
                        # Use default boundary, but either specified directly or not.
                        if random.random() < 0.5:
                            boundary = torch.tensor(
                                [0, 0, S, T],
                                dtype=torch.int64).unsqueeze(0).expand(
                                    B, 4).to(device)
                        else:
                            boundary = None

                    if device == torch.device('cpu'):
                        if random_px:
                            px = torch.randn(B, S, T + 1, dtype=dtype).to(
                                device)  # log of an odds ratio
                        else:
                            px = torch.zeros(B, S, T + 1, dtype=dtype).to(
                                device)  # log of an odds ratio
                        # px and py get exponentiated, and then multiplied together up to
                        # 32 times (BLOCK_SIZE in the CUDA code), so 15 is actually a big number that
                        # could lead to overflow.
                        if big_px:
                            px += 15.0
                        if random_py:
                            py = torch.randn(B, S + 1, T, dtype=dtype).to(
                                device)  # log of an odds ratio
                        else:
                            py = torch.zeros(B, S + 1, T, dtype=dtype).to(
                                device)  # log of an odds ratio
                        if big_py:
                            py += 15.0

                    else:
                        px = px.to(device).detach()
                        py = py.to(device).detach()
                    px.requires_grad = True
                    py.requires_grad = True

                    m, grad = k2.mutual_information_recursion(px, py, boundary)

                    m_grad = torch.randn(B, dtype=dtype, device=device)
                    m.backward(gradient=m_grad)
                    delta = 1.0e-04
                    delta_px = delta * torch.randn_like(px)
                    m2, grad = k2.mutual_information_recursion(
                        px + delta_px, py, boundary)
                    delta_m = m2 - m
                    observed_delta = (delta_m * m_grad).sum().to('cpu')
                    predicted_delta = (delta_px * px.grad).sum().to('cpu')
                    print(
                        f"For px: observed,predicted objf changes are: {observed_delta},{predicted_delta}, absolute objf was {(m * m_grad).sum()}"
                    )

                    atol = 1.0e-02 if dtype == torch.float32 else 1.0e-04
                    rtol = 1.0e-02 if dtype == torch.float32 else 1.0e-04

                    if not torch.allclose(
                            observed_delta, predicted_delta, atol=atol,
                            rtol=rtol):
                        print(
                            f"Error: observed and predicted delta too different."
                        )
                        assert 0

                    delta_py = delta * torch.randn_like(py)
                    m2, grad = k2.mutual_information_recursion(
                        px, py + delta_py, boundary)
                    delta_m = m2 - m
                    observed_delta = (delta_m * m_grad).sum().to('cpu')
                    predicted_delta = (delta_py * py.grad).sum().to('cpu')
                    print(
                        f"For py: observed,predicted objf changes are: {observed_delta},{predicted_delta}, absolute objf was {(m * m_grad).sum()}"
                    )

                # if not torch.allclose(m_vals[0], m_vals[1], atol=1.0e-02, rtol=1.0e-02):
                #     print(f"m_vals differed CPU vs CUDA: {m_vals[0]} vs. {m_vals[1]}")
                #     assert 0
                # if not torch.allclose(px_grads[0], px_grads[1], atol=1.0e-02, rtol=1.0e-02):
                #     print(f"px_grads differed CPU vs CUDA: {px_grads[0]} vs. {px_grads[1]}")
                #     assert 0
                # if not torch.allclose(py_grads[0], py_grads[1], atol=1.0e-02, rtol=1.0e-02):
                #     print(f"py_grads differed CPU vs CUDA: {py_grads[0]} vs. {py_grads[1]}")
                #     assert 0


if __name__ == "__main__":
    unittest.main()
