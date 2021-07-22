#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R ctc_loss_test_py

from typing import List

import unittest

import k2
import torch


def _visualize_ctc_topo():
    '''See https://git.io/JtqyJ
    for what the resulting ctc_topo looks like.
    '''
    symbols = k2.SymbolTable.from_str('''
        <blk> 0
        a 1
        b 2
    ''')
    aux_symbols = k2.SymbolTable.from_str('''
        a 1
        b 2
    ''')
    ctc_topo = k2.ctc_topo(2)
    ctc_topo.labels_sym = symbols
    ctc_topo.aux_labels_sym = aux_symbols
    ctc_topo.draw('ctc_topo.pdf')


# Test cases are modified from
# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
#
#
# The CTC losses computed by warp-ctc, PyTorch, and k2 are identical.
#
# The gradients with respect to network outputs are also identical
# for PyTorch and k2.
class TestCtcLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_case1(self):
        for device in self.devices:
            # suppose we have four symbols: <blk>, a, b, c, d
            torch_activation = torch.tensor([0.2, 0.2, 0.2, 0.2,
                                             0.2]).to(device)
            k2_activation = torch_activation.detach().clone()

            # (T, N, C)
            torch_activation = torch_activation.reshape(
                1, 1, -1).requires_grad_(True)

            # (N, T, C)
            k2_activation = k2_activation.reshape(1, 1,
                                                  -1).requires_grad_(True)

            torch_log_probs = torch.nn.functional.log_softmax(
                torch_activation, dim=-1)  # (T, N, C)

            # we have only one sequence and its label is `a`
            targets = torch.tensor([1]).to(device)
            input_lengths = torch.tensor([1]).to(device)
            target_lengths = torch.tensor([1]).to(device)
            torch_loss = torch.nn.functional.ctc_loss(
                log_probs=torch_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='mean')

            assert torch.allclose(torch_loss,
                                  torch.tensor([1.6094379425049]).to(device))

            # (N, T, C)
            k2_log_probs = torch.nn.functional.log_softmax(k2_activation,
                                                           dim=-1)

            supervision_segments = torch.tensor([[0, 0, 1]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = k2.ctc_topo(4)
            linear_fsa = k2.linear_fsa([1])
            decoding_graph = k2.compose(ctc_topo, linear_fsa).to(device)

            k2_loss = k2.ctc_loss(decoding_graph,
                                  dense_fsa_vec,
                                  reduction='mean',
                                  target_lengths=target_lengths)

            assert torch.allclose(torch_loss, k2_loss)

            torch_loss.backward()
            k2_loss.backward()
            assert torch.allclose(torch_activation.grad, k2_activation.grad)

    def test_case2(self):
        for device in self.devices:
            # (T, N, C)
            torch_activation = torch.arange(1, 16).reshape(1, 3, 5).permute(
                1, 0, 2).to(device)
            torch_activation = torch_activation.to(torch.float32)
            torch_activation.requires_grad_(True)

            k2_activation = torch_activation.detach().clone().requires_grad_(
                True)

            torch_log_probs = torch.nn.functional.log_softmax(
                torch_activation, dim=-1)  # (T, N, C)
            # we have only one sequence and its labels are `c,c`
            targets = torch.tensor([3, 3]).to(device)
            input_lengths = torch.tensor([3]).to(device)
            target_lengths = torch.tensor([2]).to(device)

            torch_loss = torch.nn.functional.ctc_loss(
                log_probs=torch_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='mean')

            act = k2_activation.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
            k2_log_probs = torch.nn.functional.log_softmax(act, dim=-1)

            supervision_segments = torch.tensor([[0, 0, 3]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = k2.ctc_topo(4)
            linear_fsa = k2.linear_fsa([3, 3])
            decoding_graph = k2.compose(ctc_topo, linear_fsa).to(device)

            k2_loss = k2.ctc_loss(decoding_graph,
                                  dense_fsa_vec,
                                  reduction='mean',
                                  target_lengths=target_lengths)

            expected_loss = torch.tensor([7.355742931366],
                                         device=device) / target_lengths
            assert torch.allclose(torch_loss, k2_loss)
            assert torch.allclose(torch_loss, expected_loss)

            torch_loss.backward()
            k2_loss.backward()
            assert torch.allclose(torch_activation.grad, k2_activation.grad)

    def test_case3(self):
        for device in self.devices:
            # (T, N, C)
            torch_activation = torch.tensor([[
                [-5, -4, -3, -2, -1],
                [-10, -9, -8, -7, -6],
                [-15, -14, -13, -12, -11.],
            ]]).permute(1, 0, 2).to(device).requires_grad_(True)
            torch_activation = torch_activation.to(torch.float32)
            torch_activation.requires_grad_(True)

            k2_activation = torch_activation.detach().clone().requires_grad_(
                True)

            torch_log_probs = torch.nn.functional.log_softmax(
                torch_activation, dim=-1)  # (T, N, C)
            # we have only one sequence and its labels are `b,c`
            targets = torch.tensor([2, 3]).to(device)
            input_lengths = torch.tensor([3]).to(device)
            target_lengths = torch.tensor([2]).to(device)

            torch_loss = torch.nn.functional.ctc_loss(
                log_probs=torch_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='mean')

            act = k2_activation.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
            k2_log_probs = torch.nn.functional.log_softmax(act, dim=-1)

            supervision_segments = torch.tensor([[0, 0, 3]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = k2.ctc_topo(4)
            linear_fsa = k2.linear_fsa([2, 3])
            decoding_graph = k2.compose(ctc_topo, linear_fsa).to(device)

            k2_loss = k2.ctc_loss(decoding_graph,
                                  dense_fsa_vec,
                                  reduction='mean',
                                  target_lengths=target_lengths)

            expected_loss = torch.tensor([4.938850402832],
                                         device=device) / target_lengths

            assert torch.allclose(torch_loss, k2_loss)
            assert torch.allclose(torch_loss, expected_loss)

            torch_loss.backward()
            k2_loss.backward()
            assert torch.allclose(torch_activation.grad, k2_activation.grad)

    def test_case4(self):
        for device in self.devices:
            # put case3, case2 and case1 into a batch
            torch_activation_1 = torch.tensor(
                [[0., 0., 0., 0., 0.]]).to(device).requires_grad_(True)

            torch_activation_2 = torch.arange(1, 16).reshape(3, 5).to(
                torch.float32).to(device).requires_grad_(True)

            torch_activation_3 = torch.tensor([
                [-5, -4, -3, -2, -1],
                [-10, -9, -8, -7, -6],
                [-15, -14, -13, -12, -11.],
            ]).to(device).requires_grad_(True)

            k2_activation_1 = torch_activation_1.detach().clone(
            ).requires_grad_(True)
            k2_activation_2 = torch_activation_2.detach().clone(
            ).requires_grad_(True)
            k2_activation_3 = torch_activation_3.detach().clone(
            ).requires_grad_(True)

            # [T, N, C]
            torch_activations = torch.nn.utils.rnn.pad_sequence(
                [torch_activation_3, torch_activation_2, torch_activation_1],
                batch_first=False,
                padding_value=0)

            # [N, T, C]
            k2_activations = torch.nn.utils.rnn.pad_sequence(
                [k2_activation_3, k2_activation_2, k2_activation_1],
                batch_first=True,
                padding_value=0)

            # [[b,c], [c,c], [a]]
            targets = torch.tensor([2, 3, 3, 3, 1]).to(device)
            input_lengths = torch.tensor([3, 3, 1]).to(device)
            target_lengths = torch.tensor([2, 2, 1]).to(device)

            torch_log_probs = torch.nn.functional.log_softmax(
                torch_activations, dim=-1)  # (T, N, C)

            torch_loss = torch.nn.functional.ctc_loss(
                log_probs=torch_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='sum')

            expected_loss = torch.tensor(
                [4.938850402832, 7.355742931366, 1.6094379425049]).sum()

            assert torch.allclose(torch_loss, expected_loss.to(device))

            k2_log_probs = torch.nn.functional.log_softmax(k2_activations,
                                                           dim=-1)
            supervision_segments = torch.tensor(
                [[0, 0, 3], [1, 0, 3], [2, 0, 1]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = k2.ctc_topo(4)
            # [ [b, c], [c, c], [a]]
            linear_fsa = k2.linear_fsa([[2, 3], [3, 3], [1]])
            decoding_graph = k2.compose(ctc_topo, linear_fsa).to(device)

            k2_loss = k2.ctc_loss(decoding_graph,
                                  dense_fsa_vec,
                                  reduction='sum',
                                  target_lengths=target_lengths)

            assert torch.allclose(torch_loss, k2_loss)

            scale = torch.tensor([1., -2, 3.5]).to(device)
            (torch_loss * scale).sum().backward()
            (k2_loss * scale).sum().backward()
            assert torch.allclose(torch_activation_1.grad,
                                  k2_activation_1.grad)
            assert torch.allclose(torch_activation_2.grad,
                                  k2_activation_2.grad)
            assert torch.allclose(torch_activation_3.grad,
                                  k2_activation_3.grad)

    def test_random_case1(self):
        # 1 sequence
        for device in self.devices:
            T = torch.randint(10, 100, (1,)).item()
            C = torch.randint(20, 30, (1,)).item()
            torch_activation = torch.rand((1, T + 10, C),
                                          dtype=torch.float32,
                                          device=device).requires_grad_(True)

            k2_activation = torch_activation.detach().clone().requires_grad_(
                True)

            # [N, T, C] -> [T, N, C]
            torch_log_probs = torch.nn.functional.log_softmax(
                torch_activation.permute(1, 0, 2), dim=-1)

            input_lengths = torch.tensor([T]).to(device)
            target_lengths = torch.randint(1, T, (1,)).to(device)
            targets = torch.randint(1, C - 1,
                                    (target_lengths.item(),)).to(device)

            torch_loss = torch.nn.functional.ctc_loss(
                log_probs=torch_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='mean')
            k2_log_probs = torch.nn.functional.log_softmax(k2_activation,
                                                           dim=-1)
            supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)
            ctc_topo = k2.ctc_topo(C - 1)
            linear_fsa = k2.linear_fsa([targets.tolist()])
            decoding_graph = k2.compose(ctc_topo, linear_fsa).to(device)

            k2_loss = k2.ctc_loss(decoding_graph,
                                  dense_fsa_vec,
                                  reduction='mean',
                                  target_lengths=target_lengths)

            assert torch.allclose(torch_loss, k2_loss)
            scale = torch.rand_like(torch_loss) * 100
            (torch_loss * scale).sum().backward()
            (k2_loss * scale).sum().backward()
            assert torch.allclose(torch_activation.grad,
                                  k2_activation.grad,
                                  atol=1e-2)

    def test_random_case2(self):
        # 2 sequences
        for device in self.devices:
            T1 = torch.randint(10, 200, (1,)).item()
            T2 = torch.randint(9, 100, (1,)).item()
            C = torch.randint(20, 30, (1,)).item()
            if T1 < T2:
                T1, T2 = T2, T1

            torch_activation_1 = torch.rand((T1, C),
                                            dtype=torch.float32,
                                            device=device).requires_grad_(True)
            torch_activation_2 = torch.rand((T2, C),
                                            dtype=torch.float32,
                                            device=device).requires_grad_(True)

            k2_activation_1 = torch_activation_1.detach().clone(
            ).requires_grad_(True)
            k2_activation_2 = torch_activation_2.detach().clone(
            ).requires_grad_(True)

            # [T, N, C]
            torch_activations = torch.nn.utils.rnn.pad_sequence(
                [torch_activation_1, torch_activation_2],
                batch_first=False,
                padding_value=0)

            # [N, T, C]
            k2_activations = torch.nn.utils.rnn.pad_sequence(
                [k2_activation_1, k2_activation_2],
                batch_first=True,
                padding_value=0)

            target_length1 = torch.randint(1, T1, (1,)).item()
            target_length2 = torch.randint(1, T2, (1,)).item()

            target_lengths = torch.tensor([target_length1,
                                           target_length2]).to(device)
            targets = torch.randint(1, C - 1,
                                    (target_lengths.sum(),)).to(device)

            # [T, N, C]
            torch_log_probs = torch.nn.functional.log_softmax(
                torch_activations, dim=-1)
            input_lengths = torch.tensor([T1, T2]).to(device)

            torch_loss = torch.nn.functional.ctc_loss(
                log_probs=torch_log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction='mean')

            assert T1 >= T2
            supervision_segments = torch.tensor([[0, 0, T1], [1, 0, T2]],
                                                dtype=torch.int32)
            k2_log_probs = torch.nn.functional.log_softmax(k2_activations,
                                                           dim=-1)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)
            ctc_topo = k2.ctc_topo(C - 1)
            linear_fsa = k2.linear_fsa([
                targets[:target_length1].tolist(),
                targets[target_length1:].tolist()
            ])
            decoding_graph = k2.compose(ctc_topo, linear_fsa).to(device)

            k2_loss = k2.ctc_loss(decoding_graph,
                                  dense_fsa_vec,
                                  reduction='mean',
                                  target_lengths=target_lengths)

            assert torch.allclose(torch_loss, k2_loss)
            scale = torch.rand_like(torch_loss) * 100
            (torch_loss * scale).sum().backward()
            (k2_loss * scale).sum().backward()
            assert torch.allclose(torch_activation_1.grad,
                                  k2_activation_1.grad,
                                  atol=1e-2)
            assert torch.allclose(torch_activation_2.grad,
                                  k2_activation_2.grad,
                                  atol=1e-2)


if __name__ == '__main__':
    torch.manual_seed(20210109)
    unittest.main()
