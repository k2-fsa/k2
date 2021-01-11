#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R ctc_gradients_test_py

from typing import List

import unittest

import k2
import torch


def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    '''Build CTC topology.

    The resulting topology converts repeated input
    symbols to a single output symbol.

    Caution:
      The resulting topo is an FST. Epsilons are on the left
      side (i.e., ilabels) and tokens are on the right side (i.e., olabels)

    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    '''
    assert 0 in tokens, 'We assume 0 is ID of the blank symbol'

    num_states = len(tokens)
    final_state = num_states
    rules = ''
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                rules += f'{i} {i} 0 {tokens[i]} 0.0\n'
            else:
                rules += f'{i} {j} {tokens[j]} {tokens[j]} 0.0\n'
        rules += f'{i} {final_state} -1 -1 0.0\n'
    rules += f'{final_state}'
    ans = k2.Fsa.from_str(rules)
    ans = k2.arc_sort(ans)
    return ans


# Test cases are modified from
# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
#
#
# The CTC losses computed by warp-ctc, PyTorch, and k2 are identical.
#
# The gradients with respect to network outputs are also identical
# for PyTorch and k2.
class TestCtcLossGradients(unittest.TestCase):

    def test_case1(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
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
                reduction='none')

            assert torch.allclose(torch_loss,
                                  torch.tensor([1.6094379425049]).to(device))

            # (N, T, C)
            k2_log_probs = torch.nn.functional.log_softmax(k2_activation,
                                                           dim=-1)

            supervision_segments = torch.tensor([[0, 0, 1]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = build_ctc_topo([0, 1, 2, 3, 4])
            linear_fsa = k2.linear_fsa([1])
            decoding_graph = k2.intersect(ctc_topo, linear_fsa)
            decoding_graph = k2.connect(decoding_graph).invert_().to(device)

            target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                              100.0)

            k2_scores = k2.get_tot_scores(target_graph,
                                          log_semiring=True,
                                          use_double_scores=False)
            assert torch.allclose(torch_loss, -1 * k2_scores)

            torch_loss.backward()
            (-k2_scores).backward()
            assert torch.allclose(torch_activation.grad, k2_activation.grad)

    def test_case2(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
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
                reduction='none')

            act = k2_activation.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
            k2_log_probs = torch.nn.functional.log_softmax(act, dim=-1)

            supervision_segments = torch.tensor([[0, 0, 3]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = build_ctc_topo([0, 1, 2, 3, 4])
            linear_fsa = k2.linear_fsa([3, 3])
            decoding_graph = k2.intersect(ctc_topo, linear_fsa)
            decoding_graph = k2.connect(decoding_graph).invert_().to(device)

            target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                              100.0)

            k2_scores = k2.get_tot_scores(target_graph,
                                          log_semiring=True,
                                          use_double_scores=False)
            assert torch.allclose(torch_loss, -1 * k2_scores)
            assert torch.allclose(torch_loss,
                                  torch.tensor([7.355742931366]).to(device))

            torch_loss.backward()
            (-k2_scores).backward()
            assert torch.allclose(torch_activation.grad, k2_activation.grad)

    def test_case3(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
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
                reduction='none')

            act = k2_activation.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
            k2_log_probs = torch.nn.functional.log_softmax(act, dim=-1)

            supervision_segments = torch.tensor([[0, 0, 3]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = build_ctc_topo([0, 1, 2, 3, 4])
            linear_fsa = k2.linear_fsa([2, 3])
            decoding_graph = k2.intersect(ctc_topo, linear_fsa)
            decoding_graph = k2.connect(decoding_graph).invert_().to(device)

            target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                              100.0)

            k2_scores = k2.get_tot_scores(target_graph,
                                          log_semiring=True,
                                          use_double_scores=False)
            assert torch.allclose(torch_loss, -1 * k2_scores)
            assert torch.allclose(torch_loss,
                                  torch.tensor([4.938850402832]).to(device))

            torch_loss.backward()
            (-k2_scores).backward()
            assert torch.allclose(torch_activation.grad, k2_activation.grad)

    def test_case4(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
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
                reduction='none')

            assert torch.allclose(
                torch_loss,
                torch.tensor([4.938850402832, 7.355742931366,
                              1.6094379425049]).to(device))

            k2_log_probs = torch.nn.functional.log_softmax(k2_activations,
                                                           dim=-1)
            supervision_segments = torch.tensor(
                [[0, 0, 3], [1, 0, 3], [2, 0, 1]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)

            ctc_topo = build_ctc_topo([0, 1, 2, 3, 4])
            # [ [b, c], [c, c], [a]]
            linear_fsa = k2.linear_fsa([[2, 3], [3, 3], [1]])
            decoding_graph = k2.intersect(ctc_topo, linear_fsa)
            decoding_graph = k2.connect(decoding_graph).invert_().to(device)

            target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                              100.0)

            k2_scores = k2.get_tot_scores(target_graph,
                                          log_semiring=True,
                                          use_double_scores=False)
            assert torch.allclose(torch_loss, -1 * k2_scores)

            scale = torch.tensor([1., -2, 3.5]).to(device)
            (torch_loss * scale).sum().backward()
            (-k2_scores * scale).sum().backward()
            assert torch.allclose(torch_activation_1.grad,
                                  k2_activation_1.grad)
            assert torch.allclose(torch_activation_2.grad,
                                  k2_activation_2.grad)
            assert torch.allclose(torch_activation_3.grad,
                                  k2_activation_3.grad)

    def test_random_case1(self):
        # 1 sequence
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
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
                reduction='none')
            k2_log_probs = torch.nn.functional.log_softmax(k2_activation,
                                                           dim=-1)
            supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)
            ctc_topo = build_ctc_topo(list(range(C)))
            linear_fsa = k2.linear_fsa([targets.tolist()])

            decoding_graph = k2.intersect(ctc_topo, linear_fsa)
            decoding_graph = k2.connect(decoding_graph).invert_().to(device)

            target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                              100.0)

            k2_scores = k2.get_tot_scores(target_graph,
                                          log_semiring=True,
                                          use_double_scores=False)
            assert torch.allclose(torch_loss, -1 * k2_scores)
            scale = torch.rand_like(torch_loss) * 100
            (torch_loss * scale).sum().backward()
            (-k2_scores * scale).sum().backward()
            assert torch.allclose(torch_activation.grad,
                                  k2_activation.grad,
                                  atol=1e-2)

    def test_random_case2(self):
        # 2 sequences
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda', 0))

        for device in devices:
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
                reduction='none')

            assert T1 >= T2
            supervision_segments = torch.tensor([[0, 0, T1], [1, 0, T2]],
                                                dtype=torch.int32)
            k2_log_probs = torch.nn.functional.log_softmax(k2_activations,
                                                           dim=-1)
            dense_fsa_vec = k2.DenseFsaVec(k2_log_probs,
                                           supervision_segments).to(device)
            ctc_topo = build_ctc_topo(list(range(C)))
            linear_fsa = k2.linear_fsa([
                targets[:target_length1].tolist(),
                targets[target_length1:].tolist()
            ])
            decoding_graph = k2.intersect(ctc_topo, linear_fsa)
            decoding_graph = k2.connect(decoding_graph).invert_().to(device)
            target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                              100.0)
            k2_scores = k2.get_tot_scores(target_graph,
                                          log_semiring=True,
                                          use_double_scores=False)
            assert torch.allclose(torch_loss, -1 * k2_scores)
            scale = torch.rand_like(torch_loss) * 100
            (torch_loss * scale).sum().backward()
            (-k2_scores * scale).sum().backward()
            assert torch.allclose(torch_activation_1.grad,
                                  k2_activation_1.grad,
                                  atol=1e-2)
            assert torch.allclose(torch_activation_2.grad,
                                  k2_activation_2.grad,
                                  atol=1e-2)


if __name__ == '__main__':
    torch.manual_seed(20210109)
    unittest.main()
