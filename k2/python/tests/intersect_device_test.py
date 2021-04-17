#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang
#                                                   Daniel Povey)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R intersect_device_test_py

import unittest

import k2
import torch


class TestIntersectDevice(unittest.TestCase):

    def test(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        for device in devices:
            for use_identity_map, sorted_match_a in [(True, True),
                                                     (False, True),
                                                     (True, False),
                                                     (False, False)]:
                # recognizes (0|1)(0|2)
                s1 = '''
                    0 1 0 0.1
                    0 1 1 0.2
                    1 2 0 0.4
                    1 2 2 0.3
                    2 3 -1 0.5
                    3
                '''

                # recognizes 02*
                s2 = '''
                    0 1 0 1
                    1 1 2 2
                    1 2 -1 3
                    2
                '''

                # recognizes 1*0
                s3 = '''
                    0 0 1 10
                    0 1 0 20
                    1 2 -1 30
                    2
                '''
                a_fsa = k2.Fsa.from_str(s1).to(device)
                b_fsa_1 = k2.Fsa.from_str(s2).to(device)
                b_fsa_2 = k2.Fsa.from_str(s3).to(device)

                a_fsa.requires_grad_(True)
                b_fsa_1.requires_grad_(True)
                b_fsa_2.requires_grad_(True)

                b_fsas = k2.create_fsa_vec([b_fsa_1, b_fsa_2])
                if use_identity_map:
                    a_fsas = k2.create_fsa_vec([a_fsa, a_fsa])
                    b_to_a_map = torch.tensor([0, 1],
                                              dtype=torch.int32).to(device)
                else:
                    a_fsas = k2.create_fsa_vec([a_fsa])
                    b_to_a_map = torch.tensor([0, 0],
                                              dtype=torch.int32).to(device)

                c_fsas = k2.intersect_device(a_fsas, b_fsas, b_to_a_map,
                                             sorted_match_a)
                assert c_fsas.shape == (2, None, None)
                c_fsas = k2.connect(c_fsas.to('cpu'))
                # c_fsas[0] recognizes: 02
                # c_fsas[1] recognizes: 10

                actual_str_0 = k2.to_str(c_fsas[0])
                expected_str_0 = '\n'.join(
                    ['0 1 0 1.1', '1 2 2 2.3', '2 3 -1 3.5', '3'])
                assert actual_str_0.strip() == expected_str_0

                actual_str_1 = k2.to_str(c_fsas[1])
                expected_str_1 = '\n'.join(
                    ['0 1 1 10.2', '1 2 0 20.4', '2 3 -1 30.5', '3'])
                assert actual_str_1.strip() == expected_str_1

                loss = c_fsas.scores.sum()
                (-loss).backward()
                assert torch.allclose(
                    a_fsa.grad,
                    torch.tensor([-1, -1, -1, -1, -2]).to(a_fsa.grad))
                assert torch.allclose(
                    b_fsa_1.grad,
                    torch.tensor([-1, -1, -1]).to(b_fsa_1.grad))
                assert torch.allclose(
                    b_fsa_2.grad,
                    torch.tensor([-1, -1, -1]).to(b_fsa_2.grad))


if __name__ == '__main__':
    unittest.main()
