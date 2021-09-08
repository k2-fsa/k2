#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R  replace_fsa_test_py

import unittest

import k2
import torch
import _k2


# See comments here: https://github.com/k2-fsa/k2/pull/759#discussion_r662006539
def _construct_f(fsa_vec: k2.Fsa) -> k2.Fsa:
    num_fsa = fsa_vec.shape[0]
    union = k2.union(fsa_vec)
    union.aux_labels = torch.zeros(union.num_arcs)
    union.aux_labels[0:num_fsa] = torch.tensor(list(range(1, 1 + num_fsa)),
                                               dtype=torch.int32)
    union_str = k2.to_str_simple(union)
    states_num = union.shape[0]

    new_str_array = []
    new_str_array.append("0 {} -1 0 0".format(states_num - 1))
    for line in union_str.strip().split("\n"):
        tokens = line.strip().split(" ")
        if len(tokens) == 5:
            tokens[1] = '0' if int(tokens[1]) == states_num - 1 else tokens[1]
            tokens[2] = '0' if int(tokens[2]) == -1 else tokens[2]
        new_str_array.append(" ".join(tokens))
    new_str = "\n".join(new_str_array)

    new_fsa = k2.Fsa.from_str(new_str, num_aux_labels=1)
    new_fsa_invert = k2.invert(new_fsa)
    return new_fsa_invert


# gennerate random FsaVec that connect and contains no empty fsa
def _generate_fsa_vec(min_num_fsas: int = 20,
                      max_num_fsas: int = 21,
                      acyclic: bool = True,
                      max_symbol: int = 20,
                      min_num_arcs: int = 10,
                      max_num_arcs: int = 15) -> k2.Fsa:
    fsa = k2.random_fsa_vec(min_num_fsas, max_num_fsas, acyclic, min_num_arcs,
                            max_num_arcs)
    fsa = k2.connect(fsa)
    while True:
        success = True
        for i in range(fsa.shape[0]):
            if fsa[i].shape[0] == 0:
                success = False
                break
        if success:
            break
        else:
            fsa = k2.random_fsa_vec(min_num_fsas, max_num_fsas, acyclic,
                                    min_num_arcs, max_num_arcs)
            fsa = k2.connect(fsa)
    return fsa


class TestReplaceFsa(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test(self):
        for device in self.devices:
            s1 = '''
            0 1 11 11
            0 2 12 12
            0 3 13 13
            1 4 -1 0
            2 4 -1 0
            3 4 -1 0
            4
            '''
            fsa1 = k2.Fsa.from_str(s1)

            s2 = '''
            0 1 21 21
            0 2 22 22
            1 2 23 23
            1 3 -1 0
            2 3 -1 0
            3
            '''
            fsa2 = k2.Fsa.from_str(s2)

            s3 = '''
            0 1 31 31
            1 2 32 32
            1 3 33 33
            2 4 -1 0
            3 4 -1 0
            4
            '''
            fsa3 = k2.Fsa.from_str(s3)
            src = k2.create_fsa_vec([fsa1, fsa2, fsa3]).to(device)
            src.requires_grad_(True)

            s0 = '''
            0 1 1 1
            0 2 2 2
            1 3 4 4
            2 3 3 3
            2 4 -1 0
            3 4 -1 0
            4
            '''
            index = k2.Fsa.from_str(s0).to(device)
            index.requires_grad_(True)

            index.aux_label = torch.tensor([1, 2, 3, 4, 5, 6],
                                           dtype=torch.int32,
                                           device=device)

            dest = k2.replace_fsa(src, index, 1)

            actual_str = k2.to_str_simple(dest)
            expected_str = '\n'.join([
                '0 1 0 1', '0 5 0 2', '1 2 11 11', '1 3 12 12', '1 4 13 13',
                '2 8 0 0', '3 8 0 0', '4 8 0 0', '5 6 21 21', '5 7 22 22',
                '6 7 23 23', '6 9 0 0', '7 9 0 0', '8 14 4 4', '9 10 0 3',
                '9 15 -1 0', '10 11 31 31', '11 12 32 32', '11 13 33 33',
                '12 14 0 0', '13 14 0 0', '14 15 -1 0', '15'
            ])

            assert actual_str.strip() == expected_str

            assert torch.all(
                torch.eq(
                    dest.aux_label,
                    torch.tensor([
                        1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0,
                        0, 0, 0, 6
                    ]).to(device).to(torch.int32)))

            loss = dest.scores.sum()
            (-loss).backward()
            assert torch.allclose(
                index.grad,
                torch.tensor([-1, -1, -1, -1, -1, -1]).to(index.grad))

            assert torch.allclose(
                src.grad,
                torch.tensor([
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1
                ]).to(src.grad))

    # see the discussion here:
    # https://github.com/k2-fsa/k2/pull/759#discussion_r655052289
    def test_composition_equivalence(self):
        index = _generate_fsa_vec()
        index = k2.arc_sort(k2.connect(k2.remove_epsilon(index)))

        src = _generate_fsa_vec()

        replace = k2.replace_fsa(src, index, 1)
        replace = k2.top_sort(replace)

        f_fsa = _construct_f(src)
        f_fsa = k2.arc_sort(f_fsa)
        intersect = k2.intersect(index, f_fsa, treat_epsilons_specially=True)
        intersect = k2.invert(intersect)
        intersect = k2.top_sort(intersect)
        delattr(intersect, 'aux_labels')

        assert k2.is_rand_equivalent(replace,
                                     intersect,
                                     log_semiring=True,
                                     delta=1e-3)


if __name__ == '__main__':
    unittest.main()
