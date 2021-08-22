#!/usr/bin/env python3
#
# Copyright      2020   Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R  ragged_test_py

import os
import unittest

import k2
import torch


class TestRagged(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            torch.cuda.set_device(0)
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_ragged_int_from_str(self):
        s = '''
        [ [1 2] [3] ]
        '''
        for device in self.devices:
            ragged_int = k2.RaggedInt(s).to(device)
            print(ragged_int)
            assert torch.all(
                torch.eq(ragged_int.values(),
                         torch.tensor([1, 2, 3], device=device)))
            assert ragged_int.dim0() == 2
            assert torch.all(
                torch.eq(ragged_int.row_splits(1),
                         torch.tensor([0, 2, 3], device=device)))

            self.assertEqual([2, 3], ragged_int.tot_sizes())

    def test_pickle_ragged(self):
        for device in self.devices:
            # test num_axes == 2
            raggeds = ("[ ]", "[ [ ] ]", "[ [1 2] [3] ]")
            for s in raggeds:
                for cls in [k2.RaggedInt, k2.RaggedFloat]:
                    ragged = cls(s).to(device)
                    torch.save(ragged, "ragged.pt")
                    ragged_reload = torch.load("ragged.pt")
                    self.assertEqual(ragged, ragged_reload)
                    os.remove("ragged.pt")

            # test num_axes == 3
            raggeds = ("[ [ [ ] ] ]", "[ [ [1 2] [3] ] [ [4 5] [6] ] ]")
            for s in raggeds:
                for cls in [k2.RaggedInt, k2.RaggedFloat]:
                    ragged = cls(s).to(device)
                    torch.save(ragged, "ragged.pt")
                    ragged_reload = torch.load("ragged.pt")
                    self.assertEqual(ragged, ragged_reload)
                    os.remove("ragged.pt")

    def test_to_same_device(self):
        for Type in [k2.RaggedInt, k2.RaggedFloat]:
            for device in self.devices:
                src = Type('[[1 2] [3]]').to(device)
                dst = src.to(device)

                assert src.device() == dst.device() == device
                assert src == dst

                src.values()[0] = 10

                # dst shares the underlying memory with src
                # since src was already on the given device
                assert src == dst

    def test_cpu_to_cuda(self):
        if not (torch.cuda.is_available() and k2.with_cuda):
            return

        cpu = torch.device('cpu')
        cuda_devices = [torch.device('cuda', 0)]
        torch.cuda.set_device(0)
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(1)
            cuda_devices.append(torch.device('cuda', 1))

        for Type in [k2.RaggedInt, k2.RaggedFloat]:
            src = Type('[[1 2] [3]]')
            for device in cuda_devices:
                dst = src.to(device)
                assert dst.device() == device
                assert str(src) == str(dst)

                src.values()[0] = 10
                assert str(src) != str(dst)

    def test_cuda_to_cpu(self):
        pass

    def test_cuda_to_cuda(self):
        pass

    def test_to_same_type(self):
        pass

    def test_int_to_float(self):
        pass

    def test_float_to_int(self):
        pass


if __name__ == '__main__':
    unittest.main()
