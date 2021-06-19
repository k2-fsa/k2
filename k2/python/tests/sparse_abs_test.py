#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang)
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
#  ctest --verbose -R sparse_abs_test_py

import unittest

import k2.sparse
import torch


class TestSparseAbs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_no_repeats(self):
        for device in self.devices:
            row_indexes = torch.tensor([0, 0, 1, 2, 2]).to(device)
            col_indexes = torch.tensor([0, 1, 0, 1, 0]).to(device)
            indexes = torch.stack([row_indexes, col_indexes])
            size = (3, 3)
            values = torch.tensor([1, -2, -3, 4, 0],
                                  dtype=torch.float32,
                                  requires_grad=True,
                                  device=device)

            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            ans = k2.sparse.abs(sparse_tensor)

            assert ans.is_sparse
            assert torch.all(
                torch.eq(ans._values(),
                         sparse_tensor._values().abs()))

            s = torch.sparse.sum(ans)
            assert s.item() == 10
            scale = 2
            (scale * s).backward()
            grad1 = values.grad.clone()

            values.grad = None

            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = sparse_tensor.to_dense().abs().sum()
            (scale * s).backward()

            assert torch.allclose(grad1, values.grad)

    def test_with_repeats(self):
        for device in self.devices:
            row_indexes = torch.tensor([0, 0, 1, 2, 0, 2, 2]).to(device)
            col_indexes = torch.tensor([0, 1, 0, 1, 0, 1, 0]).to(device)
            indexes = torch.stack([row_indexes, col_indexes])
            size = (3, 3)
            # (0, 0): 1 + (-2) = -1
            # (2, 1): 4 + (-6) == -2
            values = torch.tensor([1, 2, -3, 4, -2, -6, 0],
                                  dtype=torch.float32,
                                  requires_grad=True,
                                  device=device)
            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            ans = k2.sparse.abs(sparse_tensor)
            assert ans.is_sparse
            assert torch.all(
                torch.eq(ans._values(),
                         sparse_tensor._values().abs()))

            s = torch.sparse.sum(ans)
            assert s.item() == 8
            scale = -3
            (scale * s).backward()
            grad1 = values.grad.clone()

            values.grad = None
            sparse_tensor = torch.sparse_coo_tensor(indexes,
                                                    values=values,
                                                    size=size).coalesce()
            s = sparse_tensor.to_dense().abs().sum()
            (scale * s).backward()

            assert torch.allclose(grad1, values.grad)

            # minus, abs, sum
            major, minor = torch.__version__.split('.')[:2]
            major = int(major)
            minor = int(minor)
            if major < 1 or (major == 1 and minor < 7):
                print(f'Current PyTorch version is: {torch.__version__}')
                print('Skip it for version less than 1.7.0')
            else:
                values.grad = None
                scale = 10
                sparse_tensor1 = torch.sparse_coo_tensor(indexes,
                                                         values=values,
                                                         size=size).coalesce()
                sparse_tensor2 = torch.sparse_coo_tensor(indexes,
                                                         values=values * scale,
                                                         size=size).coalesce()
                # this works only for torch >= 1.7.0
                sparse_tensor = sparse_tensor2 + (-sparse_tensor1)
                s1 = torch.sparse.sum(k2.sparse.abs(sparse_tensor.coalesce()))
                s1.backward()

                grad1 = values.grad.clone()
                values.grad = None

                sparse_tensor1 = torch.sparse_coo_tensor(indexes,
                                                         values=values,
                                                         size=size).coalesce()
                sparse_tensor2 = torch.sparse_coo_tensor(indexes,
                                                         values=values * scale,
                                                         size=size).coalesce()
                s2 = (sparse_tensor1 +
                      (-sparse_tensor2)).to_dense().abs().sum()
                s2.backward()

                assert s1.item() == s2.item()
                assert torch.allclose(grad1, values.grad)


if __name__ == '__main__':
    unittest.main()
