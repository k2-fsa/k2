#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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
#  ctest --verbose -R ragged_tensor_test_py

import os
import tempfile
import unittest

import k2
import k2.ragged as k2r

import torch


class TestRaggedTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))
        cls.dtypes = [torch.float32, torch.float64, torch.int32]

    def test_create_ragged_tensor(self):
        funcs = [k2r.create_ragged_tensor, k2r.RaggedTensor]
        for func in funcs:
            for device in self.devices:
                a = func([[1000, 2], [3]], device=device)
                assert isinstance(a, k2r.RaggedTensor)
                assert a.dtype == torch.int32
                assert a.device == device

                a = func([[1000, 2], [3]], dtype=torch.float32, device=device)
                assert a.dtype == torch.float32
                assert a.device == device

                a = func([[1000, 2], [3]], dtype=torch.float64, device=device)
                assert a.dtype == torch.float64
                assert a.device == device
                for dtype in self.dtypes:
                    a = func([[1000, 2], [3]], dtype=dtype, device=device)
                    assert a.dtype == dtype
                    assert a.device == device

    def test_create_ragged_tensor_from_string(self):
        funcs = [k2r.create_ragged_tensor, k2r.RaggedTensor]
        for func in funcs:
            for device in self.devices:
                for dtype in self.dtypes:
                    a = func(
                        [[1], [2, 3, 4, 5], []], dtype=dtype, device=device
                    )
                    b = func("[[1] [2 3 4 5] []]", dtype=dtype, device=device)
                    assert a == b
                    assert b.dim0 == 3
                    assert a.dtype == dtype
                    assert a.device == device

                    b = k2r.RaggedTensor(
                        "[[[1] [2 3] []] [[10]]]", dtype=dtype, device=device
                    )
                    assert b.num_axes == 3
                    assert b.dim0 == 2
                    assert b.dtype == dtype
                    assert b.device == device

    def test_create_ragged_tensor_from_torch_tensor(self):
        for device in self.devices:
            for func in [k2r.create_ragged_tensor, k2r.RaggedTensor]:
                for dtype in self.dtypes:
                    a = torch.arange(24, dtype=dtype, device=device).reshape(
                        2, 3, 4
                    )
                    b = func(a)
                    assert b.shape.tot_sizes() == (2, 2 * 3, 2 * 3 * 4)

                    # a is contiguous, so memory is shared
                    c = a.reshape(-1)
                    c[0] = 10
                    assert b.values[0] == 10
                    b.values[1] = 100
                    assert c[1] == 100

                    assert b.dtype == dtype
                    assert b.device == device

                    assert torch.all(torch.eq(c, b.values))

        for device in self.devices:
            for func in [k2r.create_ragged_tensor, k2r.RaggedTensor]:
                for dtype in self.dtypes:
                    a = torch.arange(100, dtype=dtype, device=device).reshape(
                        10, 10
                    )[:, ::2]
                    assert a.shape == (10, 5)
                    b = func(a)
                    assert b.dtype == dtype
                    assert b.device == device

                    assert b.shape.tot_sizes() == (10, 10 * 5)

                    c = a.reshape(-1)
                    assert torch.all(torch.eq(c, b.values))

                    # a is not contiguous, so memory is copied
                    c[0] = -10
                    assert b.values[0] != -10
                    b.values[1] = -100
                    assert c[1] != -100

    def test_property_values(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor(
                    [[1], [2], [], [3, 4]], device=device, dtype=dtype
                )
                assert torch.all(
                    torch.eq(
                        a.values,
                        torch.tensor([1, 2, 3, 4], dtype=dtype, device=device),
                    )
                )

                with self.assertRaises(AttributeError):
                    # the `values` attribute is const. You cannot rebind it
                    a.values = 10

                # However, we can change the elements of a.values
                a.values[0] = 10
                a.values[-1] *= 2

                expected = k2r.RaggedTensor(
                    [[10], [2], [], [3, 8]], dtype=dtype, device=device
                )
                assert a == expected

                a.values[0] = 1
                assert a != expected

    def test_clone(self):
        a = k2r.RaggedTensor([[1, 2], [], [3]])
        b = a.clone()

        assert a == b
        a.values[0] = 10

        assert a != b

    def test_cpu_to_cpu(self):
        pass

    def test_cpu_to_cuda(self):
        pass

    def test_cuda_to_cpu(self):
        pass

    def test_cuda_to_cuda(self):
        pass

    def test_dtypes_conversion(self):
        pass

    def test_grad(self):
        a = k2r.RaggedTensor([[1, 2], [10], []], dtype=torch.float32)
        assert a.grad is None
        assert a.requires_grad is False

        a.requires_grad = True
        assert a.requires_grad is True

        a.requires_grad_(False)
        assert a.requires_grad is False

        a.requires_grad_(True)
        assert a.requires_grad is True

    def test_sum_with_grad(self):
        for device in self.devices:
            for dtype in [torch.float32, torch.float64]:
                a = k2r.RaggedTensor([[1, 2], [], [5]], dtype=dtype)
                a = a.to(device)
                a.requires_grad_(True)
                b = a.sum()
                expected_sum = torch.tensor(
                    [3, 0, 5], dtype=dtype, device=device
                )

                assert torch.all(torch.eq(b, expected_sum))

                c = b[0] * 10 + b[1] * 20 + b[2] * 30
                c.backward()
                expected_grad = torch.tensor(
                    [10, 10, 30], device=device, dtype=dtype
                )
                assert torch.all(torch.eq(a.grad, expected_grad))

    def test_sum_no_grad(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor([[1, 2], [], [5]], dtype=dtype)
                a = a.to(device)
                b = a.sum()
                expected_sum = torch.tensor(
                    [3, 0, 5], dtype=dtype, device=device
                )

                assert torch.all(torch.eq(b, expected_sum))

    def test_getitem(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor(
                    "[ [[1 2] [] [10]] [[3] [5]] ]", dtype=dtype
                )
                a = a.to(device)
                b = a[0]
                expected = k2r.RaggedTensor("[[1 2] [] [10]]", dtype=dtype).to(
                    device
                )
                assert b == expected

                b = a[1]
                expected = k2r.RaggedTensor("[[3] [5]]", dtype=dtype).to(device)
                assert b == expected

    def test_getstate_2axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor([[1, 2], [3], []], dtype=dtype).to(device)
                b = a.__getstate__()
                assert isinstance(b, tuple)
                assert len(b) == 3
                # b contains (row_splits, "row_ids1", values)
                b_0 = torch.tensor(
                    [0, 2, 3, 3], dtype=torch.int32, device=device
                )
                b_1 = "row_ids1"
                b_2 = a.values

                assert torch.all(torch.eq(b[0], b_0))
                assert b[1] == b_1
                assert torch.all(torch.eq(b[2], b_2))

    def test_getstate_3axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor(
                    "[[[1 2] [3] []] [[4] [5 6]]]", dtype=dtype
                ).to(device)
                b = a.__getstate__()
                assert isinstance(b, tuple)
                assert len(b) == 5
                # b contains (row_splits1, "row_ids1", row_splits2,
                # "row_ids2", values)
                b_0 = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
                b_1 = "row_ids1"
                b_2 = torch.tensor(
                    [0, 2, 3, 3, 4, 6], dtype=torch.int32, device=device
                )  # noqa
                b_3 = "row_ids2"
                b_4 = a.values

                assert torch.all(torch.eq(b[0], b_0))
                assert b[1] == b_1
                assert torch.all(torch.eq(b[2], b_2))
                assert b[3] == b_3
                assert torch.all(torch.eq(b[4], b_4))

    def test_setstate_2axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor([[1], [2, 3], []], dtype=dtype)
                a = a.to(device)
                fid, tmp_filename = tempfile.mkstemp()
                os.close(fid)

                torch.save(a, tmp_filename)

                b = torch.load(tmp_filename)
                os.remove(tmp_filename)

                # It checks both dtype and device, not just value
                assert a == b

    def test_setstate_3axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor("[ [[1] [2 3] []]  [[10]]]", dtype=dtype)
                fid, tmp_filename = tempfile.mkstemp()
                os.close(fid)

                torch.save(a, tmp_filename)

                b = torch.load(tmp_filename)
                os.remove(tmp_filename)

                assert a == b

    def test_tot_size_2axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor("[ [1 2 3] [] [5 8] ]", dtype=dtype)
                a = a.to(device)

                assert a.tot_size(0) == 3
                assert a.tot_size(1) == 5

    def test_tot_size_3axes(self):
        for device in self.devices:
            for dtype in self.dtypes:
                a = k2r.RaggedTensor(
                    "[ [[1 2 3] [] [5 8]] [[] [1 5 9 10 -1] [] [] []] ]",
                    dtype=dtype,
                )
                a = a.to(device)

                assert a.tot_size(0) == 2
                assert a.tot_size(1) == 8
                assert a.tot_size(2) == 10


if __name__ == "__main__":
    unittest.main()
