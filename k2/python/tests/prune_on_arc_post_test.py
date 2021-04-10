#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

# To run this single test, use
#
#  ctest --verbose -R prune_on_arc_post_test_py

import unittest

import k2
import torch


class TestPruneOnArcPost(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            cls.devices.append(torch.device('cuda', 0))


if __name__ == '__main__':
    unittest.main()
