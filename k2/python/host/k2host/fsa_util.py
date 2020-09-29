# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)

# See ../../../LICENSE for clarification regarding multiple authors

import re
import struct
from collections import defaultdict

import torch

from .fsa import Fsa


def float_to_int(f):
    f = struct.pack('f', f)
    return int.from_bytes(f, 'little')


def str_to_fsa(s: str) -> Fsa:
    '''Create an FSA from a string.

    The input string `s` consists of several lines; every line except the
    last line has the following format:

        <src_state> <dest_state> <label> <weight>

    The last line of `s` contains:

        <final_state>

    Args:
        s (str): a string representation of the fsa
    Returns:
        k2.Fsa
    '''
    rule_pattern = re.compile(
        r'^[ \t]*(\d+)[ \t]+(\d+)[ \t]+([-]?\d+)[ \t]+([-]?\d*[.]?\d+)[ \t]*$')
    final_state_pattern = re.compile(r'^[ \t]*(\d+)[ \t]*$')
    rules = s.strip().split('\n')

    final_state = None
    state_to_rules = defaultdict(list)
    for i, r in enumerate(rules):
        m = rule_pattern.match(r)
        if m:
            src_state = int(m.group(1))
            dest_state = int(m.group(2))
            label = int(m.group(3))
            weight = float(m.group(4))
            weight = float_to_int(weight)
            state_to_rules[src_state].append(
                [src_state, dest_state, label, weight])
        else:
            m = final_state_pattern.match(r)
            assert m
            final_state = int(m.group(1))
            assert i == len(rules) - 1
    assert final_state is not None
    arcs = list()
    indexes = list()
    num = 0
    for i in range(final_state + 1):
        indexes.append(num)
        if i in state_to_rules:
            t = state_to_rules[i]
            arcs.extend(t)
            num += len(t)

    # the final state is repeated
    indexes.append(num)

    data = torch.tensor(arcs, dtype=torch.int32)
    indexes = torch.tensor(indexes, dtype=torch.int32)

    fsa = Fsa(indexes, data)
    return fsa
