#!/usr/bin/env python3

import k2
import torch

shape = k2.ragged.create_ragged_shape2(
    row_splits=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
)
print(type(shape))
print(shape)
"""
<class '_k2.ragged.RaggedShape'>
[ [ x x ] [ x ] [ ] ]
"""
print("num_states:", shape.dim0)
print("num_arcs:", shape.numel())
