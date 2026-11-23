#!/usr/bin/env python3

import k2
import torch

a = k2.RaggedTensor([[1, 2], [1]])
b = k2.RaggedTensor([[1, 2], [1]], dtype=torch.int32)
c = k2.RaggedTensor([[1, 2], [1.5]])
d = k2.RaggedTensor([[1, 2], [1.5]], dtype=torch.float32)
e = k2.RaggedTensor([[1, 2], [1.5]], dtype=torch.float64)
f = k2.RaggedTensor([[1, 2], [1]], dtype=torch.float32, device=torch.device("cuda", 0))
g = k2.RaggedTensor([[1, 2], [1]], device="cuda:0", dtype=torch.float64)
print(f"a:\n{a}")
print(f"b:\n{b}")
print(f"c:\n{c}")
print(f"d:\n{d}")
print(f"e:\n{e}")
print(f"f:\n{f}")
print(f"g:\n{g}")
print(f"g.to_str_simple():\n{g.to_str_simple()}")
print(f"a.dtype: {a.dtype}, g.device: {g.device}")
print(f"a.to(g.device).device: {a.to(g.device).device}")
print(f"a.to(g.dtype).dtype: {a.to(g.dtype).dtype}")
"""
a:
RaggedTensor([[1, 2],
              [1]], dtype=torch.int32)
b:
RaggedTensor([[1, 2],
              [1]], dtype=torch.int32)
c:
RaggedTensor([[1, 2],
              [1.5]], dtype=torch.float32)
d:
RaggedTensor([[1, 2],
              [1.5]], dtype=torch.float32)
e:
RaggedTensor([[1, 2],
              [1.5]], dtype=torch.float64)
f:
RaggedTensor([[1, 2],
              [1]], device='cuda:0', dtype=torch.float32)
g:
RaggedTensor([[1, 2],
              [1]], device='cuda:0', dtype=torch.float64)
g.to_str_simple():
RaggedTensor([[1, 2], [1]], device='cuda:0', dtype=torch.float64)
a.dtype: torch.int32, g.device: cuda:0
a.to(g.device).device: cuda:0
a.to(g.dtype).dtype: torch.float64
"""
