

#ifndef K2_PYTHON_CSRC_TORCH_DOC_ANY_H_
#define K2_PYTHON_CSRC_TORCH_DOC_ANY_H_

namespace k2 {

static constexpr const char *kRaggedAnyInitDataDoc = R"doc(
Create a ragged tensor with two axes.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([ [1, 2], [5], [], [9] ])
>>> a
[ [ 1 2 ] [ 5 ] [ ] [ 9 ] ]
>>> a.dtype
torch.int32
>>> b = k2r.Tensor([ [1, 3.0], [] ])
>>> b
[ [ 1 3 ] [ ] ]
>>> b.dtype
torch.float32
>>> c = k2r.Tensor([ [1] ], dtype=torch.float64)
>>> c
[ [ 1 ] ]
>>> c.dtype
torch.float64

Args:
  data:
    A list-of-list of integers or real numbers.
  dtype:
    Optional. If None, it infers the dtype from ``data``
    automatically, which is either ``torch.int32`` or
    ``torch.float32``.
)doc";

static constexpr const char *kRaggedAnyInitStrDoc = R"doc(
Create a ragged tensor from its string representation.

An example string for a 2-axis ragged tensor is given below::

    [ [1]  [2] ]

An example string for a 3-axis ragged tensor is given below::

    [ [[1] [2 3]]  [[2] [] [3, 4,]] ]

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor('[ [1] [] [3 4] ]')
>>> a
[ [ 1 ] [ ] [ 3 4 ] ]
>>> a.num_axes
2
>>> a.dtype
torch.int32
>>> b = k2r.Tensor('[ [[] [3]]  [[10]] ]', dtype=torch.float32)
>>> b
[ [ [ ] [ 3 ] ] [ [ 10 ] ] ]
>>> b.dtype
torch.float32
>>> b.num_axes
3

Note::
  Number of spaces in ``s`` does not affect the result.
  Of course, numbers have to be separated by at least one space.

Args:
  s:
    A string representation of the tensor.
  dtype:
    The desired dtype of the tensor. If it is ``None``, it tries
    to infer the correct dtype from `s`, which is assumed to be
    either ``torch.int32`` or ``torch.float32``.

)doc";

static constexpr const char *kRaggedAnyToDeviceDoc = R"doc(
Transfer this tensor to a given device.

Note::
  If `self` is already on the specified device, return a
  ragged tensor sharing the underlying memory with `self`.
  Otherwise, a new tensor is returned.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1], [2, 3]])
>>> a.device
device(type='cpu')
>>> b = a.to(torch.device('cuda', 0))
>>> b.device
device(type='cuda', index=0)

Args:
  device:
    The target device to move this tensor.

Returns:
  Return a tensor on the given device.
)doc";

static constexpr const char *kRaggedAnyToDtypeDoc = R"doc(
Convert this tensor to a specific dtype.

Note::
  If `self` is already of the specified `dtype`, return
  a ragged tensor sharing the underlying memory with `self`.
  Otherwise, a new tensor is returned.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1], [2, 3, 5]])
>>> a.dtype
torch.int32
>>> b = a.to(torch.float64)
>>> b.dtype
torch.float64

Caution::
  Currently, only for dtypes torch.int32, torch.float32, and
  torch.float64 are implemented. We can support other types
  if needed.

Args:
  dtype:
    The `dtype` this tensor should be converted to.

Returns:
  Return a tensor of the given `dtype`.
)doc";

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_DOC_ANY_H_
