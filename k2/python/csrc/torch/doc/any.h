

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

Note:
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

Note:
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

Note:
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

Caution:
  Currently, only support for dtypes ``torch.int32``, ``torch.float32``, and
  ``torch.float64`` are implemented. We can support other types if needed.

Args:
  dtype:
    The `dtype` this tensor should be converted to.

Returns:
  Return a tensor of the given `dtype`.
)doc";

static constexpr const char *kRaggedAnyStrDoc = R"doc(
Return a string representation of this tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1], [2, 3], []])
>>> a
[ [ 1 ] [ 2 3 ] [ ] ]
>>> str(a)
'[ [ 1 ] [ 2 3 ] [ ] ]'
)doc";

static constexpr const char *kRaggedAnyGetItemDoc = R"doc(
Select the i-th sublist along axis 0.

Caution:
  Support for autograd is to be implemented.

Note:
  It requires that this tensor has at least 3 axes.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor('[ [[1 3] [] [9]]  [[8]] ]')
>>> a
[ [ [ 1 3 ] [ ] [ 9 ] ] [ [ 8 ] ] ]
>>> a[0]
[ [ 1 3 ] [ ] [ 9 ] ]
>>> a[1]
[ [ 8 ] ]

Args:
  i:
    The i-th sublist along axis 0.
Returns:
  Return a new ragged tensor with one fewer axis.
)doc";

static constexpr const char *kRaggedAnyCloneDoc = R"doc(
Return a copy of this tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1, 2], [3]])
>>> b = a
>>> c = a.clone()
>>> a
[ [ 1 2 ] [ 3 ] ]
>>> b.data[0] = 10
>>> a
[ [ 10 2 ] [ 3 ] ]
>>> c
[ [ 1 2 ] [ 3 ] ]
>>> c.data[0] = -1
>>> c
[ [ -1 2 ] [ 3 ] ]
>>> a
[ [ 10 2 ] [ 3 ] ]
>>> b
[ [ 10 2 ] [ 3 ] ]
)doc";

static constexpr const char *kRaggedAnyEqDoc = R"doc(
Compare two ragged tensors.

Caution:
  The two tensors MUST have the same dtype. Otherwise,
  it throws.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1]])
>>> b = a.clone()
>>> a ==  b
True
>>> c = a.to(torch.float32)
>>> try:
...   c == b
... except RuntimeError:
...   print("raised exception")

Args:
  other:
    The tensor to be compared.
Returns:
  Return True if the two tensors are equal.
  Return False otherwise.
)doc";

static constexpr const char *kRaggedAnyNeDoc = R"doc(
Compare two ragged tensors.

Caution:
  The two tensors MUST have the same dtype. Otherwise,
  it throws.

>>> import torch
impor>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1, 2], [3]])
>>> b = a.clone()
>>> b != a
False
>>> c = k2r.Tensor([[1], [2], [3]])
>>> c != a
True

Args:
  other:
    The tensor to be compared.
Returns:
  Return ``True`` if the two tensors are NOT equal.
  Return ``False`` otherwise.
)doc";

static constexpr const char *kRaggedAnyRequiresGradPropDoc = R"doc(
Return ``True`` if gradients need to be computed for this tensor.
Return ``False`` otherwise.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1]], dtype=torch.float32)
>>> a.requires_grad
False
>>> a.requires_grad = True
>>> a.requires_grad
True
)doc";

static constexpr const char *kRaggedAnyGradPropDoc = R"doc(
This attribute is ``None`` by default. PyTorch will set it
during ``backward()``.

The attribute will contain the gradients computed and future
calls to ``backward()`` will accumulate (add) gradients into it.

>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1, 2], [3], [5, 6], []], dtype=torch.float32)
>>> a.requires_grad_(True)
[ [ 1 2 ] [ 3 ] [ 5 6 ] [ ] ]
>>> b = a.sum()
>>> b
tensor([ 3.,  3., 11.,  0.], grad_fn=<SumFunction>>)
>>> c = b * torch.arange(4)
>>> c.sum().backward()
>>> a.grad
tensor([0., 0., 1., 2., 2.])
)doc";

static constexpr const char *kRaggedAnyRequiresGradMethodDoc = R"doc(
Change if autograd should record operations on this tensor: Set
this tensor's :attr:`requires_grad` attribute **in-place**.

Note:
  If this tensor is not a float tensor, PyTorch will throw a
  RuntimeError exception.

Caution:
  This method ends with an underscore, meaning it changes this tensor
  **in-place**.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1]], dtype=torch.float64)
>>> a.requires_grad
False
>>> a.requires_grad_(True)
[ [ 1 ] ]
>>> a.requires_grad
True

Args:
  requires_grad:
    If autograd should record operations on this tensor.
Returns:
  Return this tensor.
)doc";

static constexpr const char *kRaggedAnySumDoc = R"doc(
Compute the sum of sublists over the last axis of this tensor.

Note:
  If a sublist is empty, the sum for it is the provided
  ``initial_value``.

Note:
  This operation supports autograd if this tensor is a float tensor,
  i.e., with dtype being torch.float32 or torch.float64.

>>> import k2.ragged as k2r
>>> a = k2r.Tensor('[ [[1 2] [] [5]]  [[10]] ]', dtype=torch.float32)
>>> a.requires_grad_(True)
[ [ [ 1 2 ] [ ] [ 5 ] ] [ [ 10 ] ] ]
>>> b = a.sum()
>>> c = (b * torch.arange(4)).sum()
>>> c.backward()
>>> a.grad
tensor([0., 0., 2., 3.])
>>> b
tensor([ 3.,  0.,  5., 10.], grad_fn=<SumFunction>>)
>>> c
tensor(40., grad_fn=<SumBackward0>)

Args:
  initial_value:
    This value is added to the sum of each sublist. So when
    a sublist is empty, its sum is this value.
Returns:
  Return a 1-D tensor with the same dtype of this tensor
  containing the computed sum.
)doc";

static constexpr const char *kRaggedAnyNumelDoc = R"doc(
Returns:
  Return number of elements in this tensor. It equals to
  `self.data.numel()`.
>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([[1], [], [3, 4, 5, 6]])
>>> a.numel()
5
>>> b = k2r.Tensor('[ [[1] [] []]  [[2 3]]]')
>>> b.numel()
3
)doc";
static constexpr const char *kRaggedAnyTotSizeDoc = R"doc(
Return the number of elements of an given axis. If axis is 0, it's
equivalent to the property ``dim0``.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor('[ [1 2 3] [] [5 8 ] ]')
>>> a.tot_size(0)
3
>>> a.tot_size(1)
5
>>> import k2.ragged as k2r
>>> b = k2r.Tensor('[ [[1 2 3] [] [5 8]] [[] [1 5 9 10 -1] [] [] []] ]')
>>> b.tot_size(0)
2
>>> b.tot_size(1)
8
>>> b.tot_size(2)
10
)doc";

static constexpr const char *kRaggedAnyGetStateDoc = R"doc(
__getstate__(self: _k2.ragged.Tensor) -> tuple

Requires a tensor with 2 axes or 3 axes. Other number
of axes are not implemented yet.

This method is to support ``pickle``, e.g., used by ``torch.save()``.
You are not expected to call it by yourself.

Returns:
  If this tensor has 2 axes, return a tuple containing
  (self.row_splits(1), "row_ids1", self.data).
  If this tensor has 3 axes, return a tuple containing
  (self.row_splits(1), "row_ids1", self.row_splits(1),
  "row_ids2", self.data)

Note:
  "row_ids1" and "row_ids2" in the returned value is for
  backward compatibility.
"""
)doc";

static constexpr const char *kRaggedAnySetStateDoc = R"doc(
__setstate__(self: _k2.ragged.Tensor, arg0: tuple) -> None

Set the content of this class from ``arg0``.

This method is to support ``pickle``, e.g., used by torch.load().
You are not expected to call it by yourself.

Args:
  arg0:
    It is the return value from the method ``__getstate__``.
)doc";

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_DOC_ANY_H_
