

#ifndef K2_PYTHON_CSRC_TORCH_V2_DOC_ANY_H_
#define K2_PYTHON_CSRC_TORCH_V2_DOC_ANY_H_

namespace k2 {

static constexpr const char *kCreateRaggedTensorDataDoc = R"doc(
Create a ragged tensor with arbitrary number of axes.

Note:
  A ragged tensor has at least two axes.

Hint:
  The returned tensor is on CPU.

>>> import k2.ragged as k2r
>>> a = k2r.create_ragged_tensor([ [1, 2], [5], [], [9] ])
>>> a
[ [ 1 2 ] [ 5 ] [ ] [ 9 ] ]
>>> a.dtype
torch.int32
>>> b = k2r.create_ragged_tensor([ [1, 3.0], [] ])
>>> b
[ [ 1 3 ] [ ] ]
>>> b.dtype
torch.float32
>>> c = k2r.create_ragged_tensor([ [1] ], dtype=torch.float64)
>>> c.dtype
torch.float64
>>> d = k2r.create_ragged_tensor([ [[1], [2, 3]], [[4], []] ])
>>> d
[ [ [ 1 ] [ 2 3 ] ] [ [ 4 ] [ ] ] ]
>>> d.num_axes
3
>>> e = k2r.create_ragged_tensor([])
>>> e
[ ]
>>> e.num_axes
2
>>> e.shape.row_splits(1)
tensor([0], dtype=torch.int32)
>>> e.shape.row_ids(1)
tensor([], dtype=torch.int32)

Args:
  data:
    A list-of sublist(s) of integers or real numbers.
    It can have arbitrary number of axes (at least two).
  dtype:
    Optional. If None, it infers the dtype from ``data``
    automatically, which is either ``torch.int32`` or
    ``torch.float32``. Supported dtypes are: ``torch.int32``,
    ``torch.float32``, and ``torch.float64``.
Returns:
  Return a ragged tensor.
)doc";

static constexpr const char *kCreateRaggedTensorStrDoc = R"doc(
Create a ragged tensor from its string representation.

An example string for a 2-axis ragged tensor is given below::

    [ [1]  [2] ]

An example string for a 3-axis ragged tensor is given below::

    [ [[1]] [[]] ]

Hint:
  The returned tensor is on CPU.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.create_ragged_tensor('[ [1] [] [3 4] ]')
>>> a
[ [ 1 ] [ ] [ 3 4 ] ]
>>> a.num_axes
2
>>> a.dtype
torch.int32
>>> b = k2r.create_ragged_tensor('[ [[] [3]]  [[10]] ]', dtype=torch.float32)
>>> b = k2r.create_ragged_tensor('[ [[] [3]]  [[10]] ]', dtype=torch.float32)
>>> b
[ [ [ ] [ 3 ] ] [ [ 10 ] ] ]
>>> b.dtype
torch.float32
>>> b.num_axes
3
>>> c = k2r.create_ragged_tensor('[[1.]]')
>>> c.dtype
torch.float32

Args:
  s:
    A string representation of a ragged tensor.
  dtype:
    The desired dtype of the tensor. If it is ``None``, it tries
    to infer the correct dtype from ``s``, which is assumed to be
    either ``torch.int32`` or ``torch.float32``. Supported dtypes are:
    ``torch.int32``, ``torch.float32``, and ``torch.float64``.
Returns:
  Return a ragged tensor.
)doc";

static constexpr const char *kRaggedAnyInitDataDoc = R"doc(
Create a ragged tensor with arbitrary number of axes.

Note:
  A ragged tensor has at least two axes.

Hint:
  The returned tensor is on CPU.

**Example 1**:

  >>> import torch
  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([ [1, 2], [5], [], [9] ])
  >>> a
  [ [ 1 2 ] [ 5 ] [ ] [ 9 ] ]
  >>> a.dtype
  torch.int32
  >>> b = k2r.RaggedTensor([ [1, 3.0], [] ])
  >>> b
  [ [ 1 3 ] [ ] ]
  >>> b.dtype
  torch.float32
  >>> c = k2r.RaggedTensor([ [1] ], dtype=torch.float64)
  >>> c
  [ [ 1 ] ]
  >>> c.dtype
  torch.float64
  >>> d = k2r.RaggedTensor([ [[1], [2, 3]], [[4], []] ])
  >>> d
  [ [ [ 1 ] [ 2 3 ] ] [ [ 4 ] [ ] ] ]
  >>> d.num_axes
  3
  >>> e = k2r.RaggedTensor([])
  >>> e
  [ ]
  >>> e.num_axes
  2
  >>> e.shape.row_splits(1)
  tensor([0], dtype=torch.int32)
  >>> e.shape.row_ids(1)
  tensor([], dtype=torch.int32)

**Example 2**:

  >>> k2r.RaggedTensor([ [[1, 2]], [], [[]] ])
  [ [ [ 1 2 ] ] [ ] [ [ ] ] ]

Args:
  data:
    A list-of sublist(s) of integers or real numbers.
    It can have arbitrary number of axes (at least two).
  dtype:
    Optional. If None, it infers the dtype from ``data``
    automatically, which is either ``torch.int32`` or
    ``torch.float32``. Supported dtypes are: ``torch.int32``,
    ``torch.float32``, and ``torch.float64``.
)doc";

static constexpr const char *kRaggedAnyInitStrDoc = R"doc(
Create a ragged tensor from its string representation.

An example string for a 2-axis ragged tensor is given below::

    [ [1]  [2] ]

An example string for a 3-axis ragged tensor is given below::

    [ [[1]] [[]] ]

Hint:
  The returned tensor is on CPU.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor('[ [1] [] [3 4] ]')
>>> a
[ [ 1 ] [ ] [ 3 4 ] ]
>>> a.num_axes
2
>>> a.dtype
torch.int32
>>> b = k2r.RaggedTensor('[ [[] [3]]  [[10]] ]', dtype=torch.float32)
>>> b
[ [ [ ] [ 3 ] ] [ [ 10 ] ] ]
>>> b.dtype
torch.float32
>>> b.num_axes
3
>>> c = k2r.RaggedTensor('[[1.]]')
>>> c.dtype
torch.float32

Note:
  Number of spaces in ``s`` does not affect the result.
  Of course, numbers have to be separated by at least one space.

Args:
  s:
    A string representation of a ragged tensor.
  dtype:
    The desired dtype of the tensor. If it is ``None``, it tries
    to infer the correct dtype from ``s``, which is assumed to be
    either ``torch.int32`` or ``torch.float32``. Supported dtypes are:
    ``torch.int32``, ``torch.float32``, and ``torch.float64``.
)doc";

static constexpr const char *kRaggedAnyToDeviceDoc = R"doc(
Transfer this tensor to a given device.

Note:
  If ``self`` is already on the specified device, return a
  ragged tensor sharing the underlying memory with ``self``.
  Otherwise, a new tensor is returned.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1], [2, 3]])
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

static constexpr const char *kRaggedAnyToDeviceStrDoc = R"doc(
Transfer this tensor to a given device.

Note:
  If ``self`` is already on the specified device, return a
  ragged tensor sharing the underlying memory with ``self``.
  Otherwise, a new tensor is returned.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1]])
>>> a.device
device(type='cpu')
>>> b = a.to('cuda:0')
>>> b.device
device(type='cuda', index=0)
>>> c = b.to('cpu')
>>> c.device
device(type='cpu')
>>> d = c.to('cuda:1')
>>> d.device
device(type='cuda', index=1)

Args:
  device:
    The target device to move this tensor.
    Note: The device is represented as a string.
    Valid strings are: "cpu", "cuda:0", "cuda:1", etc.

Returns:
  Return a tensor on the given device.
)doc";

static constexpr const char *kRaggedAnyToDtypeDoc = R"doc(
Convert this tensor to a specific dtype.

Note:
  If ``self`` is already of the specified `dtype`, return
  a ragged tensor sharing the underlying memory with ``self``.
  Otherwise, a new tensor is returned.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1], [2, 3, 5]])
>>> a.dtype
torch.int32
>>> b = a.to(torch.float64)
>>> b.dtype
torch.float64

Caution:
  Currently, only support dtypes ``torch.int32``, ``torch.float32``, and
  ``torch.float64``. We can support other types if needed.

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
>>> a = k2r.RaggedTensor([[1], [2, 3], []])
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
>>> a = k2r.RaggedTensor('[ [[1 3] [] [9]]  [[8]] ]')
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
>>> a = k2r.RaggedTensor([[1, 2], [3]])
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
  it throws an exception.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1]])
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
  Return ``True`` if the two tensors are equal.
  Return ``False`` otherwise.
)doc";

static constexpr const char *kRaggedAnyNeDoc = R"doc(
Compare two ragged tensors.

Caution:
  The two tensors MUST have the same dtype. Otherwise,
  it throws an exception.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1, 2], [3]])
>>> b = a.clone()
>>> b != a
False
>>> c = k2r.RaggedTensor([[1], [2], [3]])
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
>>> a = k2r.RaggedTensor([[1]], dtype=torch.float32)
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
>>> a = k2r.RaggedTensor([[1, 2], [3], [5, 6], []], dtype=torch.float32)
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
>>> a = k2r.RaggedTensor([[1]], dtype=torch.float64)
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
>>> a = k2r.RaggedTensor('[ [[1 2] [] [5]]  [[10]] ]', dtype=torch.float32)
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
  ``self.data.numel()``.
>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1], [], [3, 4, 5, 6]])
>>> a.numel()
5
>>> b = k2r.RaggedTensor('[ [[1] [] []]  [[2 3]]]')
>>> b.numel()
3
)doc";
static constexpr const char *kRaggedAnyTotSizeDoc = R"doc(
Return the number of elements of an given axis. If axis is 0, it's
equivalent to the property ``dim0``.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor('[ [1 2 3] [] [5 8 ] ]')
>>> a.tot_size(0)
3
>>> a.tot_size(1)
5
>>> import k2.ragged as k2r
>>> b = k2r.RaggedTensor('[ [[1 2 3] [] [5 8]] [[] [1 5 9 10 -1] [] [] []] ]')
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

static constexpr const char *kRaggedAnyDtypeDoc = R"doc(
Return the dtype of this tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1], []])
>>> a.dtype
torch.int32
>>> a = a.to(torch.float32)
>>> a.dtype
torch.float32
>>> b = k2r.RaggedTensor([[3]], dtype=torch.float64)
>>> b.dtype
torch.float64
)doc";

static constexpr const char *kRaggedAnyDeviceDoc = R"doc(
Return the device of this tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1]])
>>> a.device
device(type='cpu')
>>> b = a.to(torch.device('cuda', 0))
>>> b.device
device(type='cuda', index=0)
>>> b.device == torch.device('cuda:0')
)doc";

static constexpr const char *kRaggedAnyDataDoc = R"doc(
Return the underlying memory as a 1-D tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1, 2], [], [5], [], [8, 9, 10]])
>>> a.data
tensor([ 1,  2,  5,  8,  9, 10], dtype=torch.int32)
>>> isinstance(a.data, torch.Tensor)
True
>>> a.data[0] = -1
>>> a
[ [ -1 2 ] [ ] [ 5 ] [ ] [ 8 9 10 ] ]
>>> a.data[3] = -3
>>> a
[ [ -1 2 ] [ ] [ 5 ] [ ] [ -3 9 10 ] ]
>>> a.data[2] = -2
>>> a
[ [ -1 2 ] [ ] [ -2 ] [ ] [ -3 9 10 ] ]
)doc";

static constexpr const char *kRaggedAnyIsCudaDoc = R"doc(
Returns:
  Return ``True`` if the tensor is stored on the GPU, ``False``
  otherwise.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1]])
>>> a.is_cuda
False
>>> b = a.to(torch.device('cuda', 0))
>>> b.is_cuda
True
)doc";

static constexpr const char *kRaggedAnyNumAxesDoc = R"doc(
Return the number of axes of this tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor('[ [] [] [] [] ]')
>>> a.num_axes
2
>>> b = k2r.RaggedTensor('[ [[] []] [[]] ]')
>>> b.num_axes
3
>>> c = k24.Tensor('[ [ [[] [1]] [[3 4] []] ]  [ [[1]] [[2] [3 4]] ] ]')
>>> c.num_axes
4

Returns:
  Return number of axes of this tensor, which is at least 2.
)doc";

static constexpr const char *kRaggedAnyDim0Doc = R"doc(
Return number of sublists at axis 0.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([ [1, 2], [3], [], [], [] ])
>>> a.dim0
5
>>> b = k2r.RaggedTensor('[ [[]] [[] []]]')
>>> b.dim0
2
)doc";

static constexpr const char *kRaggedAnyRemoveAxisDoc = R"doc(
Remove an axis; if it is not the first or last axis, this is done by appending
lists (effectively the axis is combined with the following axis).  If it is the
last axis it is just removed and the number of elements may be changed.

Caution:
  The tensor has to have more than two axes.

**Example 1**:

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([ [[1], [], [0, -1]], [[], [2, 3], []], [[0]], [[]] ])
  >>> a
  [ [ [ 1 ] [ ] [ 0 -1 ] ] [ [ ] [ 2 3 ] [ ] ] [ [ 0 ] ] [ [ ] ] ]
  >>> a.num_axes
  3
  >>> b = a.remove_axis(0)
  >>> b
  [ [ 1 ] [ ] [ 0 -1 ] [ ] [ 2 3 ] [ ] [ 0 ] [ ] ]
  >>> c = a.remove_axis(1)
  >>> c
  [ [ 1 0 -1 ] [ 2 3 ] [ 0 ] [ ] ]

**Example 2**:

  >>> a = k2r.RaggedTensor([ [[[1], [], [2]]], [[[3, 4], [], [5, 6], []]], [[[], [0]]] ])
  >>> a.num_axes
  4
  >>> a
  [ [ [ [ 1 ] [ ] [ 2 ] ] ] [ [ [ 3 4 ] [ ] [ 5 6 ] [ ] ] ] [ [ [ ] [ 0 ] ] ] ]
  >>> b = a.remove_axis(0)
  >>> b
  [ [ [ 1 ] [ ] [ 2 ] ] [ [ 3 4 ] [ ] [ 5 6 ] [ ] ] [ [ ] [ 0 ] ] ]
  >>> c = a.remove_axis(1)
  >>> c
  [ [ [ 1 ] [ ] [ 2 ] ] [ [ 3 4 ] [ ] [ 5 6 ] [ ] ] [ [ ] [ 0 ] ] ]
  >>> d = a.remove_axis(2)
  >>> d
  [ [ [ 1 2 ] ] [ [ 3 4 5 6 ] ] [ [ 0 ] ] ]

Args:
  axis:
    The axis to move.
Returns:
  Return a ragged tensor with one fewer axes.
)doc";

static constexpr const char *kRaggedAnyArangeDoc = R"doc(
Return a sub-range of ``self`` containing indexes ``begin``
through ``end - 1`` along axis ``axis`` of ``self``.

The ``axis`` argument may be confusing; its behavior is equivalent to:

.. code-block:: python

  for i in range(axis):
    self = self.remove_axis(0)

  return self.arange(0, begin, end)

Caution:
  The returned tensor shares the underlying memory with ``self``.

**Example 1**

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([ [[1], [], [2]], [[], [4, 5], []], [[], [1]], [[]] ])
  >>> a
  [ [ [ 1 ] [ ] [ 2 ] ] [ [ ] [ 4 5 ] [ ] ] [ [ ] [ 1 ] ] [ [ ] ] ]
  >>> a.num_axes
  3
  >>> b = a.arange(axis=0, begin=1, end=3)
  >>> b
  [ [ [ ] [ 4 5 ] [ ] ] [ [ ] [ 1 ] ] ]
  >>> b.num_axes
  3
  >>> c = a.arange(axis=0, begin=1, end=2)
  >>> c
  [ [ [ ] [ 4 5 ] [ ] ] ]
  >>> c.num_axes
  3
  >>> d = a.arange(axis=1, begin=0, end=4)
  >>> d
  [ [ 1 ] [ ] [ 2 ] [ ] ]
  >>> d.num_axes
  2
  >>> e = a.arange(axis=1, begin=2, end=5)
  >>> e
  [ [ 2 ] [ ] [ 4 5 ] ]
  >>> e.num_axes
  2

**Example 2**

  >>> a = k2r.RaggedTensor([ [[[], [1], [2, 3]],[[5, 8], [], [9]]], [[[10], [0], []]], [[[], [], [1]]] ])
  >>> a.num_axes
  4
  >>> b = a.arange(axis=0, begin=0, end=2)
  >>> b
  [ [ [ [ ] [ 1 ] [ 2 3 ] ] [ [ 5 8 ] [ ] [ 9 ] ] ] [ [ [ 10 ] [ 0 ] [ ] ] ] ]
  >>> b.num_axes
  4
  >>> c = a.arange(axis=1, begin=1, end=3)
  >>> c
  [ [ [ 5 8 ] [ ] [ 9 ] ] [ [ 10 ] [ 0 ] [ ] ] ]
  >>> c.num_axes
  3
  >>> d = a.arange(axis=2, begin=0, end=5)
  >>> d
  [ [ ] [ 1 ] [ 2 3 ] [ 5 8 ] [ ] ]
  >>> d.num_axes
  2

**Example 3**

  >>> a = k2r.RaggedTensor([[0], [1], [2], [], [3]])
  >>> a
  [ [ 0 ] [ 1 ] [ 2 ] [ ] [ 3 ] ]
  >>> a.num_axes
  2
  >>> b = a.arange(axis=0, begin=1, end=4)
  >>> b
  [ [ 1 ] [ 2 ] [ ] ]
  >>> b.data[0] = -1
  >>> a
  [ [ 0 ] [ -1 ] [ 2 ] [ ] [ 3 ] ]

Args:
  axis:
    The axis from which ``begin`` and ``end`` correspond to.
  begin:
    The beginning of the range (inclusive).
  end:
    The end of the range (exclusive).

)doc";

static constexpr const char *kRaggedAnyRemoveValuesEqDoc = R"doc(
Returns a ragged tensor after removing all 'values' that equal a provided
target.  Leaves all layers of the shape except for the last one unaffected.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1, 2, 3, 0, 3, 2], [], [3, 2, 3], [3]])
>>> b = a.remove_values_eq(3)
>>> b
[ [ 1 2 0 2 ] [ ] [ 2 ] [ ] ]
>>> c = a.remove_values_eq(2)
>>> c
[ [ 1 3 0 3 ] [ ] [ 3 3 ] [ 3 ] ]

Args:
  target:
    The target value to delete.
Return:
  Return a ragged tensor whose values don't contain the ``target``.
)doc";

static constexpr const char *kRaggedAnyRemoveValuesLeqDoc = R"doc(
Returns a ragged tensor after removing all 'values' that are
equal to or less than a provided cutoff.
Leaves all layers of the shape except for the last one unaffected.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1, 2, 3, 0, 3, 2], [], [3, 2, 3], [3]])
>>> b = a.remove_values_leq(3)
>>> b
[ [ ] [ ] [ ] [ ] ]
>>> c = a.remove_values_leq(2)
>>> c
[ [ 3 3 ] [ ] [ 3 3 ] [ 3 ] ]
>>> d = a.remove_values_leq(1)
>>> d
[ [ 2 3 3 2 ] [ ] [ 3 2 3 ] [ 3 ] ]

Args:
  cutoff:
    Values less than or equal to this ``cutoff`` are deleted.
Return:
  Return a ragged tensor whose values are all above ``cutoff``.
)doc";

static constexpr const char *kRaggedAnyArgMaxDoc = R"doc(
Return a tensor containing maximum value indexes within each sub-list along the
last axis of ``self``, i.e. the max taken over the last axis, The index is -1
if the sub-list was empty or all values in the sub-list are less
than ``initial_value``.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([ [3, -1], [], [], [] ])
>>> a.argmax()
tensor([ 0, -1, -1, -1], dtype=torch.int32)
>>> b = a.argmax(initial_value=0)
>>> b
tensor([ 0, -1, -1, -1], dtype=torch.int32)
>>> c = k2r.RaggedTensor([ [3, 0, 2, 5, 1], [], [1, 3, 8, 2, 0] ])
>>> c.argmax()
tensor([ 3, -1,  7], dtype=torch.int32)
>>> d = c.argmax(initial_value=0)
>>> d
tensor([ 3, -1,  7], dtype=torch.int32)
>>> c.data[3], c.data[7]
(tensor(5, dtype=torch.int32), tensor(8, dtype=torch.int32))
>>> c.argmax(initial_value=6)
tensor([-1, -1,  7], dtype=torch.int32)
>>> c.to('cuda:0').argmax(0)
tensor([ 3, -1,  7], device='cuda:0', dtype=torch.int32)
>>> import torch
>>> c.to(torch.float32).argmax(0)
tensor([ 3, -1,  7], dtype=torch.int32)

Args:
  initial_value:
    A base value to compare. If values in a sublist are all less
    than this value, then the ``argmax`` of this sublist is -1.
    If a sublist is empty, the ``argmax`` of it is also -1.
    If it is ``None``, the lowest value of ``self.dtype`` is used.

Returns:
  Return a 1-D ``torch.int32`` tensor. It is on the same device
  as ``self``.
)doc";

static constexpr const char *kRaggedAnyMaxDoc = R"doc(
Return a tensor containing the maximum of each sub-list along the last
axis of ``self``. The max is taken over the last axis or ``initial_value``,
whichever was larger.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([ [[1, 3, 0], [2, 5, -1, 1, 3], [], []], [[1, 8, 9, 2], [], [2, 4, 6, 8]] ])
>>> a.max()
tensor([          3,           5, -2147483648, -2147483648,           9,
        -2147483648,           8], dtype=torch.int32)
>>> a.max(initial_value=-10)
tensor([  3,   5, -10, -10,   9, -10,   8], dtype=torch.int32)
>>> a.max(initial_value=7)
tensor([7, 7, 7, 7, 9, 7, 8], dtype=torch.int32)
>>> import torch
>>> a.to(torch.float32).max(-3)
tensor([ 3.,  5., -3., -3.,  9., -3.,  8.])
>>> a.to('cuda:0').max(-2)
tensor([ 3,  5, -2, -2,  9, -2,  8], device='cuda:0', dtype=torch.int32)

Args:
  initial_value:
   The base value to compare. If values in a sublist are all less
   than this value, then the max of this sublist is ``initial_value``.
   If a sublist is empty, its max is also ``initial_value``.

Returns:
  Return 1-D tensor containing the max value of each sublist.
  It shares the same dtype and device with ``self``.
)doc";

static constexpr const char *kRaggedAnyMinDoc = R"doc(
Return a tensor containing the minimum of each sub-list along the last
axis of ``self``. The min is taken over the last axis or ``initial_value``,
whichever was smaller.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([ [[1, 3, 0], [2, 5, -1, 1, 3], [], []], [[1, 8, 9, 2], [], [2, 4, 6, 8]] ], dtype=torch.float32)
>>> a.min()
tensor([ 0.0000e+00, -1.0000e+00,  3.4028e+38,  3.4028e+38,  1.0000e+00,
         3.4028e+38,  2.0000e+00])
>>> a.min(initial_value=float('inf'))
tensor([ 0., -1., inf, inf,  1., inf,  2.])
>>> a.min(100)
tensor([  0.,  -1., 100., 100.,   1., 100.,   2.])
>>> a.to(torch.int32).min(20)
tensor([ 0, -1, 20, 20,  1, 20,  2], dtype=torch.int32)
>>> a.to('cuda:0').min(15)
tensor([ 0., -1., 15., 15.,  1., 15.,  2.], device='cuda:0')

Args:
  initial_value:
   The base value to compare. If values in a sublist are all larger
   than this value, then the minimum of this sublist is ``initial_value``.
   If a sublist is empty, its minimum is also ``initial_value``.

Returns:
  Return 1-D tensor containing the minimum of each sublist.
  It shares the same dtype and device with ``self``.
)doc";

static constexpr const char *kRaggedCatDoc = R"doc(
Concatenate a list of ragged tensor over a specified axis.

**Example 1**

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([[1], [], [2, 3]])
  >>> k2r.cat([a, a], axis=0)
  [ [ 1 ] [ ] [ 2 3 ] [ 1 ] [ ] [ 2 3 ] ]
  >>> k2r.cat((a, a), axis=1)
  [ [ 1 1 ] [ ] [ 2 3 2 3 ] ]

**Example 2**

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([[1, 3], [], [5, 8], [], [9]])
  >>> b = k2r.RaggedTensor([[0], [1, 8], [], [-1], [10]])
  >>> c = k2r.cat([a, b], axis=0)
  >>> c
  [ [ 1 3 ] [ ] [ 5 8 ] [ ] [ 9 ] [ 0 ] [ 1 8 ] [ ] [ -1 ] [ 10 ] ]
  >>> c.num_axes
  2
  >>> d = k2r.cat([a, b], axis=1)
  >>> d
  [ [ 1 3 0 ] [ 1 8 ] [ 5 8 ] [ -1 ] [ 9 10 ] ]
  >>> d.num_axes
  2
  >>> k2r.RaggedTensor.cat([a, b], axis=1)
  [ [ 1 3 0 ] [ 1 8 ] [ 5 8 ] [ -1 ] [ 9 10 ] ]
  >>> k2r.cat((b, a), axis=0)
  [ [ 0 ] [ 1 8 ] [ ] [ -1 ] [ 10 ] [ 1 3 ] [ ] [ 5 8 ] [ ] [ 9 ] ]

Args:
  srcs:
    A list (or a tuple) of ragged tensors to concatenate. They **MUST** all
    have the same dtype and on the same device.
  axis:
    Only 0 and 1 are supported right now. If it is 1, then
    ``srcs[i].dim0`` must all have the same value.

Return:
  Return a concatenated tensor.
)doc";

static constexpr const char *kRaggedAnyUniqueDoc = R"doc(
If ``self`` has two axes, this will return the unique sub-lists
(in a possibly different order, but without repeats).
If ``self`` has 3 axes, it will do the above but separately for each
index on axis 0; if more than 3 axes, the earliest axes will be ignored.

Caution:
  It does not completely guarantee that all unique sequences will be
  present in the output, as it relies on hashing and ignores collisions.
  If several sequences have the same hash, only one of them is kept, even
  if the actual content in the sequence is different.

Caution:
  Even if there are no repeated sequences, the output may be different
  from ``self``. That is, `new2old_indexes` may NOT be an identity map even
  if nothing was removed.

**Example 1**

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([[3, 1], [3], [1], [1], [3, 1], [2]])
  >>> a.unique()
  ([ [ 1 ] [ 2 ] [ 3 ] [ 3 1 ] ], None, None)
  >>> a.unique(need_num_repeats=True, need_new2old_indexes=True)
  ([ [ 1 ] [ 2 ] [ 3 ] [ 3 1 ] ], [ [ 2 1 1 2 ] ], tensor([2, 5, 1, 0], dtype=torch.int32))
  >>> a.unique(need_num_repeats=True)
  ([ [ 1 ] [ 2 ] [ 3 ] [ 3 1 ] ], [ [ 2 1 1 2 ] ], None)
  >>> a.unique(need_new2old_indexes=True)
  ([ [ 1 ] [ 2 ] [ 3 ] [ 3 1 ] ], None, tensor([2, 5, 1, 0], dtype=torch.int32))

**Example 2**

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([[[1, 2], [2, 1], [1, 2], [1, 2]], [[3], [2], [0, 1], [2]], [[], [2, 3], [], [3]] ])
  >>> a.unique()
  ([ [ [ 1 2 ] [ 2 1 ] ] [ [ 2 ] [ 3 ] [ 0 1 ] ] [ [ ] [ 3 ] [ 2 3 ] ] ], None, None)
  >>> a.unique(need_num_repeats=True, need_new2old_indexes=True)
  ([ [ [ 1 2 ] [ 2 1 ] ] [ [ 2 ] [ 3 ] [ 0 1 ] ] [ [ ] [ 3 ] [ 2 3 ] ] ], [ [ 3 1 ] [ 2 1 1 ] [ 2 1 1 ] ], tensor([ 0,  1,  5,  4,  6,  8, 11,  9], dtype=torch.int32))
  >>> a.unique(need_num_repeats=True)
  ([ [ [ 1 2 ] [ 2 1 ] ] [ [ 2 ] [ 3 ] [ 0 1 ] ] [ [ ] [ 3 ] [ 2 3 ] ] ], [ [ 3 1 ] [ 2 1 1 ] [ 2 1 1 ] ], None)
  >>> a.unique(need_new2old_indexes=True)
  ([ [ [ 1 2 ] [ 2 1 ] ] [ [ 2 ] [ 3 ] [ 0 1 ] ] [ [ ] [ 3 ] [ 2 3 ] ] ], None, tensor([ 0,  1,  5,  4,  6,  8, 11,  9], dtype=torch.int32))

**Example 3**

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([[1], [3], [2]])
  >>> a.unique(True, True)
  ([ [ 1 ] [ 2 ] [ 3 ] ], [ [ 1 1 1 ] ], tensor([0, 2, 1], dtype=torch.int32))


Args:
  need_num_repeats:
    If True, it also returns the number of repeats of each sequence.
  need_new2old_indexes:
    If true, it returns an extra 1-D tensor `new2old_indexes`.
    If `src` has 2 axes, this tensor contains `src_idx0`;
    if `src` has 3 axes, this tensor contains `src_idx01`.

    Caution:
      For repeated sublists, only one of them is kept.
      The choice of which one to keep is **deterministic** and
      is an implementation detail.

Returns:
  Returns a tuple containing:

    - ans: A ragged tensor with the same number of axes as ``self`` and possibly
      fewer elements due to removing repeated sequences on the last axis
      (and with the last-but-one indexes possibly in a different order).

    - num_repeats: A tensor containing number of repeats of each returned
      sequence if ``need_num_repeats`` is True; it is ``None`` otherwise.
      If it is not ``None``, ``num_repeats.num_axes`` is always 2.
      If ``ans.num_axes`` is 2, then ``num_repeats.dim0 == 1`` and
      ``num_repeats.numel() == ans.dim0``.
      If ``ans.num_axes`` is 3, then ``num_repeats.dim0 == ans.dim0`` and
      ``num_repeats.numel() == ans.tot_size(1)``.

    - new2old_indexes: A 1-D tensor whose i-th element specifies the
      input sublist that the i-th output sublist corresponds to.

)doc";

static constexpr const char *kRaggedAnyNormalizeDoc = R"doc(
Normalize a ragged tensor over the last axis.

If ``use_log`` is ``True``, the normalization per sublist is done as follows:

    1. Compute the log sum per sublist
    2. Subtract the log sum computed above from the sublist and return
    it

If ``use_log`` is ``False``, the normalization per sublist is done as follows:

    1. Compute the sum per sublist
    2. Divide the sublist by the above sum and return the resulting sublist

Note:
  If a sublist contains 3 elements ``[a, b, c]``, then the log sum
  is defined as::

    s = log(exp(a) + exp(b) + exp(c))

  The resulting sublist looks like below if ``use_log`` is ``True``::

    [a - s, b - s, c - s]

  If ``use_log`` is ``False``, the resulting sublist looks like::

    [a/(a+b+c), b/(a+b+c), c/(a+b+c)]

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[0.1, 0.3], [], [1], [0.2, 0.8]])
>>> a.normalize(use_log=False)
[ [ 0.25 0.75 ] [ ] [ 1 ] [ 0.2 0.8 ] ]
>>> a.normalize(use_log=True)
[ [ -0.798139 -0.598139 ] [ ] [ 0 ] [ -1.03749 -0.437488 ] ]
>>> b = k2r.RaggedTensor([ [[0.1, 0.3], []], [[1], [0.2, 0.8]] ])
>>> b.normalize(use_log=False)
[ [ [ 0.25 0.75 ] [ ] ] [ [ 1 ] [ 0.2 0.8 ] ] ]
>>> b.normalize(use_log=True)
[ [ [ -0.798139 -0.598139 ] [ ] ] [ [ 0 ] [ -1.03749 -0.437488 ] ] ]
>>> a.num_axes
2
>>> b.num_axes
3
>>> import torch
>>> (torch.tensor([0.1, 0.3]).exp() / torch.tensor([0.1, 0.3]).exp().sum()).log()
tensor([-0.7981, -0.5981])

Args:
  use_log:
    It indicates which kind of normalization to be applied.
Returns:
  Returns a 1-D tensor, sharing the same dtype and device with ``self``.
)doc";

static constexpr const char *kRaggedAnyPadDoc = R"doc(
Pad a ragged tensor with 2-axes to a 2-D torch tensor.

For example, if ``self`` has the following values::

    [ [1 2 3] [4] [5 6 7 8] ]

Then it returns a 2-D tensor as follows if ``padding_value`` is 0 and
mode is ``constant``::

    tensor([[1, 2, 3, 0],
            [4, 0, 0, 0],
            [5, 6, 7, 8]])

Caution:
  It requires that ``self.num_axes == 2``.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([[1], [], [2, 3], [5, 8, 9, 8, 2]])
>>> a.pad(mode='constant', padding_value=-1)
tensor([[ 1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [ 2,  3, -1, -1, -1],
        [ 5,  8,  9,  8,  2]], dtype=torch.int32)
>>> a.pad(mode='replicate', padding_value=-1)
tensor([[ 1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1],
        [ 2,  3,  3,  3,  3],
        [ 5,  8,  9,  8,  2]], dtype=torch.int32)

Args:
  mode:
    Valid values are: ``constant``, ``replicate``. If it is
    ``constant``, the given ``padding_value`` is used for filling.
    If it is ``replicate``, the last entry in a list is used for filling.
    If a list is empty, then the given `padding_value` is also used for filling.
  padding_value:
    The filling value.

Returns:
  A 2-D torch tensor, sharing the same dtype and device with ``self``.
)doc";

static constexpr const char *kRaggedAnyToListDoc = R"doc(
Turn a ragged tensor into a list of lists [of lists..].

Hint:
  You can pass the returned list to the constructor of :class:`RaggedTensor`.

>>> a = k2r.RaggedTensor([ [[], [1, 2], [3], []], [[5, 6, 7]], [[], [0, 2, 3], [], []]])
>>> a.tolist()
[[[], [1, 2], [3], []], [[5, 6, 7]], [[], [0, 2, 3], [], []]]
>>> b = k2r.RaggedTensor(a.tolist())
>>> a == b
True
>>> c = k2r.RaggedTensor([[1.], [2.], [], [3.25, 2.5]])
>>> c.tolist()
[[1.0], [2.0], [], [3.25, 2.5]]

Returns:
   A list of list of lists [of lists ...] containing the same elements
   and structure as ``self``.
)doc";

static constexpr const char *kRaggedAnySortDoc = R"doc(
Sort a ragged tensor over the last axis **in-place**.

Caution:
  ``sort_`` ends with an underscore, meaning this operation
  changes ``self`` **in-place**.

>>> import k2.ragged as k2r
>>> a = k2r.RaggedTensor([ [1, 3, 0], [2, 5, 3], [], [1, 3, 0.] ])
>>> a_clone = a.clone()
>>> b = a.sort_(descending=True, need_new2old_indexes=True)
>>> b
tensor([1, 0, 2, 4, 5, 3, 7, 6, 8], dtype=torch.int32)
>>> a
[ [ 3 1 0 ] [ 5 3 2 ] [ ] [ 3 1 0 ] ]
>>> a_clone.data[b.long()]
tensor([3., 1., 0., 5., 3., 2., 3., 1., 0.])
>>> a_clone = a.clone()
>>> c = a.sort_(descending=False, need_new2old_indexes=True)
>>> c
tensor([2, 1, 0, 5, 4, 3, 8, 7, 6], dtype=torch.int32)
>>> a
[ [ 0 1 3 ] [ 2 3 5 ] [ ] [ 0 1 3 ] ]
>>> a_clone.data[c.long()]
tensor([0., 1., 3., 2., 3., 5., 0., 1., 3.])

Args:
  descending:
    ``True`` to sort in **descending** order.
    ``False`` to sort in **ascending** order.
  need_new2old_indexes:
    If ``True``, also returns a 1-D tensor, containing the indexes mapping
    from the sorted elements to the unsorted elements. We can use
    ``self.clone().data[returned_tensor]`` to get a sorted tensor.
Returns:
  If ``need_new2old_indexes`` is False, returns None. Otherwise, returns
  a 1-D tensor of dtype ``torch.int32``.
)doc";

static constexpr const char *kRaggedAnyRaggedIndexDoc = R"doc(
Index a ragged tensor with a ragged tensor.

**Example 1**:

  >>> import k2.ragged as k2r
  >>> src = k2r.RaggedTensor([[10, 11], [12, 13.5]])
  >>> indexes = k2r.RaggedTensor([[0, 1]])
  >>> src.index(indexes, remove_axis=True)
  [ [ 10 11 12 13.5 ] ]
  >>> src.index(indexes, remove_axis=False)
  [ [ [ 10 11 ] [ 12 13.5 ] ] ]
  >>> i = k2r.RaggedTensor([[0], [1], [0, 0]])
  >>> src.index(i, remove_axis=True)
  [ [ 10 11 ] [ 12 13.5 ] [ 10 11 10 11 ] ]
  >>> src.index(i, remove_axis=False)
  [ [ [ 10 11 ] ] [ [ 12 13.5 ] ] [ [ 10 11 ] [ 10 11 ] ] ]

**Example 2**:

  >>> import k2.ragged as k2r
  >>> src = k2r.RaggedTensor([ [[1, 0], [], [2]], [[], [3], [0, 0, 1]], [[1, 2], [-1]]])
  >>> i = k2r.RaggedTensor([[[0, 2], [1]], [[0]]])
  >>> src.index(i, remove_axis=True)
  [ [ [ [ 1 0 2 ] [ 1 2 -1 ] ] [ [ 3 0 0 1 ] ] ] [ [ [ 1 0 2 ] ] ] ]
  >>> src.index(i, remove_axis=False)
  [ [ [ [ [ 1 0 ] [ ] [ 2 ] ] [ [ 1 2 ] [ -1 ] ] ] [ [ [ ] [ 3 ] [ 0 0 1 ] ] ] ] [ [ [ [ 1 0 ] [ ] [ 2 ] ] ] ] ]

Args:
  indexes:
    Its values must satisfy ``0 <= data[i] < self.dim0``.

    Caution:
      Its dtype has to be ``torch.int32``.

  remove_axis:
    If ``True``, then we remove the last-but-one axis,
    which has the effect of appending lists, e.g.
    if ``self`` is ``[[ 10 11 ] [ 12 13 ]]`` and ``indexes``
    is ``[[0 1]]`, this function will give us ``[[ 10 11 12 13 ]]``.
    If ``False`` the answer will have at least 3 axes, e.g., ``[[[10 11]] [12 13]]]`` ,
    in this case.
Returns:
  Return indexed tensor.
)doc";

static constexpr const char *kRaggedAnyTensorIndexDoc = R"doc(
Indexing operation on ragged tensor, returns ``self[indexes]``, where
the elements of ``indexes`` are interpreted as indexes into axis ``axis`` of
``self``.

Caution:
  ``indexes`` is a 1-D tensor and ``indexes.dtype == torch.int32``.

**Example 1**:

  >>> import torch
  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([[0, 2, 3], [], [0, 1, 2], [], [], [3, -1.25]])
  >>> i = torch.tensor([2, 0, 3, 5], dtype=torch.int32)
  >>> b, value_indexes = a.index(i, axis=0, need_value_indexes=True)
  >>> b
  [ [ 0 1 2 ] [ 0 2 3 ] [ ] [ 3 -1.25 ] ]
  >>> value_indexes
  tensor([3, 4, 5, 0, 1, 2, 6, 7], dtype=torch.int32)
  >>> a.data[value_indexes.long()]
  tensor([ 0.0000,  1.0000,  2.0000,  0.0000,  2.0000,  3.0000,  3.0000, -1.2500])
  >>> k = torch.tensor([2, -1, 0], dtype=torch.int32)
  >>> a.index(k, axis=0, need_value_indexes=True)
  ([ [ 0 1 2 ] [ ] [ 0 2 3 ] ], tensor([3, 4, 5, 0, 1, 2], dtype=torch.int32))

**Example 2**:

  >>> import k2.ragged as k2r
  >>> a = k2r.RaggedTensor([ [[1, 3], [], [2]], [[5, 8], [], [-1], [2]] ])
  >>> i = torch.tensor([0, 2, 1, 6, 3, 5, 4], dtype=torch.int32)
  >>> a.shape.row_ids(1)[i.long()]
  tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
  >>> b, value_indexes = a.index(i, axis=1, need_value_indexes=True)
  >>> b
  [ [ [ 1 3 ] [ 2 ] [ ] ] [ [ 2 ] [ 5 8 ] [ -1 ] [ ] ] ]
  >>> value_indexes
  tensor([0, 1, 2, 6, 3, 4, 5], dtype=torch.int32)
  >>> a.data[value_indexes.long()]
  tensor([ 1,  3,  2,  2,  5,  8, -1], dtype=torch.int32)


Args:
  indexes:
    Array of indexes, which will be interpreted as indexes into axis ``axis``
    of ``self``, i.e. with ``0 <= indexes[i] < self.tot_size(axis)``.
    Note that if ``axis`` is 0, then -1 is also a valid entry in ``index``,
    -1 as an index, which will result in an empty list (as if it were the index
    into a position in ``self`` that had an empty list at that point).

    Caution:
      It is currently not allowed to change the order on axes less than
      ``axis``, i.e. if ``axis > 0``, we require:
      ``IsMonotonic(self.shape.row_ids(axis)[indexes])``.
  axis:
    The axis to be indexed. Must satisfy ``0 <= axis < self.num_axes``.
  need_value_indexes:
    If ``True``, it will return a torch.Tensor containing the indexes into
    ``self.data`` that ``ans.data`` has, as in
    ``ans.data = self.data[value_indexes]``.

Returns:
  Return a tuple containing:
   - A ragged tensor, sharing the same dtype and device with ``self``
   - ``None`` if ``need_value_indexes`` is False; a 1-D torch.tensor of
     dtype ``torch.int32`` containing the indexes into ``self.data`` that
     ``ans.data`` has.
)doc";

static constexpr const char *kRaggedAnyIndexTensorWithRaggedDoc = R"doc(
Use a ragged tensor to index a 1-d torch tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> i = k2r.RaggedTensor([ [1, 5, 3], [0, 2] ])
>>> src = torch.arange(6, dtype=torch.int32) * 10
>>> src
tensor([ 0, 10, 20, 30, 40, 50], dtype=torch.int32)
>>> k2r.index(src, i)
[ [ 10 50 30 ] [ 0 20 ] ]
>>> k = k2r.RaggedTensor([ [[1, 5, 3], [0]], [[0, 2], [1, 3]] ])
>>> k2r.index(src, k)
[ [ [ 10 50 30 ] [ 0 ] ] [ [ 0 20 ] [ 10 30 ] ] ]
>>> n = k2r.RaggedTensor([ [1, -1], [-1, 0], [-1] ])
>>> k2r.index(src, n)
[ [ 10 0 ] [ 0 0 ] [ 0 ] ]
>>> k2r.index(src, n, default_value=-2)
[ [ 10 -2 ] [ -2 0 ] [ -2 ] ]

Args:
  src:
    A 1-D torch tensor.
  indexes:
    A ragged tensor with dtype ``torch.int32``.
  default_value:
    Used only when an entry in ``indexes`` is -1, in which case
    it returns ``default_value`` as -1 is not a valid index.
    If it is ``None`` and an entry in ``indexes`` is -1, 0 is returned.

Return:
  Return a ragged tensor with the same dtype and device as ``src``.
)doc";

static constexpr const char *kRaggedAnyIndexAndSumDoc = R"doc(
Index a 1-D tensor with a ragged tensor of indexes, perform
a sum-per-sublist operation, and return the resulting 1-D tensor.

>>> import torch
>>> import k2.ragged as k2r
>>> i = k2r.RaggedTensor([[1, 3, 5], [0, 2, 3]])
>>> src = torch.arange(6, dtype=torch.float32) * 10
>>> src
tensor([ 0., 10., 20., 30., 40., 50.])
>>> k2r.index_and_sum(src, i)
tensor([90., 50.])
>>> k = k2r.RaggedTensor([[1, -1, 2], [-1], [2, 5, -1]])
>>> k2r.index_and_sum(src, k)
tensor([30.,  0., 70.])
>>> k2r.index_and_sum(src, k, default_value=8)
tensor([38.,  8., 78.])

Args:
  src:
    A 1-D tensor.
  indexes:
    A ragged tensor with two axes. Its dtype MUST be ``torch.int32``.
  default_value:
    Used only when an entry in ``indexes`` is -1, in which case
    it returns ``default_value`` as -1 is not a valid index.
    If it is ``None`` and an entry in ``indexes`` is -1, 0 is returned.
Returns:
  Return a 1-D tensor with the same dtype and device as ``src``.
)doc";

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_DOC_ANY_H_
