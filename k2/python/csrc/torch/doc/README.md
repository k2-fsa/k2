## Introduction

This folder contains documentations for the k2's python wrapper.

**CAUTION**: Format of the documentation has to follow the one
described in <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>.
We choose the **Google style** docstring.

**CAUTION**: Indentation matters.

**Incorrect** indentation:

```cpp
static constexpr const char *kRaggedAnyInitDataDoc = R"doc(Create a ragged
tensor with two axes.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([ [1, 2], [5], [], [9] ])
)doc";
```


**Correct** indentation:

```cpp
static constexpr const char *kRaggedAnyInitDataDoc = R"doc(
Create a ragged tensor with two axes.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.Tensor([ [1, 2], [5], [], [9] ])
)doc";
```
