_k2
===

If you have used PyTorch, you have probably seen ``torch._C``

.. code-block:: python

  >>> import torch
  >>> import torch._C
  >>> print(torch._C.__file__)
  /star-fj/fangjun/py38/lib/python3.8/site-packages/torch/_C.cpython-38-x86_64-linux-gnu.so

Similarly, we have ``_k2`` in `k2`_:

.. code-block:: bash

  >>> import torch
  >>> import _k2
  >>> print(_k2.__file__)
  /root/fangjun/open-source/k2/build-cpu-debug/lib/_k2.cpython-38-x86_64-linux-gnu.so

You can see that both ``_C`` and ``_k2`` are contained in a shared library because
they are implemented in C++.

.. hint::

   In case you are interested in how to wrap C++ code to Python, please have
   a look at `pybind11`_.

.. toctree::
   :maxdepth: 2

   ./how-to-debug.rst
