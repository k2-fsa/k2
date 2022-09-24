Install using pip (pypi.org)
============================

.. |pypi_python_versions| image:: ./images/python_ge_3.7-blue.svg
  :alt: Supported python versions

.. |pypi_cuda_versions| image:: ./images/cuda-10.2-orange.svg
  :alt: Supported cuda versions

.. |pypi_pytorch_versions| image:: ./images/torch-1.12.1-green.svg
  :alt: Supported pytorch versions

``k2`` on PyPI supports the following versions of Python, CUDA, and PyTorch:

  - |pypi_python_versions|
  - |pypi_cuda_versions|
  - |pypi_pytorch_versions|

.. caution::

  We assume that you have installed cudatoolkit.
  If not, please install them before proceeding.

.. hint::

  See also :ref:`install using conda`. It installs all the dependencies for you
  automagically. You don't need to pre-install PyTorch and cudatoolkit when using
  ``conda install``.

The following command installs ``k2`` from PyPI:

.. code-block:: bash

  pip install k2

.. Caution::

  The wheel packages on PyPI are built using `torch==1.12.1+cu102` on Ubuntu 18.04.
  If you are using other Linux systems or a different PyTorch version, the
  pre-built wheel packages may NOT work on your system, please consider one of
  the following alternatives to install k2:

      - :ref:`install using conda`
      - :ref:`install k2 from source`

To verify that ``k2`` is installed successfully, run:

.. code-block::

  $ python3 -m k2.version

You should see something like below:

.. code-block::

  Collecting environment information...

  k2 version: 1.20
  Build type: Release
  Git SHA1: 89465dfc3085c1e7148f8c5e78a861c42ad77730
  Git date: Wed Sep 21 14:18:54 2022
  Cuda used to build k2: 10.2
  cuDNN used to build k2: 8.3.2
  Python version used to build k2: 3.8
  OS used to build k2: Ubuntu 18.04.5 LTS
  CMake version: 3.21.6
  GCC version: 7.5.0
  CMAKE_CUDA_FLAGS:   -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_35,code=sm_35  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_50,code=sm_50  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_60,code=sm_60  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_61,code=sm_61  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_70,code=sm_70  -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_75,code=sm_75 -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=integer_sign_change,--diag_suppress=useless_using_declaration,--diag_suppress=set_but_not_used,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=unsigned_compare_with_zero,--diag_suppress=declared_but_not_referenced,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -D_GLIBCXX_USE_CXX11_ABI=0 --compiler-options -Wall  --compiler-options -Wno-strict-overflow  --compiler-options -Wno-unknown-pragmas 
  CMAKE_CXX_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-unused-variable  -Wno-strict-overflow 
  PyTorch version used to build k2: 1.12.1+cu102
  PyTorch is using Cuda: 10.2
  NVTX enabled: True
  With CUDA: True
  Disable debug: True
  Sync kernels : False
  Disable checks: False
  Max cpu memory allocate: 214748364800 bytes (or 200.0 GB)
  k2 abort: False
  __file__: /ceph-fj/fangjun/py38-1.12.1/lib/python3.8/site-packages/k2/version/version.py
  _k2.__file__: /ceph-fj/fangjun/py38-1.12.1/lib/python3.8/site-packages/_k2.cpython-38-x86_64-linux-gnu.so


Congratulations! You have installed ``k2`` successfully.
