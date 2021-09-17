Install using pip (pypi.org)
============================

.. |pypi_python_versions| image:: ./images/pypi_python-3.6_3.7_3.8-blue.svg
  :alt: Supported python versions

.. |pypi_cuda_versions| image:: ./images/pypi_cuda-10.1-orange.svg
  :alt: Supported cuda versions

.. |pypi_pytorch_versions| image:: ./images/pypi_pytorch-1.7.1-green.svg
  :alt: Supported pytorch versions

k2 on PyPI supports the following versions of Python, CUDA, and PyTorch:

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

The following command installs k2 from PyPI:

.. code-block:: bash

  pip install k2

.. Caution::

  The wheel packages on PyPI are built using `torch==1.7.1+cu101` on Ubuntu 18.04.
  If you are using other Linux systems or a different PyTorch version, the
  pre-built wheel packages may NOT work on your system, please consider one of
  the following alternatives to install k2:

      - :ref:`install using conda`
      - :ref:`install k2 from source`

To verify that k2 is installed successfully, run:

.. code-block::

  $ python3 -m k2.version

  k2 version: 1.8
  Build type: Release
  Git SHA1: 646704e142438bcd1aaf4a6e32d95e5ccd93a174
  Git date: Thu Sep 16 13:05:12 2021
  Cuda used to build k2: 10.1
  cuDNN used to build k2: 8.0.2
  Python version used to build k2: 3.8
  OS used to build k2: Ubuntu 18.04.5 LTS
  CMake version: 3.21.2
  GCC version: 7.5.0
  CMAKE_CUDA_FLAGS:  --expt-extended-lambda -gencode arch=compute_35,code=sm_35 --expt-extended-lambda -gencode arch=compute_50,code=sm_50 --expt-extended-lambda -gencode arch=compute_60,code=sm_60 --expt-extended-lambda -gencode arch=compute_61,code=sm_61 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 --expt-extended-lambda -gencode arch=compute_75,code=sm_75 -D_GLIBCXX_USE_CXX11_ABI=0 --compiler-options -Wall --compiler-options -Wno-unknown-pragmas --compiler-options -Wno-strict-overflow
  CMAKE_CXX_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-strict-overflow
  PyTorch version used to build k2: 1.7.1+cu101
  PyTorch is using Cuda: 10.1
  NVTX enabled: True
  With CUDA: True
  Disable debug: True
  Sync kernels : False
  Disable checks: False

Congratulations! You have installed k2 successfully.
