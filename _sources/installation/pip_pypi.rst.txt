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

  pip install --pre k2

The wheel packages on PyPI are built using `torch==1.7.1+cu101` on Ubuntu 18.04.
If you are using other Linux systems or a different PyTorch version,
the pre-built wheel packages may NOT work on your system, please install
k2 from source in this case.

.. CAUTION::

    k2 is still under active development and we are trying to keep
    the packages on PyPI up to date. Please use ``--pre`` in ``pip install``.

    If you want to try the latest version, please refer to
    :ref:`install k2 from source`.

To verify that k2 is installed successfully, run:

.. code-block::

  $ python3 -m k2.version
  /xxx/lib/python3.8/runpy.py:127: RuntimeWarning: 'k2.version' found in sys.modules after import of package 'k2', but prior to execution of 'k2.version'; this may result in unpredictable behaviour
    warn(RuntimeWarning(msg))
  Collecting environment information...

  k2 version: 0.3.3
  Build type: Release
  Git SHA1: d66cad5067563bb87710a40cf401af35cae816ff
  Git date: Fri Apr 30 13:33:47 2021
  Cuda used to build k2: 10.1
  cuDNN used to build k2: 8.0.2
  Python version used to build k2: 3.8
  OS used to build k2: Ubuntu 18.04.5 LTS
  CMake version: 3.20.1
  GCC version: 5.5.0
  CMAKE_CUDA_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0  --expt-extended-lambda -gencode arch=compute_35,code=sm_35 --expt-extended-lambda -gencode arch=compute_50,code=sm_50 --expt-extended-lambda -gencode arch=compute_60,code=sm_60 --expt-extended-lambda -gencode arch=compute_61,code=sm_61 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 --expt-extended-lambda -gencode arch=compute_75,code=sm_75 --compiler-options -Wall --compiler-options -Wno-unknown-pragmas
  CMAKE_CXX_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0
  PyTorch version used to build k2: 1.7.1+cu101
  PyTorch is using Cuda: 10.1
  NVTX enabled: True
  Disable debug: True
  Sync kernels : False
  Disable checks: False

Congratulations! You have installed k2 successfully.
