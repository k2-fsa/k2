.. _install using pip via k2-fsa.org:

Install using pip (k2-fsa.org)
==============================

.. |pip_python_versions| image:: ./images/python_ge_3.6-blue.svg
  :alt: Supported python versions

.. |pip_cuda_versions| image:: ./images/cuda_ge_10.1-orange.svg
  :alt: Supported cuda versions

.. |pip_pytorch_versions| image:: ./images/pytorch_ge_1.6.0-green.svg
  :alt: Supported pytorch versions

You can find a list of nightly pre-built
wheel packages at `<https://k2-fsa.org/nightly/index.html>`_ with the following
versions of Python, CUDA, and PyTorch.

  - |pip_python_versions|
  - |pip_cuda_versions|
  - |pip_pytorch_versions|

.. caution::

  We assume that you have installed cudatoolkit.
  If not, please install them before proceeding.

.. hint::

  See also :ref:`install using conda`. It installs all the dependencies for you
  automagically. You don't need to pre-install PyTorch and cudatoolkit when using
  ``conda install``.

The following commands install k2 with different versions of CUDA and PyTorch:

.. code-block:: bash

  # Install k2 1.8 with CUDA 10.1 built on 20210916
  #
  # You don't need to specifiy the Python version
  #
  pip install k2==1.8.dev20210916+cuda10.1.torch1.7.1 -f https://k2-fsa.org/nightly/

  # Install k2 1.8 with CUDA 10.2 built on 20210916
  #
  #
  pip install k2==1.8.dev20210916+cuda10.2.torch1.7.1 -f https://k2-fsa.org/nightly/

  # Install k2 1.8 with CUDA 11.0 built on 20210916
  #
  pip install k2==1.8.dev20210916+cuda11.0.torch1.7.1 -f https://k2-fsa.org/nightly/

  # Please always select the latest version. That is, the version
  # with the latest date.

To install a version for CPU only, please use:

.. code-block:: bash

  # Install a CPU version compiled against PyTorch 1.8.1 on 2021.10.22
  #
  pip install k2==1.9.dev20211022+cpu.torch1.8.1 -f https://k2-fsa.org/nightly/

  # Install a CPU version compiled against PyTorch 1.9.0 on 2021.10.22
  #
  pip install k2==1.9.dev20211022+cpu.torch1.9.0 -f https://k2-fsa.org/nightly/

  # Please visit https://k2-fsa.org/nightly/ for more versions of k2

.. Caution::

  We only provide pre-compiled versions of k2 with torch 1.7.1. If you need
  other versions of PyTorch, please consider one of the following alternatives
  to install k2:

    - :ref:`install using conda`
    - :ref:`install k2 from source`

The following is the log for installing k2:

.. code-block::

  $ pip install k2==1.8.dev20210916+cuda10.1.torch1.7.1 -f https://k2-fsa.org/nightly

  Looking in links: https://k2-fsa.org/nightly
  Collecting k2==1.8.dev20210916+cuda10.1.torch1.7.1
    Downloading https://k2-fsa.org/nightly/whl/k2-1.8.dev20210916%2Bcuda10.1.torch1.7.1-cp38-cp38-linux_x86_64.whl (77.7 MB)
       |________________________________| 77.7 MB 1.6 MB/s
  Collecting torch==1.7.1
    Using cached torch-1.7.1-cp38-cp38-manylinux1_x86_64.whl (776.8 MB)
  Collecting graphviz
    Using cached graphviz-0.17-py3-none-any.whl (18 kB)
  Collecting typing-extensions
    Downloading typing_extensions-3.10.0.2-py3-none-any.whl (26 kB)
  Collecting numpy
    Using cached numpy-1.21.2-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.8 MB)
  Installing collected packages: typing-extensions, numpy, torch, graphviz, k2
  Successfully installed graphviz-0.17 k2-1.8.dev20210916+cuda10.1.torch1.7.1 numpy-1.21.2 torch-1.7.1 typing-extensions-3.10.0.2

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
