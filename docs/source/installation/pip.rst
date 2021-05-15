Install using pip (k2-fsa.org)
==============================

.. |pip_python_versions| image:: ./images/pip_python-3.6_3.7_3.8-blue.svg
  :alt: Supported python versions

.. |pip_cuda_versions| image:: ./images/pip_cuda-10.1_10.2_11.0-orange.svg
  :alt: Supported cuda versions

.. |pip_pytorch_versions| image:: ./images/pip_pytorch-1.7.1-green.svg
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

The following commands install k2 with different CUDA versions:

.. code-block:: bash

  # Install k2 0.3.3 with CUDA 10.2 built on 20210509
  #
  # cu102 means CUDA 10.2
  #
  pip install k2==0.3.3+cu102.dev20210509 -f https://k2-fsa.org/nightly/

  # Install k2 0.3.3 with CUDA 11.0 built on 20210509
  #
  # cu110 means CUDA 11.0
  #
  pip install k2==0.3.3+cu110.dev20210509 -f https://k2-fsa.org/nightly/

  # Install k2 0.3.3 with CUDA 10.1 built on 20210509
  #
  # CAUTION: you don't need to specify cu101 since CUDA 10.1 is the default
  # CUDA version
  #
  pip install k2==0.3.3.dev20210509 -f https://k2-fsa.org/nightly/

  #
  # dev20210509 means that version is built on 2021.05.09
  #
  # Please always select the latest version. That is, the version
  # with the latest date.

The following is the log for installing k2:

.. code-block::

  $ pip install k2==0.3.3.dev20210509 -f https://k2-fsa.org/nightly/
  Looking in links: https://k2-fsa.org/nightly/
  Collecting k2==0.3.3.dev20210509
    Downloading https://k2-fsa.org/nightly/whl/k2-0.3.3.dev20210509-cp38-cp38-linux_x86_64.whl (54.4 MB)
       |________________________________| 54.4 MB 487 kB/s
  Requirement already satisfied: torch in ./py38/lib/python3.8/site-packages (from k2==0.3.3.dev20210509) (1.7.1+cu101)
  Requirement already satisfied: graphviz in ./py38/lib/python3.8/site-packages (from k2==0.3.3.dev20210509) (0.15)
  Requirement already satisfied: numpy in ./py38/lib/python3.8/site-packages (from torch->k2==0.3.3.dev20210509) (1.19.5)
  Requirement already satisfied: typing-extensions in ./py38/lib/python3.8/site-packages (from torch->k2==0.3.3.dev20210509) (3.7.4.3)
  Installing collected packages: k2
  Successfully installed k2-0.3.3.dev20210509
  WARNING: You are using pip version 21.0.1; however, version 21.1.1 is available.
  You should consider upgrading via the '/xxx/bin/python3.8 -m pip install --upgrade pip' command.

To verify that k2 is installed successfully, run:

.. code-block::

  $ python3 -m k2.version
  /xxx/lib/python3.8/runpy.py:127: RuntimeWarning: 'k2.version' found in sys.modules after import of package 'k2', but prior to execution of 'k2.version'; this may result in unpredictable behaviour
    warn(RuntimeWarning(msg))
  Collecting environment information...

  k2 version: 0.3.3
  Build type: Release
  Git SHA1: 8e2fa82dca767782351fec57ec187aa04015dcf2
  Git date: Thu May 6 18:55:15 2021
  Cuda used to build k2: 10.1
  cuDNN used to build k2: 8.0.2
  Python version used to build k2: 3.8
  OS used to build k2: Ubuntu 18.04.5 LTS
  CMake version: 3.20.2
  GCC version: 7.5.0
  CMAKE_CUDA_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0  --expt-extended-lambda -gencode arch=compute_35,code=sm_35 --expt-extended-lambda -gencode arch=compute_50,code=sm_50 --expt-extended-lambda -gencode arch=compute_60,code=sm_60 --expt-extended-lambda -gencode arch=compute_61,code=sm_61 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 --expt-extended-lambda -gencode arch=compute_75,code=sm_75 --compiler-options -Wall --compiler-options -Wno-unknown-pragmas
  CMAKE_CXX_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0
  PyTorch version used to build k2: 1.7.1+cu101
  PyTorch is using Cuda: 10.1
  NVTX enabled: True
  Disable debug: True
  Sync kernels : False
  Disable checks: False

Congratulations! You have installed k2 successfully.
