.. _install k2 from source:

Install from source
===================

.. hint::

    It supports Linux (CPU + CUDA), macOS (CPU), and Windows (CPU + CUDA).

.. hint::

  You can pass the option ``-DK2_WITH_CUDA=OFF`` to ``cmake`` to build
  a CPU only version of k2. In that case, you have to install a CPU version
  of PyTorch; otherwise, you will get a CMake configuration error, saying
  that cuDNN cannot be found.

The following versions of Python, CUDA, and PyTorch are known to work.

    - |source_python_versions|
    - |source_cuda_versions|
    - |source_pytorch_versions|

.. |source_python_versions| image:: ./images/python_ge_3.6-blue.svg
  :alt: Supported python versions

.. |source_cuda_versions| image:: ./images/cuda_ge_10.1-orange.svg
  :alt: Supported cuda versions

.. |source_pytorch_versions| image:: ./images/pytorch_ge_1.6.0-green.svg
  :alt: Supported pytorch versions

Before compiling k2, some preparation work has to be done:

  - Have a compiler supporting at least C++14, e.g., GCC >= 7.0, Clang >= 3.4.
  - Install CMake. CMake 3.11.0 and 3.18.0 are known to work.
  - Install Python3.
  - Install PyTorch.
  - Install CUDA toolkit and cuDNN.


.. hint::

  You can use ``pip install cmake`` to install the latest version of CMake.

.. caution::

  cudatoolkit installed by ``conda install`` cannot be used to compile ``k2``.

  Please follow :ref:`cuda_and_cudnn` to install cudatoolkit and cuDNN.

After setting up the environment, we are ready to build k2:

.. code-block:: bash

  git clone https://github.com/k2-fsa/k2.git
  cd k2
  export K2_MAKE_ARGS="-j6"
  python3 setup.py install

That is all you need to run.

.. hint::

   We use ``export K2_MAKE_ARGS="-j6"`` to pass ``-j6`` to ``make``
   to reduce compilation time.
   If you have many GPUs and enough RAM, you can choose a larger value.

.. caution::

   If you plan to run k2 on a different machine than the one you used to build
   k2 and the two machines have different types of GPUs, please use the
   following commands to install k2.

    .. code-block:: bash

      git clone https://github.com/k2-fsa/k2.git
      cd k2
      export K2_CMAKE_ARGS="-DK2_BUILD_FOR_ALL_ARCHS=ON"
      python3 setup.py install

  Otherwise, you may get some error like below when running k2:

    .. code-block::

      [F] /xxx/k2/k2-latest/k2/csrc/eval.h:147:void k2::EvalDevice(cudaStream_t,
      int32_t, LambdaT&) [with LambdaT = __nv_dl_wrapper_t<__nv_dl_tag<k2::Array1<int>
      (*)(std::shared_ptr<k2::Context>, int, int, int), k2::Range<int>, 1>, int*,
      int, int>; cudaStream_t = CUstream_st*; int32_t = int] Check failed:
      e == cudaSuccess (98 vs. 0)  Error: invalid device function.


To test that k2 is installed successfully, you can run:

.. code-block::

  $ python3
  Python 3.8.6 (default, Dec  2 2020, 15:56:31)
  [GCC 7.5.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import k2
  >>> s = '''
  ... 0 1 -1 0.1
  ... 1
  ... '''
  >>> fsa = k2.Fsa.from_str(s)
  >>> print(fsa)
  k2.Fsa: 0 1 -1 0.1
  1
  properties_str = "Valid|Nonempty|TopSorted|TopSortedAndAcyclic|ArcSorted|ArcSortedAndDeterministic|EpsilonFree|MaybeAccessible|MaybeCoaccessible".

To uninstall k2, run:

.. code-block::

  pip uninstall k2


Read more if you want to run the tests.

.. code-block::

  sudo apt-get install graphviz
  git clone https://github.com/k2-fsa/k2.git
  cd k2
  pip3 install -r ./requirements.txt
  mkdir build_release
  cd build_release
  cmake -DCMAKE_BUILD_TYPE=Release ..
  # If you installed cudatoolkit using conda install -y -c nvidia cudatoolkit=X cudnn=Y,
  # source the conda environemt and change the cmake command to:
  # cmake -DCUDNN_LIBRARY_PATH=$(find $CONDA_PREFIX -name libcudnn.so) -DCUDNN_INCLUDE_PATH=$CONDA_PREFIX/include/ -DCMAKE_BUILD_TYPE=Release ..
  make -j
  make test

To run tests in parallel::

  make -j
  ctest --output-on-failure --parallel <JOBNUM>
  # e.g., ctest --output-on-failure --parallel 5
