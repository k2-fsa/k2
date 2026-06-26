.. _install k2 from source:

Install from source
===================

.. caution::

   Users who have issues about installing `k2`_ from source are mostly installing
   PyTorch with ``conda install``.

   We suggest that you install ``PyTorch`` using ``pip install``.

.. hint::

    It supports Linux (CPU + CUDA + ROCm), macOS (CPU), and Windows (CPU + CUDA
    + ROCm). For AMD GPUs via ROCm, see the "Building with ROCm (AMD GPUs)"
    section below.

.. hint::

  You can pass the option ``-DK2_WITH_CUDA=OFF`` to ``cmake`` to build
  a CPU only version of k2. In that case, you have to install a CPU version
  of PyTorch; otherwise, you will get a CMake configuration error, saying
  that cuDNN cannot be found.

Before compiling k2, some preparation work has to be done:

  - Have a compiler supporting at least C++14, e.g., GCC >= 7.0, Clang >= 3.4.
  - Install CMake. CMake 3.11.0 and 3.18.0 are known to work.
  - Install Python3. Please pass ``--enabled-shared`` to ``./configure`` if you install
    Python from source.
  - Install PyTorch.
  - Install CUDA toolkit and cuDNN. Please refer to :ref:`cuda_and_cudnn`.

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


Building with ROCm (AMD GPUs)
-----------------------------

k2 can also be built for AMD GPUs with ROCm/HIP. The ``.cu`` sources are
compiled as HIP and the GPU primitives are provided by hipCUB and rocThrust.

.. hint::

  This builds the Python ``_k2`` extension module and the C++ gtest suite (the
  FSA core that `icefall <https://github.com/k2-fsa/icefall>`_ and
  `sherpa <https://github.com/k2-fsa/sherpa>`_ consume). The standalone
  ``k2/torch`` C++ decoder layer is not yet built on the ROCm path.

Before compiling, prepare the environment:

  - Install ROCm (7.2 or newer) including hipCUB, rocPRIM, hipRAND and rocThrust.
  - Install a ROCm build of PyTorch.
  - libcu++ is not shipped by ROCm; vendor the ROCm fork
    (``git clone --branch amd-develop https://github.com/ROCm/libhipcxx``) and
    point ``K2_LIBHIPCXX_INCLUDE_DIR`` at its ``include`` directory.

Then configure and build, selecting your GPU architecture(s) with
``CMAKE_HIP_ARCHITECTURES`` (e.g. ``gfx90a`` for MI200, ``gfx1100`` for RDNA3;
pass a semicolon-separated list to target several). When unset it defaults to
``gfx90a``.

.. code-block:: bash

  git clone https://github.com/k2-fsa/k2.git
  cd k2
  mkdir build_rocm
  cd build_rocm
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DCMAKE_CXX_STANDARD=20 \
        -DK2_LIBHIPCXX_INCLUDE_DIR=/path/to/libhipcxx/include \
        -DK2_ENABLE_TESTS=ON \
        ..
  make -j

.. hint::

  To build and install the Python package with ROCm, pass the same options
  through ``K2_CMAKE_ARGS``:

  .. code-block:: bash

    export K2_CMAKE_ARGS="-DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx90a"
    python3 setup.py install

.. hint::

  Run the GPU tests on a single device, serially, by setting
  ``HIP_VISIBLE_DEVICES`` to one GPU and running the ``cu_*_test`` executables
  (or ``ctest``) from the build directory.

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
