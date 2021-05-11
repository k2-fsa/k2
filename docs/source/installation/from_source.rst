.. _install k2 from source:

Install from source
===================

The following versions of Python, CUDA, and PyTorch are known to work.

    |source_python_versions| |source_cuda_versions| |source_pytorch_versions|

.. |source_python_versions| image:: ./images/source_python-3.6_3.7_3.8_3.9-blue.svg
  :alt: Supported python versions

.. |source_cuda_versions| image:: ./images/source_cuda-10.1_10.2_11.0_11.1-orange.svg
  :alt: Supported cuda versions

.. |source_pytorch_versions| image:: ./images/source_pytorch-1.6.0_1.7.0_1.7.1_1.8.0_1.8.1-green.svg
  :alt: Supported pytorch versions

Before compiling k2, some preparation work has to be done:

- Have a compiler supporting at least C++14, e.g., GCC >= 5.0, Clang >= 3.4.
- Install CMake. CMake 3.11.0 and 3.18.0 are known to work.
- Install Python3.
- Install PyTorch.
- Install CUDA toolkit.
- Install cuDNN. Please install a version that is compatible with the
  CUDA toolkit you are using.

.. NOTE::

  We need NVCC to build k2, if you use conda to install CUDA toolkit,
  you may need to install `nvcc_linux-64` or `cudatoolkit-dev` as well since the
  default installation of CUDA toolkit in conda did not include NVCC.
  However, `nvcc_linux-64` or `cudatoolkit-dev` may not work well on all platforms,
  so it's better if you can install CUDA toolkit using a normal way instead of
  using conda if you want to build k2 from source.)

After setting up the environment, we are ready to build k2::

  git clone https://github.com/k2-fsa/k2.git
  cd k2
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  # If you installed cudatoolkit using conda install -y -c nvidia cudatoolkit=X cudnn=Y,
  # source the conda environemt and change the cmake command to:
  # cmake -DCUDNN_LIBRARY_PATH=$(find $CONDA_PREFIX -name libcudnn.so) -DCUDNN_INCLUDE_PATH=$CONDA_PREFIX/include/ -DCMAKE_BUILD_TYPE=Release ..
  make _k2
  cd ..
  pip3 install wheel twine
  ./scripts/build_pip.sh

  # Have a look at the `dist/` directory.

You will find the wheel file in the `dist` directory, e.g.,
`dist/k2-0.1.1.dev20201125-cp38-cp38-linux_x86_64.whl`, which
can be installed with::

  pip install dist/k2-0.1.1.dev20201125-cp38-cp38-linux_x86_64.whl

.. HINT::

  You may get a wheel with a different filename.

.. Note::

  [For developers]

  If you are developing k2, you don't need to install k2 to use its Python APIs!
  All you need to do is to setup the ``PYTHONPATH`` environment variable so that
  Python can find where k2 resides.

  k2 contains two parts. The first part consists of pure Python files that are in
  ``k2_source_tree/k2/python/k2``. The second part is the C++ part, which has been
  compiled into a bunch of shared libraries that can be invoked from Python. These
  libraries are saved in `k2_build_tree/build/lib`.

  If you set ``PYTHONPATH`` to the following values:

  .. code-block::

    export PYTHONPATH=/path/to/k2/k2/python:$PYTHONPATH
    export PYTHONPATH=/path/to/k2/build/lib:$PYTHONPATH

  After ``PYTHONPATH`` is set, you can run:

  .. code-block::

    python3 -m k2.version
    python3 -c 'import k2; print(k2.__file__)'
    python3 -c 'import _k2; print(_k2.__file__)'

  Whenver you change some files in ``k2/python/k2``, it comes into effect immediately
  without uninstalling and installing operations.

  Whenever you modify some `*.cu` files, it is also available after issuing ``make _k2``
  without any installation effort.


To run tests, you have to install the following requirements first::

  sudo apt-get install graphviz
  cd k2
  pip3 install -r ./requirements.txt

You can run tests with::

  cd build
  make -j
  make test

To run tests in parallel::

  cd build
  make -j
  ctest --output-on-failure --parallel <JOBNUM>

If `valgrind` is installed, you can check heap corruptions and memory leaks by::

  cd build
  make -j
  ctest --output-on-failure -R <TESTNAME> -D ExperimentalMemCheck

.. HINT::

  You can install `valgrind` with `sudo apt-get install valgrind`
  on Ubuntu.
