.. _install k2 from source:

Install from source
===================

The following versions of Python, CUDA, and PyTorch are known to work.

    - |source_python_versions|
    - |source_cuda_versions|
    - |source_pytorch_versions|

.. |source_python_versions| image:: ./images/source_python-3.6_3.7_3.8_3.9-blue.svg
  :alt: Supported python versions

.. |source_cuda_versions| image:: ./images/source_cuda-10.1_10.2_11.0_11.1_11.2_11.3-orange.svg
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

After setting up the environment, we are ready to build k2:

.. code-block:: bash

  git clone https://github.com/k2-fsa/k2.git
  cd k2
  python3 setup.py install

That is all you need to run.

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
