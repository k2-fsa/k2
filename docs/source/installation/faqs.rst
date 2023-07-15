FAQs
====

ImportError: /lib64/libm.so.6: version 'GLIBC_2.27' not found
-------------------------------------------------------------

If you have this issue, it is very likely that you used ``conda install``
to install k2.

The conda package is compiled using Ubuntu 18.04. If your system
is too old, you will get such an issue.

The most straightforward way to fix it is to :ref:`install k2 from source`.

error: Downloading https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz failed
----------------------------------------------------------------------------------

If you have no access to the Internet, you will get such an issue when installing
k2 from source.

Please try to find a machine that has access to the Internet, download pybind11,
and copy the downloaded file to the machine you use to compile k2.

The commands you need are as follows:

.. code-block:: bash

   # assume that you place the downloaded file in the /tmp directory
   cd /tmp
   wget https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz

and then change `<https://github.com/k2-fsa/k2/blob/master/cmake/pybind11.cmake#L23>`_:

.. code-block::

   # set(pybind11_URL  "https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz")
   set(pybind11_URL  "file:///tmp/v2.6.0.tar.gz")

MKL related issues on macOS
---------------------------

If you have the following error while importing ``k2``:

.. code-block:: bash

  $ python3 -c "import k2"
  Traceback (most recent call last):
    File "/Users/fangjun/software/miniconda3/envs/tt/lib/python3.8/site-packages/k2/__init__.py", line 24, in <module>
      from _k2 import DeterminizeWeightPushingType
  ImportError: dlopen(/Users/fangjun/software/miniconda3/envs/tt/lib/python3.8/site-packages/_k2.cpython-38-darwin.so, 2): Library not loaded: @rpath/libmkl_intel_ilp64.2.dylib
    Referenced from: /Users/fangjun/software/miniconda3/envs/tt/lib/python3.8/site-packages/_k2.cpython-38-darwin.so
    Reason: image not found


You can use the following commands to fix it:

.. code-block:: bash

   $ cd $CONDA_PREFIX/lib
   $ ls -lh libmkl_intel_ilp64.*

It will show something like below:

.. code-block:: bash

  $ ls -lh libmkl_intel_ilp64.*
  -rwxrwxr-x  2 fangjun  staff    19M Oct 18  2021 libmkl_intel_ilp64.1.dylib
  -rwxrwxr-x  2 fangjun  staff    19M Oct 18  2021 libmkl_intel_ilp64.dylib

The fix is to create a symlink inside ``$CONDA_PREFIX/lib``:

.. code-block:: bash

  $ ln -s libmkl_intel_ilp64.dylib libmkl_intel_ilp64.2.dylib

After the above fix, you may get a different error like below:

.. code-block:: bash

  $ python3 -c "import k2"
  Traceback (most recent call last):
    File "/Users/fangjun/software/miniconda3/envs/tt/lib/python3.8/site-packages/k2/__init__.py", line 24, in <module>
      from _k2 import DeterminizeWeightPushingType
  ImportError: dlopen(/Users/fangjun/software/miniconda3/envs/tt/lib/python3.8/site-packages/_k2.cpython-38-darwin.so, 2): Library not loaded: @rpath/libmkl_core.2.dylib
    Referenced from: /Users/fangjun/software/miniconda3/envs/tt/lib/python3.8/site-packages/_k2.cpython-38-darwin.so
    Reason: image not found

Please follow the above approach to create another symlink for ``libmkl_core.2.dylib``.

In summary, the commands you need to fix mkl related issues are listed below:

.. code-block:: bash

  $ cd $CONDA_PREFIX/lib
  $ ln -s libmkl_intel_ilp64.dylib libmkl_intel_ilp64.2.dylib
  $ ln -s libmkl_core.dylib libmkl_core.2.dylib
  $ ln -s libmkl_intel_thread.dylib libmkl_intel_thread.2.dylib

Error: invalid device function
------------------------------

If you get the following error while running k2:

.. code-block::

  [F] /xxx/k2/k2-latest/k2/csrc/eval.h:147:void k2::EvalDevice(cudaStream_t,
  int32_t, LambdaT&) [with LambdaT = __nv_dl_wrapper_t<__nv_dl_tag<k2::Array1<int>
  (*)(std::shared_ptr<k2::Context>, int, int, int), k2::Range<int>, 1>, int*,
  int, int>; cudaStream_t = CUstream_st*; int32_t = int] Check failed:
  e == cudaSuccess (98 vs. 0)  Error: invalid device function.

you have probably installed k2 from source. However, you are ``NOT`` running k2 on
the same machine as the one you used to build k2 and the two machines have different
types of GPUs.

The fix is to pass ``-DK2_BUILD_FOR_ALL_ARCHS=ON`` to ``cmake``.

If you have followed :ref:`installation for developers` to install k2, you need
to change

.. code-block:: bash

  cmake -DCMAKE_BUILD_TYPE=Release ..

to

.. code-block:: bash

  cmake -DCMAKE_BUILD_TYPE=Release -DK2_BUILD_FOR_ALL_ARCHS=ON ..

If you have followed :ref:`install k2 from source` to install k2, you need to
change

.. code-block:: bash

  git clone https://github.com/k2-fsa/k2.git
  cd k2
  python3 setup.py install

to

.. code-block:: bash

  git clone https://github.com/k2-fsa/k2.git
  cd k2
  export K2_CMAKE_ARGS="-DK2_BUILD_FOR_ALL_ARCHS=ON"
  python3 setup.py install

#error C++14 or later compatible compiler is required to use ATen
-----------------------------------------------------------------

If you encounter this error, it means you need to update your GCC.


What if it throws the same error after updating GCC?

The solution is to setup the following environment variables
after updating GCC:

.. code-block:: bash

  # Please change gcc_dir accordingly
  gcc_dir=/ceph-fj/fangjun/software/gcc-12.2.0

  export CC=$gcc_dir/bin/gcc
  export CXX=$gcc_dir/bin/g++
  export LIBRARY_PATH=$gcc_dir/lib64:$LIBRARY_PATH
  export LD_LIBRARY_PATH=$gcc_dir/lib64:$LD_LIBRARY_PATH
  export C_INCLUDE_PATH=$gcc_dir/include
  export CPLUS_INCLUDE_PATH=$gcc_dir/include

.. caution::

  After setting the above environment variables, please run:

    .. code-block:: bash

      rm -rf

  before you start to re-build ``k2``.

relocation R_X86_64_PC32 against symbol `_PyRuntime' can not be used when making a shared object; recompile with -fPIC
-----------------------------------------------------------------------------------------------------------------------

The error indicates that your Python is statically linked. Please pass ``--enable-shared``
to ``./configure`` when you compile Python from source.

See also `<https://github.com/k2-fsa/k2/issues/1225>`_

ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
-------------------------------------------------------------------------------------------

Please see `<https://github.com/k2-fsa/k2/issues/1011>`_. You need to set the
environment variable ``LD_LIBRARY_PATH``.

/usr/bin/ld: cannot find -lPYTHON_LIBRARY-NOTFOUND
--------------------------------------------------

Please see `<https://github.com/k2-fsa/k2/issues/1038>`.


