FAQs
====

ImportError: /lib64/libm.so.6: version `GLIBC_2.27' not found
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
