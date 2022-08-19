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
