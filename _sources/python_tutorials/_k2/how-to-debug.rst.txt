How to debug _k2
================

This section introduces how to debug ``_k2``.


Code with bugs
--------------

Suppose you have the following ``buggy`` code,

.. literalinclude:: ./code/test.py
   :language: python
   :caption: Code with bugs

Error logs of the buggy code
----------------------------

After running it, you would get the following error logs:

.. literalinclude:: ./code/test.txt
   :caption: Error logs of the buggy code

Ways to debug
-------------

In order debug it, please first follow :ref:`build_a_debug_version` to build
a debug version of `k2`_ so that you can use ``gdb`` to debug the code.

.. note::

   Since the underlying implementation is in C++ you can you use ``gdb``
   to debug it, even if you are using Python.

First, let us use ``gdb`` to run our code:

.. code-block:: bash

  (py38) kuangfangjun:build-cpu-debug$ gdb --args python3 ./test.py

You will see the following output:

.. code-block:: bash

  GNU gdb (Ubuntu 8.1-0ubuntu3.2) 8.1.0.20180409-git
  Copyright (C) 2018 Free Software Foundation, Inc.
  License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
  This is free software: you are free to change and redistribute it.
  There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
  and "show warranty" for details.
  This GDB was configured as "x86_64-linux-gnu".
  Type "show configuration" for configuration details.
  For bug reporting instructions, please see:
  <http://www.gnu.org/software/gdb/bugs/>.
  Find the GDB manual and other documentation resources online at:
  <http://www.gnu.org/software/gdb/documentation/>.
  For help, type "help".
  Type "apropos word" to search for commands related to "word"...
  Reading symbols from python3...done.
  warning: File "/star-fj/fangjun/open-source/pyenv/versions/3.8.0/bin/python3.8-gdb.py" auto-loading has been declined by your `auto-load safe-path' set to "$debugdir:$datadir/auto-load".
  To enable execution of this file add
          add-auto-load-safe-path /star-fj/fangjun/open-source/pyenv/versions/3.8.0/bin/python3.8-gdb.py
  line to your configuration file "/root/fangjun/.gdbinit".
  To completely disable this security protection add
          set auto-load safe-path /
  line to your configuration file "/root/fangjun/.gdbinit".
  For more information about this security protection see the
  "Auto-loading safe path" section in the GDB manual.  E.g., run from the shell:
          info "(gdb)Auto-loading safe path"
  (gdb)

Next, enter ``catch throw`` so that it stops the process on exception:

.. code-block:: bash

  (gdb) catch throw
  Catchpoint 1 (throw)

Then we can run the program with ``run``:

.. code-block:: bash

  (gdb) run
  Starting program: /star-fj/fangjun/py38/bin/python3 ./test.py
  [Thread debugging using libthread_db enabled]
  Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
  [New Thread 0x7fff90f66700 (LWP 1184088)]
  [New Thread 0x7fff90765700 (LWP 1184089)]
  ... ...
  [F] /root/fangjun/open-source/k2/k2/csrc/array.h:179:k2::Array1<T> k2::Array1<T>::Arange(int32_t, int32_t) const [with T = int; int32_
  t = int] Check failed: end <= dim_ (32767 vs. 3)


  [ Stack-Trace: ]
  /root/fangjun/open-source/k2/build-cpu-debug/lib/libk2_log.so(k2::internal::GetStackTrace()+0x5b) [0x7fff2913c105]
  /root/fangjun/open-source/k2/build-cpu-debug/lib/_k2.cpython-38-x86_64-linux-gnu.so(+0xb52e2) [0x7fff29a512e2]
  ... ...
  /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7) [0x7ffff7458bf7]
  /star-fj/fangjun/py38/bin/python3(_start+0x2a) [0x55555555478a]


  Thread 1 "python3" hit Catchpoint 1 (exception thrown), 0x00007ffff2553d1d in __cxa_throw ()
     from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  (gdb)

We can find the backtrace with ``backtrace``:

.. code-block:: bash

  (gdb) backtrace
  #0  0x00007ffff2553d1d in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  #1  0x00007fff29a51365 in k2::internal::Logger::~Logger (this=0x7fffffffd5e0, __in_chrg=<optimized out>)
      at /root/fangjun/open-source/k2/k2/csrc/log.h:195
  #2  0x00007fff29b54786 in k2::Array1<int>::Arange (this=0x555558db63f8, start=3, end=32767)
      at /root/fangjun/open-source/k2/k2/csrc/array.h:179
  #3  0x00007fff29b86b1d in k2::<lambda(k2::RaggedAny&, int32_t)>::operator()(k2::RaggedAny &, int32_t) const (
      __closure=0x555558d973e8, self=..., i=2) at /root/fangjun/open-source/k2/build-cpu-debug/k2/python/csrc/torch/v2/any.cc:107
  #4  0x00007fff29b97f7f in pybind11::detail::argument_loader<k2::RaggedAny&, int>::call_impl<pybind11::object, k2::PybindRaggedAny(pybind11::module&)::<lambda(k2::RaggedAny&, int32_t)>&, 0, 1, pybind11::detail::void_type>(k2::<lambda(k2::RaggedAny&, int32_t)> &, std::index_sequence, pybind11::detail::void_type &&) (this=0x7fffffffd820, f=...)
      at /star-fj/fangjun/open-source/k2/build-cpu-debug/_deps/pybind11-src/include/pybind11/cast.h:1439
  #5  0x00007fff29b96291 in pybind11::detail::argument_loader<k2::RaggedAny&, int>::call<pybind11::object, pybind11::detail::void_type,
  k2::PybindRaggedAny(pybind11::module&)::<lambda(k2::RaggedAny&, int32_t)>&>(k2::<lambda(k2::RaggedAny&, int32_t)> &) (
      this=0x7fffffffd820, f=...) at /star-fj/fangjun/open-source/k2/build-cpu-debug/_deps/pybind11-src/include/pybind11/cast.h:1408
  #6  0x00007fff29b90b2f in pybind11::cpp_function::<lambda(pybind11::detail::function_call&)>::operator()(pybind11::detail::function_call &) const (__closure=0x0, call=...)
      at /root/fangjun/open-source/k2/build-cpu-debug/_deps/pybind11-src/include/pybind11/pybind11.h:249
  #7  0x00007fff29b90ba3 in pybind11::cpp_function::<lambda(pybind11::detail::function_call&)>::_FUN(pybind11::detail::function_call &)
      () at /root/fangjun/open-source/k2/build-cpu-debug/_deps/pybind11-src/include/pybind11/pybind11.h:224
  #8  0x00007fff29a2d6e0 in pybind11::cpp_function::dispatcher (self=0x7fff29fd4a50, args_in=0x7ffff64adb40, kwargs_in=0x0)
      at /root/fangjun/open-source/k2/build-cpu-debug/_deps/pybind11-src/include/pybind11/pybind11.h:929
  #9  0x00007ffff78ca535 in cfunction_call_varargs (kwargs=<optimized out>, args=<optimized out>, func=0x7fff29fd6310)
      at Objects/call.c:742


The remaining steps are the same for debugging a C/C++ program with ``gdb``.

Summary
-------

The most important parts of the debugging process are:

  - (1) Build a debug version of `k2`_
  - (2) Run ``catch throw``
  - (3) View the backtrace on exception

