Install using pip (pypi.org)
============================


``k2`` on PyPI supports the following versions of Python, CUDA, and PyTorch:

  - 3.7 <= Python <= 3.11
  - cuda == 11.7
  - torch == 1.13.1

.. caution::

  We assume that you have installed cudatoolkit.
  If not, please install them before proceeding.

.. hint::

  See also :ref:`install using conda`. It installs all the dependencies for you
  automagically. You don't need to pre-install PyTorch and cudatoolkit when using
  ``conda install``.

The following command installs ``k2`` from PyPI:

.. code-block:: bash

  pip install k2

.. Caution::

  The wheel packages on PyPI are built using `torch==1.13.1+cu117` on Ubuntu 20.04.
  If you are using other Linux systems or a different PyTorch version, the
  pre-built wheel packages may NOT work on your system, please consider one of
  the following alternatives to install k2:

      - :ref:`install using conda`
      - :ref:`install k2 from source`

To verify that ``k2`` is installed successfully, run:

.. code-block::

  $ python3 -m k2.version

Congratulations! You have installed ``k2`` successfully.
