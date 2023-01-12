Installation
============

.. HINT::

  You don't need GPUs to build, install, and run k2.

We provide the following approaches to install k2. You can choose any of them that
is appropriate for you.

Available versions of Python, CUDA, and PyTorch of different approaches are listed
below:

  - From conda (**recommended**)

    - |conda_python_versions|
    - |conda_cuda_versions|
    - |conda_pytorch_versions|

  - From pip (k2-fsa.org)

    - |pip_python_versions|
    - |pip_cuda_versions|
    - |pip_pytorch_versions|

  - From pypi (pypi.org)

    - |pypi_python_versions|
    - |pypi_cuda_versions|
    - |pypi_pytorch_versions|

  - From source (**for advanced users**)

    - |source_python_versions|
    - |source_cuda_versions|
    - |source_pytorch_versions|

.. toctree::
   :maxdepth: 1

   cuda-cudnn.rst
   conda
   pip
   pip_pypi
   from_source
   for_developers
   faqs

.. |conda_python_versions| image:: ./images/python_ge_3.6-blue.svg
  :alt: Supported python versions

.. |conda_cuda_versions| image:: ./images/cuda_ge_10.1-orange.svg
  :alt: Supported cuda versions

.. |conda_pytorch_versions| image:: ./images/pytorch_ge_1.6.0-green.svg
  :alt: Supported pytorch versions

.. |pip_python_versions| image:: ./images/python_ge_3.6-blue.svg
  :alt: Supported python versions

.. |pip_cuda_versions| image:: ./images/cuda_ge_10.1-orange.svg
  :alt: Supported cuda versions

.. |pip_pytorch_versions| image:: ./images/pytorch_ge_1.6.0-green.svg
  :alt: Supported pytorch versions

.. |pypi_python_versions| image:: ./images/pypi_python-3.6_3.7_3.8-blue.svg
  :alt: Supported python versions

.. |pypi_cuda_versions| image:: ./images/pypi_cuda-10.1-orange.svg
  :alt: Supported cuda versions

.. |pypi_pytorch_versions| image:: ./images/pypi_pytorch-1.7.1-green.svg
  :alt: Supported pytorch versions

.. |source_python_versions| image:: ./images/python_ge_3.6-blue.svg
  :alt: Supported python versions

.. |source_cuda_versions| image:: ./images/cuda_ge_10.1-orange.svg
  :alt: Supported cuda versions

.. |source_pytorch_versions| image:: ./images/pytorch_ge_1.6.0-green.svg
  :alt: Supported pytorch versions

Reporting issues
----------------

If you encounter any errors while using k2 after installation, please
create an issue `on GitHub <https://github.com/k2-fsa/k2/issues/new>`_
and tell us your current environment by posting the output of the following
two commands::

  python3 -m k2.version
  python3 -m torch.utils.collect_env
