From pre-compiled wheels (recommended)
======================================

We provide pre-compiled wheels for the following platforms:


.. toctree::
   :maxdepth: 1

   ./pre-compiled-cpu-wheels-linux/index.rst
   ./pre-compiled-cuda-wheels-linux/index.rst
   ./pre-compiled-cpu-wheels-macos/index.rst
   ./pre-compiled-cpu-wheels-windows/index.rst

We recommend that you use this approach to install `k2`_.

.. hint::

   Please always install the latest version of `k2`_.

Installation examples
---------------------

We provide several examples below to show you how to install `k2`_

Linux (CPU) example
^^^^^^^^^^^^^^^^^^^^

Suppose that we want to install the following wheel

.. code-block::

  https://huggingface.co/csukuangfj/k2/resolve/main/cpu/k2-1.24.3.dev20230719+cpu.torch2.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

we can use one of the following methods:

.. code-block:: bash

   # method 1
   pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   pip install k2==1.24.3.dev20230719+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html

   # method 2
   pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   wget https://huggingface.co/csukuangfj/k2/resolve/main/cpu/k2-1.24.3.dev20230719+cpu.torch2.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   pip install ./k2-1.24.3.dev20230719+cpu.torch2.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

.. hint::

   You can visit `<https://k2-fsa.github.io/k2/cpu.html>`_ to see avaiable versions of `k2`_.


macOS (CPU) example
^^^^^^^^^^^^^^^^^^^

Suppose we want to use the following wheel:

.. code-block::

   https://huggingface.co/csukuangfj/k2/resolve/main/macos/k2-1.24.3.dev20230720+cpu.torch2.0.1-cp38-cp38-macosx_10_9_x86_64.whl

we can use the following methods:

.. code-block:: bash

   # method 1
   pip install torch==2.0.1
   pip install k2==1.24.3.dev20230720+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html

   # method 2
   pip install torch==2.0.1
   wget https://huggingface.co/csukuangfj/k2/resolve/main/macos/k2-1.24.3.dev20230720+cpu.torch2.0.1-cp38-cp38-macosx_10_9_x86_64.whl
   pip install ./k2-1.24.3.dev20230720+cpu.torch2.0.1-cp38-cp38-macosx_10_9_x86_64.whl

.. hint::

   You can visit `<https://k2-fsa.github.io/k2/cpu.html>`_ to see avaiable versions of `k2`_.

Windows (CPU) example
^^^^^^^^^^^^^^^^^^^^^

Suppose we want to install the following wheel

.. code-block::

   https://huggingface.co/csukuangfj/k2/resolve/main/windows-cpu/k2-1.24.3.dev20230726+cpu.torch2.0.1-cp38-cp38-win_amd64.whl

we can use the one of the following methods:

.. code-block:: bash

   # method 1
   pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   pip install k2==1.24.3.dev20230726+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html

   # method 2
   pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   wget https://huggingface.co/csukuangfj/k2/resolve/main/windows-cpu/k2-1.24.3.dev20230726+cpu.torch2.0.1-cp38-cp38-win_amd64.whl
   pip install k2-1.24.3.dev20230726+cpu.torch2.0.1-cp38-cp38-win_amd64.whl

.. hint::

   If you want to build `k2`_ with CUDA support on Windows, please consider compiling
   `k2`_ from source.

.. hint::

   You can visit `<https://k2-fsa.github.io/k2/cpu.html>`_ to see avaiable versions of `k2`_.

Linux (CUDA) example
^^^^^^^^^^^^^^^^^^^^

Suppose we want to install

.. code-block:: bash

    https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.24.3.dev20230718+cuda11.7.torch2.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

we can use the following methods:

.. code-block:: bash

   # method 1
   pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   pip install k2==1.24.3.dev20230718+cuda11.7.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html

   # method 2
   pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   wget https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.24.3.dev20230718+cuda11.7.torch2.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
   pip install ./k2-1.24.3.dev20230718+cuda11.7.torch2.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

.. hint::

   You can visit `<https://k2-fsa.github.io/k2/cuda.html>`_ to see avaiable versions of `k2`_.
