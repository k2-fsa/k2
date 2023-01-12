.. _cuda_and_cudnn:

CUDA and cuDNN
==============

``k2`` supports ``CUDA >= 10.1``.

If you want to build ``k2`` from source with GPU support, you have to install
``cudatoolkit`` and ``cuDNN`` first. This section describes how to do that.

You don't need to use ``sudo`` to install ``cudatoolkit`` and ``cuDNN``.

.. caution::

  ``cudatoolkit`` installed by ``conda install`` is not sufficient to compile ``k2``.

  ``cudatoolkit`` installed by ``conda install`` is not sufficient to compile ``k2``.

  ``cudatoolkit`` installed by ``conda install`` is not sufficient to compile ``k2``.


You can choose any CUDA version that is suitable for you.

.. note::

  You can install multiple CUDA versions into difference directories on your
  system.


  If you follow this section, you can use ``source activate-cuda-10.1.sh``
  to activate CUDA 10.1 and ``source activate-cuda-10.2.sh`` to activate
  CUDA 10.2, for instance.


CUDA 10.1
---------

You can use the following commands to install CUDA 10.1. We install it
into ``/ceph-sh1/fangjun/software/cuda-10.1.243``. You can replace it
if needed.

.. code-block:: bash

  wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run

  chmod +x cuda_10.1.243_418.87.00_linux.run

  ./cuda_10.1.243_418.87.00_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-10.1.243 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page

Install cuDNN for CUDA 10.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 10.1.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-10.1-linux-x64-v8.0.2.39.tgz

  tar xvf cudnn-10.1-linux-x64-v8.0.2.39.tgz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-10.1.243

Set environment variables for CUDA 10.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 10.1. You can save the following code to ``activate-cuda-10.1.sh``
and use ``source activate-cuda-10.1.sh`` if you want to activate CUDA 10.1.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-10.1.243
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS


To check that you have installed CUDA 10.1 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-10.1.243/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2019 NVIDIA Corporation
  Built on Sun_Jul_28_19:07:16_PDT_2019
  Cuda compilation tools, release 10.1, V10.1.243

CUDA 10.2
---------

You can use the following commands to install CUDA 10.2. We install it
into ``/ceph-sh1/fangjun/software/cuda-10.2.89``. You can replace it
if needed.

.. code-block:: bash

  wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

  chmod +x cuda_10.2.89_440.33.01_linux.run

  ./cuda_10.2.89_440.33.01_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-10.2.89 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page

Install cuDNN for CUDA 10.2
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 10.2.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-10.2-linux-x64-v8.0.2.39.tgz

  tar xvf cudnn-10.2-linux-x64-v8.0.2.39.tgz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-10.2.89


Set environment variables for CUDA 10.2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 10.2. You can save the following code to ``activate-cuda-10.2.sh``
and use ``source activate-cuda-10.2.sh`` if you want to activate CUDA 10.2.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-10.2.89
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 10.2 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-10.2.89/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2019 NVIDIA Corporation
  Built on Wed_Oct_23_19:24:38_PDT_2019
  Cuda compilation tools, release 10.2, V10.2.89

CUDA 11.0
---------

You can use the following commands to install CUDA 11.0. We install it
into ``/ceph-sh1/fangjun/software/cuda-11.0.3``. You can replace it
if needed.

.. code-block:: bash

  wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run

  chmod +x cuda_11.0.3_450.51.06_linux.run

  ./cuda_11.0.3_450.51.06_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-11.0.3 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page

Install cuDNN for CUDA 11.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 11.0.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-11.0-linux-x64-v8.0.5.39.tgz

  tar xvf cudnn-11.0-linux-x64-v8.0.4.30.tgz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-11.0.3

Set environment variables for CUDA 11.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 11.0. You can save the following code to ``activate-cuda-11.0.sh``
and use ``source activate-cuda-11.0.sh`` if you want to activate CUDA 11.0.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-11.0.3
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 11.0 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-11.0.3/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2020 NVIDIA Corporation
  Built on Wed_Jul_22_19:09:09_PDT_2020
  Cuda compilation tools, release 11.0, V11.0.221
  Build cuda_11.0_bu.TC445_37.28845127_0

CUDA 11.3
---------

You can use the following commands to install CUDA 11.3. We install it
into ``/ceph-sh1/fangjun/software/cuda-11.3.1``. You can replace it
if needed.

.. code-block:: bash

  wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run

  chmod +x cuda_11.3.1_465.19.01_linux.run

  ./cuda_11.3.1_465.19.01_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-11.3.1 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page


Install cuDNN for CUDA 11.3
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 11.3.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-11.3-linux-x64-v8.2.0.53.tgz

  tar xvf cudnn-11.3-linux-x64-v8.2.1.32.tgz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-11.3.1

Set environment variables for CUDA 11.3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 11.3. You can save the following code to ``activate-cuda-11.3.sh``
and use ``source activate-cuda-11.3.sh`` if you want to activate CUDA 11.3.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-11.3.1
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 11.3 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-11.3.1/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2021 NVIDIA Corporation
  Built on Mon_May__3_19:15:13_PDT_2021
  Cuda compilation tools, release 11.3, V11.3.109
  Build cuda_11.3.r11.3/compiler.29920130_0

CUDA 11.5
---------

You can use the following commands to install CUDA 11.5. We install it
into ``/ceph-sh1/fangjun/software/cuda-11.5.2``. You can replace it
if needed.

.. code-block:: bash

  wget https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run

  chmod +x cuda_11.5.2_495.29.05_linux.run

  ./cuda_11.5.2_495.29.05_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-11.5.2 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page

Install cuDNN for CUDA 11.5
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 11.5.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz

  tar xvf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-11.5.2

Set environment variables for CUDA 11.5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 11.5. You can save the following code to ``activate-cuda-11.5.sh``
and use ``source activate-cuda-11.5.sh`` if you want to activate CUDA 11.5.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-11.5.2
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 11.5 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-11.5.2/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2021 NVIDIA Corporation
  Built on Thu_Nov_18_09:45:30_PST_2021
  Cuda compilation tools, release 11.5, V11.5.119
  Build cuda_11.5.r11.5/compiler.30672275_0

CUDA 11.6
---------

You can use the following commands to install CUDA 11.6. We install it
into ``/ceph-sh1/fangjun/software/cuda-11.6.1``. You can replace it
if needed.

.. code-block:: bash

  wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_510.47.03_linux.run

  chmod +x cuda_11.6.1_510.47.03_linux.run

  ./cuda_11.6.1_510.47.03_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-11.6.1 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page

Install cuDNN for CUDA 11.6
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 11.6.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-11.3-linux-x64-v8.2.0.53.tgz

  # Note: cudnn-11.3 works for CUDA 11.x
  tar xvf cudnn-11.3-linux-x64-v8.2.1.32.tgz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-11.6.1

Set environment variables for CUDA 11.6
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 11.6. You can save the following code to ``activate-cuda-11.6.sh``
and use ``source activate-cuda-11.6.sh`` if you want to activate CUDA 11.6.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-11.6.1
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 11.6 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-11.6.1/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Thu_Feb_10_18:23:41_PST_2022
  Cuda compilation tools, release 11.6, V11.6.112
  Build cuda_11.6.r11.6/compiler.30978841_0

CUDA 11.7
---------

You can use the following commands to install CUDA 11.7. We install it
into ``/ceph-sh1/fangjun/software/cuda-11.7.1``. You can replace it
if needed.

.. code-block:: bash

  wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run

  chmod +x cuda_11.7.1_515.65.01_linux.run

  ./cuda_11.7.1_515.65.01_linux.run \
    --silent \
    --toolkit \
    --installpath=/ceph-sh1/fangjun/software/cuda-11.7.1 \
    --no-opengl-libs \
    --no-drm \
    --no-man-page

Install cuDNN for CUDA 11.7
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, install ``cuDNN`` for CUDA 11.7.

.. code-block:: bash

  wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-11.3-linux-x64-v8.2.0.53.tgz

  # Note: cudnn-11.3 works for CUDA 11.x
  tar xvf cudnn-11.3-linux-x64-v8.2.1.32.tgz --strip-components=1 -C /ceph-sh1/fangjun/software/cuda-11.7.1

Set environment variables for CUDA 11.7
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that we have to set the following environment variables after installing
CUDA 11.7. You can save the following code to ``activate-cuda-11.7.sh``
and use ``source activate-cuda-11.7.sh`` if you want to activate CUDA 11.7.

.. code-block:: bash

  export CUDA_HOME=/ceph-sh1/fangjun/software/cuda-11.7.1
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export CUDA_TOOLKIT_ROOT=$CUDA_HOME
  export CUDA_BIN_PATH=$CUDA_HOME
  export CUDA_PATH=$CUDA_HOME
  export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
  export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 11.7 successfully, please run:

.. code-block:: bash

  which nvcc

  nvcc --version

The output should look like the following:

.. code-block:: bash

  /ceph-sh1/fangjun/software/cuda-11.7.1/bin/nvcc

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Wed_Jun__8_16:49:14_PDT_2022
  Cuda compilation tools, release 11.7, V11.7.99
  Build cuda_11.7.r11.7/compiler.31442593_0
