#!/usr/bin/env python3

# Copyright      2020  Xiaomi Corp.   (author: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch  # noqa
import _k2

__build_type__ = _k2.version.build_type
__git_date__ = _k2.version.git_date
__git_sha1__ = _k2.version.git_sha1
__version__ = _k2.version.__version__


def main():
    '''Collect the information about the environment in which k2 was built.

    When reporting issues, please use::

        python3 -m k2.version

    to collect the environment information about k2.

    Please also attach the environment information about PyTorch using::

        python3 -m torch.utils.collect_env
    '''
    print('Collecting environment information...')
    version = _k2.version.__version__
    git_sha1 = _k2.version.git_sha1
    git_date = _k2.version.git_date
    cuda_version = _k2.version.cuda_version
    cudnn_version = _k2.version.cudnn_version
    python_version = _k2.version.python_version
    build_type = _k2.version.build_type
    os_type = _k2.version.os_type
    cmake_version = _k2.version.cmake_version
    gcc_version = _k2.version.gcc_version
    cmake_cuda_flags = _k2.version.cmake_cuda_flags
    cmake_cxx_flags = _k2.version.cmake_cxx_flags
    torch_version = _k2.version.torch_version
    torch_cuda_version = _k2.version.torch_cuda_version
    enable_nvtx = _k2.version.enable_nvtx
    with_cuda = _k2.version.with_cuda
    disable_debug = _k2.version.disable_debug
    sync_kernels = os.getenv('K2_SYNC_KERNELS', None) is not None
    disable_checks = os.getenv('K2_DISABLE_CHECKS', None) is not None

    print(f'''
k2 version: {version}
Build type: {build_type}
Git SHA1: {git_sha1}
Git date: {git_date}
Cuda used to build k2: {cuda_version}
cuDNN used to build k2: {cudnn_version}
Python version used to build k2: {python_version}
OS used to build k2: {os_type}
CMake version: {cmake_version}
GCC version: {gcc_version}
CMAKE_CUDA_FLAGS: {cmake_cuda_flags}
CMAKE_CXX_FLAGS: {cmake_cxx_flags}
PyTorch version used to build k2: {torch_version}
PyTorch is using Cuda: {torch_cuda_version}
NVTX enabled: {enable_nvtx}
With CUDA: {with_cuda}
Disable debug: {disable_debug}
Sync kernels : {sync_kernels}
Disable checks: {disable_checks}
    ''')


if __name__ == '__main__':
    main()
