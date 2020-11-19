#!/usr/bin/env python3

import torch  # noqa
import _k2


def main():
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
    ''')


if __name__ == '__main__':
    main()
