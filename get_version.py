#!/usr/bin/env python3

import datetime
import os
import platform
import re
import shutil

import torch


def is_macos():
    return platform.system() == 'Darwin'


def is_windows():
    return platform.system() == 'Windows'


def with_cuda():
    if shutil.which('nvcc') is None:
        return False

    if is_macos():
        return False

    cmake_args = os.environ.get('K2_CMAKE_ARGS', '')
    if 'K2_WITH_CUDA=OFF' in cmake_args:
        return False

    return True


def get_pytorch_version():
    # if it is 1.7.1+cuda101, then strip +cuda101
    return torch.__version__.split('+')[0]


def get_cuda_version():
    from torch.utils import collect_env
    running_cuda_version = collect_env.get_running_cuda_version(
        collect_env.run)
    cuda_version = torch.version.cuda
    if running_cuda_version is not None and cuda_version is not None:
        assert cuda_version in running_cuda_version, \
                f'PyTorch is built with CUDA version: {cuda_version}.\n' \
                f'The current running CUDA version is: {running_cuda_version}'
    return cuda_version


def get_rocm_version():
    """Return the ROCm version string, or None if not a ROCm build."""
    # When PyTorch is built with ROCm, torch.version.hip is set.
    hip_version = getattr(torch.version, "hip", None)
    if hip_version is not None:
        return hip_version
    # Fallback: check environment variable (set in CI Docker containers)
    rocm_env = os.environ.get("ROCM_VERSION", "")
    if rocm_env:
        return rocm_env
    return None


def is_rocm():
    """Return True if this is a ROCm/HIP build."""
    if get_rocm_version() is not None:
        return True
    cmake_args = os.environ.get('K2_CMAKE_ARGS', '')
    if 'K2_WITH_HIP=ON' in cmake_args:
        return True
    return False


def is_for_pypi():
    ans = os.environ.get('K2_IS_FOR_PYPI', None)
    return ans is not None


def is_stable():
    ans = os.environ.get('K2_IS_STABLE', None)
    return ans is not None


def is_for_conda():
    ans = os.environ.get('K2_IS_FOR_CONDA', None)
    return ans is not None


def get_package_version():
    # Set a default CUDA version here so that `pip install k2`
    # uses the default CUDA version.
    #
    # `pip install k2==x.x.x+cu100` to install k2 with CUDA 10.0
    #
    default_cuda_version = '10.1'  # CUDA 10.1

    if is_rocm():
        rocm_version = get_rocm_version()
        # Keep only major.minor (e.g., 7.1.52802 -> 7.1)
        rocm_version = '.'.join(rocm_version.split('.')[:2])
        pytorch_version = get_pytorch_version()
        local_version = f'+rocm{rocm_version}.torch{pytorch_version}'
    elif with_cuda():
        cuda_version = get_cuda_version()
        if is_for_pypi() and default_cuda_version == cuda_version:
            cuda_version = ''
            pytorch_version = ''
            local_version = ''
        else:
            cuda_version = f'+cuda{cuda_version}'
            pytorch_version = get_pytorch_version()
            local_version = f'{cuda_version}.torch{pytorch_version}'
    else:
        pytorch_version = get_pytorch_version()
        local_version = f'+cpu.torch{pytorch_version}'

    if is_for_conda():
        local_version = ''

    if is_for_pypi() and is_macos():
        local_version = ''

    with open('CMakeLists.txt') as f:
        content = f.read()

    latest_version = re.search(r'set\(K2_VERSION (.*)\)', content).group(1)
    latest_version = latest_version.strip('"')

    if not is_stable():
        dt = datetime.datetime.utcnow()
        package_version = f'{latest_version}.dev{dt.year}{dt.month:02d}{dt.day:02d}{local_version}'
    else:
        package_version = f'{latest_version}'
    return package_version


if __name__ == '__main__':
    print(get_package_version())
