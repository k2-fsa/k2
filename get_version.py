#!/usr/bin/env python3

import datetime
import os
import re

import torch


def get_pytorch_version():
    # if it is 1.7.1+cuda101, then strip +cuda101
    return torch.__version__.split('+')[0]


def get_cuda_version():
    from torch.utils import collect_env
    running_cuda_version = collect_env.get_running_cuda_version(
        collect_env.run)
    cuda_version = torch.version.cuda
    if running_cuda_version is not None:
        assert cuda_version in running_cuda_version, \
                f'PyTorch is built with CUDA version: {cuda_version}.\n' \
                f'The current running CUDA version is: {running_cuda_version}'
    return cuda_version


def is_for_pypi():
    ans = os.environ.get('K2_IS_FOR_PYPI', None)
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

    cuda_version = get_cuda_version()

    if is_for_pypi() and default_cuda_version == cuda_version:
        cuda_version = ''
        pytorch_version = ''
        local_version = ''
    else:
        cuda_version = f'+cuda{cuda_version}'
        pytorch_version = get_pytorch_version()
        local_version = f'{cuda_version}.torch{pytorch_version}'

    with open('CMakeLists.txt') as f:
        content = f.read()

    latest_version = re.search(r'set\(K2_VERSION (.*)\)', content).group(1)
    latest_version = latest_version.strip('"')

    dt = datetime.datetime.utcnow()
    package_version = f'{latest_version}.dev{dt.year}{dt.month:02d}{dt.day:02d}{local_version}'
    return package_version


if __name__ == '__main__':
    print(get_package_version())
