#!/usr/bin/env python3

import datetime
import os
import re


def get_cuda_version():
    ans = os.environ.get('K2_CUDA_VERSION', None)
    if ans is not None:
        return str(ans)
    import torch
    from torch.utils import collect_env
    running_cuda_version = collect_env.get_running_cuda_version(
        collect_env.run)
    cuda_version = torch.version.cuda
    if running_cuda_version is not None:
        assert cuda_version in running_cuda_version, \
                f'PyTorch is built with CUDA version: {cuda_version}.\n' \
                f'The current running CUDA version is: {running_cuda_version}'
    cuda_version = cuda_version.split('.')
    major, minor = int(cuda_version[0]), int(cuda_version[1])
    cuda_version = major * 10 + minor
    return f'{cuda_version}'


def get_package_version():
    # Set a default CUDA version here so that `pip install k2`
    # uses the default CUDA version.
    #
    # `pip install k2==x.x.x+cu100` to install k2 with CUDA 10.0
    #
    default_cuda_version = '101'  # CUDA 10.1
    cuda_version = get_cuda_version()
    if default_cuda_version != cuda_version:
        cuda_version = f'+cu{cuda_version}'
    else:
        cuda_version = ''

    with open('CMakeLists.txt') as f:
        content = f.read()

    latest_version = re.search(r'set\(K2_VERSION (.*)\)', content).group(1)
    latest_version = latest_version.strip('"')

    dt = datetime.datetime.utcnow()
    package_version = f'{latest_version}{cuda_version}.dev{dt.year}{dt.month:02d}{dt.day:02d}'
    return package_version


if __name__ == '__main__':
    print(get_package_version())
