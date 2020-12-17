# Welcome to the k2 setup.py.
#
# Please follow instructions in scripts/build_pip.sh to use this file.
#
import datetime
import re
import setuptools
import sys

if sys.version_info < (3,):
    print('Python 2 has reached end-of-life and is no longer supported by k2.')
    sys.exit(-1)

if sys.version_info < (3, 6):
    print('Python 3.5 has reached end-of-life on September 13th, 2020 '
          'and is no longer supported by k2.')
    sys.exit(-1)

# Refer to https://stackoverflow.com/questions/45150304/how-to-force-a-python-wheel-to-be-platform-specific-when-building-it
# for why to introduce `bdist_wheel`.
#
# With `bdist_wheel`, the final wheel name looks like `k2-0.0.1.dev20201104-cp37-cp37m-linux_x86_64.whl`
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


def get_long_description():
    with open('README.md', 'r') as f:
        long_description = f.read()
        return long_description


def get_cuda_version():
    from torch.utils import collect_env
    cuda_version = collect_env.get_running_cuda_version(
        collect_env.run).split('.')
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


def get_short_description():
    return 'FSA/FST algorithms, intended to (eventually) be interoperable with PyTorch and similar'


dev_requirements = [
    'clang-format==9.0.0',
    'flake8==3.8.3',
    'yapf==0.27.0',
]


setuptools.setup(
    python_requires='>=3.6',
    name='k2',
    version=get_package_version(),
    author='Daniel Povey',
    author_email='dpovey@gmail.com',
    keywords='k2, FSA, FST',
    description=get_short_description(),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/k2-fsa/k2',
    package_dir={
        'k2': 'k2/python/k2',
        'k2.ragged': 'k2/python/k2/ragged',
    },
    packages=['k2', 'k2.ragged'],
    install_requires=['torch', 'graphviz'],
    extras_require={
        'dev': dev_requirements
    },
    data_files=[('', ['LICENSE'])],
    cmdclass={'bdist_wheel': bdist_wheel},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
)
