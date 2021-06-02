#!/usr/bin/env python3
#
# To install k2, run
#
#       python3 setup.py install
#
# To test that k2 is installed successfully, run
#
#       python3 -m k2.version
#
# To uninstall k2, run
#
#       pip uninstall k2
#
# To build a wheel package, run
#
#       python3 setup.py bdist_wheel
#
#  It generates a file in the dist/ directory.
#
#  An example file looks like
#
#   ./dist/k2-0.3.4.dev20210512+cuda10.1.torch1.7.1-cp38-cp38-linux_x86_64.whl
#
# To build a wheel that can be uploaded to PyPI, run
#
#       K2_IS_FOR_PYPI=1 python3 setup.py bdist_wheel --python-tag=py38
#       twine upload ./dist/k2-0.3.4.dev20210512-py38-none-any.whl
#
# To build k2 without CUDA support, run
#
#       export K2_CMAKE_ARGS="-DK2_WITH_CUDA=OFF"
#       python3 setup.py install

import glob
import os
import setuptools
import shutil
import sys

from setuptools.command.build_ext import build_ext

import get_version

get_package_version = get_version.get_package_version
get_pytorch_version = get_version.get_pytorch_version
is_for_pypi = get_version.is_for_pypi
is_macos = get_version.is_macos
is_windows = get_version.is_windows

if sys.version_info < (3,):
    print('Python 2 has reached end-of-life and is no longer supported by k2.')
    sys.exit(-1)

if sys.version_info < (3, 6):
    print('Python 3.5 has reached end-of-life on September 13th, 2020 '
          'and is no longer supported by k2.')
    sys.exit(-1)

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            if is_for_pypi():
                # In this case, the generated wheel has a name in the form
                # k2-xxx-pyxx-none-any.whl
                self.root_is_pure = True
            else:
                # The generated wheel has a name ending with
                # -linux_x86_64.whl
                self.root_is_pure = False
except ImportError:
    bdist_wheel = None


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs['language'] = 'c++'
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):

    def build_extension(self, ext: setuptools.extension.Extension):
        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        k2_dir = os.path.dirname(os.path.abspath(__file__))

        cmake_args = os.environ.get('K2_CMAKE_ARGS', '')
        make_args = os.environ.get('K2_MAKE_ARGS', '')

        if cmake_args == '':
            cmake_args = '-DCMAKE_BUILD_TYPE=Release'

        if make_args == '' and os.environ.get('K2_IS_GITHUB_ACTIONS', None) is None:
            make_args = '-j'

        if is_macos():
            if not 'K2_WITH_CUDA=OFF' in cmake_args:
                print('Disable CUDA for macOS')
                cmake_args += ' -DK2_WITH_CUDA=OFF'

        if 'PYTHON_EXECUTABLE' not in cmake_args:
            print(f'Setting PYTHON_EXECUTABLE to {sys.executable}')
            cmake_args += f' -DPYTHON_EXECUTABLE={sys.executable}'

        if is_windows():
            build_cmd = f'''
                cmake {cmake_args} -B {self.build_temp} -S {k2_dir}
                cmake --build {self.build_temp} --target _k2 --config Release -- -m
            '''
            print(f'build command is:\n{build_cmd}')
            ret = os.system(f'cmake {cmake_args} -B {self.build_temp} -S {k2_dir}')
            if ret != 0:
                raise Exception('Failed to build k2')
            ret = os.system(f'cmake --build {self.build_temp} --target _k2 --config Release -- -m')
            if ret != 0:
                raise Exception('Failed to build k2')            
        else:
            build_cmd = f'''
                cd {self.build_temp}

                cmake {cmake_args} {k2_dir}

                cat k2/csrc/version.h

                make {make_args} _k2
            '''
            print(f'build command is:\n{build_cmd}')

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception('Failed to build k2')

        lib_so = glob.glob(f'{self.build_temp}/lib/*k2*.so')
        for so in lib_so:
            print(f'Copying {so} to {self.build_lib}/')
            shutil.copy(f'{so}', f'{self.build_lib}/')

        if is_macos():
            lib_so = glob.glob(f'{self.build_temp}/lib/*k2*.dylib')
            for so in lib_so:
                print(f'Copying {so} to {self.build_lib}/')
                shutil.copy(f'{so}', f'{self.build_lib}/')
        elif is_windows():
            # bin/Release/_k2.cp38-win_amd64.pyd
            lib_so = glob.glob(f'{self.build_temp}/**/*k2*.pyd',
                               recursive=True)
            for so in lib_so:
                print(f'Copying {so} to {self.build_lib}/')
                shutil.copy(f'{so}', f'{self.build_lib}/')

            # lib/Release/{_k2,k2_log,k2context,k2fsa}.lib
            lib_so = glob.glob(f'{self.build_temp}/**/*k2*.lib',
                               recursive=True)
            for so in lib_so:
                print(f'Copying {so} to {self.build_lib}/')
                shutil.copy(f'{so}', f'{self.build_lib}/')


def get_long_description():
    with open('README.md', 'r') as f:
        long_description = f.read()
        return long_description


def get_short_description():
    return 'FSA/FST algorithms, intended to (eventually) be interoperable with PyTorch and similar'


with open('k2/python/k2/__init__.py', 'a') as f:
    f.write(f"__dev_version__ = '{get_package_version()}'\n")

dev_requirements = [
    'clang-format==9.0.0',
    'flake8==3.8.3',
    'yapf==0.27.0',
]

install_requires = [
    f'torch=={get_pytorch_version()}',
    'graphviz',
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
        'k2.sparse': 'k2/python/k2/sparse',
        'k2.version': 'k2/python/k2/version',
    },
    packages=['k2', 'k2.ragged', 'k2.sparse', 'k2.version'],
    install_requires=install_requires,
    extras_require={'dev': dev_requirements},
    data_files=[('', ['LICENSE'])],
    ext_modules=[cmake_extension('_k2')],
    cmdclass={
        'build_ext': BuildExtension,
        'bdist_wheel': bdist_wheel
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

# remove the line __dev_version__ from k2/python/k2/__init__.py
with open('k2/python/k2/__init__.py', 'r') as f:
    lines = f.readlines()

with open('k2/python/k2/__init__.py', 'w') as f:
    for line in lines:
        if '__dev_version__' not in line:
            f.write(line)
