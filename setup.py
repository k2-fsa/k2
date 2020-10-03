import setuptools


def get_long_description():
    with open('README.md', 'r') as f:
        long_description = f.read()
        return long_description


version = '0.0.1'
description = 'FSA/FST algorithms, intended to (eventually) be interoperable with PyTorch and similar'

setuptools.setup(
    python_requires='>=3.6',
    name='k2',
    version=version,
    author='Daniel Povey',
    author_email='dpovey@gmail.com',
    description=description,
    keywords='k2, FSA, FST',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/k2-fsa/k2',
    package_dir={'': 'k2/python'},
    packages=['k2'],
    install_requires=['torch', 'graphviz'],
    data_files=[('', ['LICENSE'])],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
)
