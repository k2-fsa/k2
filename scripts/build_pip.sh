#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)
#
# Steps to build a pip package:
#
# (1)
#    pip install wheel twine
#    sudo apt-get install chrpath
#
# (2)
#    cd /path/to/k2
#    mkdir build
#    cd build
#    cmake -DCMAKE_BUILD_TYPE=Release ..
#    make -j _k2
#
# It will generate 3 files in the folder `lib`:
#   libcontext.so
#   libfsa.so
#   _k2.cpython-3?m-x86_64-linux-gnu.so, where ? depends on your python version.
#
# Please delete other files inside `lib`; otherwise, they will be copied to
# the final `whl`!
#
# (3)
#    cd /path/to/k2
#    ./scripts/build_pip.sh
#
# (4) You will find the `whl` file in the `dist` directory.
#
# (5) Upload the generated `whl` file to pypi. Example:
#
#     twine upload dist/k2-0.0.1-py36-none-any.whl
#
#   Enter your username and password!
#
# (6) Done.

if ! command -v chrpath > /dev/null 2>&1; then
  echo "Please install chrpath  first"
  exit 1
fi

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)
build_dir=$k2_dir/build

cd $k2_dir

if [ ! -d $build_dir/lib ]; then
  echo "Please run: "
  echo "  mkdir $build_dir"
  echo "  cd $build_dir"
  echo "  cmake .."
  echo "  make -j _k2 "
  echo "before running this script"
  exit 1
fi

for lib in $build_dir/lib/*.so; do
  chrpath -r '$ORIGIN' $lib
done

tag=$(python -c "import sys; print(sys.version[:3].replace('.', ''))")

python3 setup.py bdist_wheel --python-tag py$tag
