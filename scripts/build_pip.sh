#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)
#
# Steps to build a pip package:
#
# (1)
#    pip install wheel twine
#    sudo apt-get install chrpath # Skip it if you don't have sudo permission
#
# (2)
#    cd /path/to/k2
#    mkdir build
#    cd build
#    cmake -DCMAKE_BUILD_TYPE=Release ..
#    make -j _k2
#
# It will generate 3 files in the folder `lib`:
#   libk2context.so
#   libk2fsa.so
#   _k2.cpython-3?m-x86_64-linux-gnu.so, where ? depends on your python version.
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

mkdir -p .temp_lib

mv $build_dir/lib/lib*test*.so .temp_lib

# save the generated libs as we want to restore their RPATH
mkdir -p $build_dir/.lib-bak

cp -v $build_dir/lib/*.so $build_dir/.lib-bak

for lib in $build_dir/lib/*.so; do
  #strip $lib
  if command -v chrpath > /dev/null 2>&1; then
    # remove RPATH information
    chrpath -r '$ORIGIN' $lib
  fi
done

python3 setup.py bdist_wheel

mv $build_dir/.lib-bak/*.so $build_dir/lib/

mv .temp_lib/lib*test*.so $build_dir/lib/

rm -rfv $build_dir/.lib-bak
rm -rf $build_dir/lib/k2
