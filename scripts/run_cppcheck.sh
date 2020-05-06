#!/bin/bash

# Copyright 2020 Fangjun Kuang (csukuangfj@gmail.com)

# See ../LICENSE for clarification regarding multiple authors

# Usage:
#  ./scripts/run_cppcheck.sh
#     Use the default build dir `build`
#
#  ./scripts/run_cppcheck.sh /path/to/build
#     Use the specified build dir `/path/to/build`
#
# Before running this scripts, you have to install `cppcheck`.
# You can use one of the following methods to install it.
#
#  (1) Ubuntu
#
#       sudo apt-get install cppcheck
#
#  (2) macOS
#
#       brew install cppcheck
#
#  (3) Install from source
#
#     git clone https://github.com/danmar/cppcheck.git
#     cd cppcheck
#     mkdir build
#     cd build
#     cmake ..
#     make
#     make install

if ! command -v cppcheck > /dev/null 2>&1; then
  echo "Please install cppcheck first"
  exit 1
fi

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

if [ $# -ge 1 ]; then
  build_dir=$(cd $1 && pwd)
  shift
else
  build_dir=$k2_dir/build
fi

if [ ! -f $build_dir/compile_commands.json ]; then
  echo "$build_dir/compile_commands.json is missing"
  echo "Did you forget to configure the project?"
  echo "Please run: "
  echo "  cd build"
  echo "  cmake .."
  echo "before running this script."
  exit 1
fi

suppression_file="$k2_dir/scripts/suppressions.txt"

tmpfile=$(mktemp)
trap "{ rm $tmpfile; }" EXIT

cat $suppression_file |
  sed -e "s:build_dir:$build_dir:g" \
      -e "s:k2_dir:$k2_dir:g" > $tmpfile

cfg="$k2_dir/scripts/googletest.cfg"

echo "Running cppcheck ......"

source $k2_dir/scripts/utils.sh

cppcheck \
  -q \
  --enable=style \
  -i $build_dir/_deps \
  --library="$cfg" \
  --suppressions-list="$tmpfile" \
  --project=$build_dir/compile_commands.json \
  --error-exitcode=1 \
  --template='{file}:{line},{severity},{id},{message}'

if [ $? -ne 0 ]; then
  error "cppcheck failed, please check the error message."
else
  ok "Great! cppcheck passed!"
fi

# cppcheck is optional.
# we always return 0.
exit 0
