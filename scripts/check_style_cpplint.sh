#!/bin/bash

# Copyright 2020 Fangjun Kuang (csukuangfj@gmail.com)

# See ../LICENSE for clarification regarding multiple authors

# Usage:
#  ./scripts/check_style_cpplint.sh
#     check modified files using the default build directory "build"
#
#  ./scripts/check_style_cpplint.sh ./build
#     check modified files using the specified build directory "./build"
#
#  ./scripts/check_style_cpplint.sh ./build 1
#     check modified files of last commit using the specified build directory "./build"
#
# You can also go to the build directory and do the following:
#   cd build
#   cmake ..
#   make check_style

default='\033[0m'
bold='\033[1m'
red='\033[31m'
green='\033[32m'

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

if [ $# -ge 1 ]; then
  build_dir=$1
  shift
else
  # we assume that the build dir is "./build"; cpplint
  # is downloaded automatically when the project is configured.
  build_dir=$k2_dir/build
fi
cpplint_src=$build_dir/third_party/cpplint/src/cpplint_py/cpplint.py

function ok() {
  printf "${bold}${green}[OK]${default} $1\n"
}

function error() {
  printf "${bold}${red}[FAILED]${default} $1\n"
  exit 1
}

# return true if the given file is a c++ source file
# return false otherwise
function is_source_code_file() {
  case "$1" in
    *.cc|*.h)
      echo true;;
    *)
      echo false;;
  esac
}

function check_style() {
  python3 $cpplint_src $1 || error $1
}

function check_last_commit() {
  files=$(git diff HEAD^1 --name-only --diff-filter=ACDMRUXB)
  echo $files
}

function check_current_dir() {
  files=$(git status -s -uno --porcelain | awk '{
  if (NF == 4) {
    # a file has been renamed
    print $NF
  } else {
    print $2
  }}')

  echo $files
}

function do_check() {
  if [ $# -eq 1 ]; then
    echo "checking last commit"
    files=$(check_last_commit)
  else
    echo "checking current dir"
    files=$(check_current_dir)
  fi

  for f in $files; do
    need_check=$(is_source_code_file $f)
    if $need_check; then
      [[ -f $f ]] && check_style $f
    fi
  done
}

function main() {
  if [ ! -f $cpplint_src ]; then
    error "\n$cpplint_src does not exist.\n\
Please run
    mkdir build
    cd build
    cmake ..
before running this script."
  fi

  do_check $1

  ok "Great! Style check passed!"
}

cd $k2_dir

main $1
