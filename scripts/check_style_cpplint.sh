#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)
#
# Usage:
#
# (1) To check files of the last commit
#  ./scripts/check_style_cpplint.sh
#
# (2) To check changed files not committed yet
#  ./scripts/check_style_cpplint.sh 1
#
# (3) To check all files in the project
#  ./scripts/check_style_cpplint.sh 2


cpplint_version="1.5.4"
cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

build_dir=$k2_dir/build
mkdir -p $build_dir

cpplint_src=$build_dir/cpplint-${cpplint_version}/cpplint.py

if [ ! -d "$build_dir/cpplint-${cpplint_version}" ]; then
  pushd $build_dir
  wget https://github.com/cpplint/cpplint/archive/${cpplint_version}.tar.gz
  tar xf ${cpplint_version}.tar.gz
  rm ${cpplint_version}.tar.gz

  # cpplint will report the following error for: __host__ __device__ (
  #
  #     Extra space before ( in function call  [whitespace/parens] [4]
  #
  # the following patch disables the above error
  sed -i "3490i\        not Search(r'__host__ __device__\\\s+\\\(', fncall) and" $cpplint_src
  popd
fi

source $k2_dir/scripts/utils.sh

# return true if the given file is a c++ source file
# return false otherwise
function is_source_code_file() {
  case "$1" in
    *.cc|*.h|*.cu)
      echo true;;
    *)
      echo false;;
  esac
}

function check_style() {
  python3 $cpplint_src $1 || abort $1
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
  case "$1" in
    1)
      echo "Check changed files"
      files=$(check_current_dir)
      ;;
    2)
      echo "Check all files"
      files=$(find $k2_dir/k2 -name "*.h" -o -name "*.cc" -o -name "*.cu")
      ;;
    *)
      echo "Check last commit"
      files=$(check_last_commit)
      ;;
  esac

  for f in $files; do
    need_check=$(is_source_code_file $f)
    if $need_check; then
      [[ -f $f ]] && check_style $f
    fi
  done
}

function main() {
  do_check $1

  ok "Great! Style check passed!"
}

cd $k2_dir

main $1
