#!/bin/bash
#
# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

cd $k2_dir/k2/python/tests

for f in *.py; do
  echo "Run $f"
  python3 $f
  if [ $? -ne 0 ]; then
    echo "$f failed!"
    exit 1
  fi
done

exit 0
