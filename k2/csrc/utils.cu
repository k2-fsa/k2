/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/cub.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/moderngpu_allocator.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/utils.h"

namespace k2 {


/*

  See declaration of RowSplitsToRowIds() in utils.h.  These are implementation
  notes.

    Suppose the range we need to fill with a
    particular number (say, x) is from 1010 to 10000 inclusive (binary) The
    first kernel writes x to positions 1010, 1100, 10000; the significance of
    that sequence is we keep adding the smallest number we can add to get
    another zero at the end of the binary representation, until we exceed the
    range we're supposed to fill.  The second kernel: for a given index into x
    that is must fill (say, 1111), it asks "is the index currently here already
    the right one?", which it can test using the function is_valid_index()
    below; if it's not already correct, it searches in a sequence of positions:
    1110, 1100, 1000, 0000, like our sequence above but going downwards, again
    getting more zeros at the end of the binary representation, until it finds
    the correct value in the array at the searched position; then it copies the
    discovered value the original position requested (here, 1111).


    First kernel pseudocode: for each index 'i' into 't', it does:
      for (int32_t n=0, j = t[i]; j < t[i+1]; n++) {
         x[j] = i;
         if (j & (1<<n))  j += (1 << n);
      }
    Second kernel pseudocode: for each element of x, it searches for the right
  index.  Suppose we're given num_indexes == length(n) == length(t) - 1.  Define
  is_valid_index as follows:
       // returns true if j is the value that we should be putting at position
  'i' in x:
       // that is, if t[j] <= i < t[j+1].
       bool is_valid_index(i, j) {
          return (j >= 0 && j < num_indexes && t[j] <= i && i < t[j+1]);
       }
       // We suppose we are given i (the position into x that we're responsible
  for
       // setting:
       orig_i = i;
       for (int32_t n=0; !is_valid_index(i, x[i]); n++) {
         if (i & (1<<n))  i -= (1 << n);
       }
       x[orig_i] = x[i];
*/
void RowSplitsToRowIds(ContextPtr c, int32_t num_rows,
                       const int32_t *row_splits, int32_t num_elems,
                       int32_t *row_ids) {
  NVTX_RANGE(K2_FUNC);
  if (num_rows <= 0 || num_elems <= 0) return;
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    int32_t cur_row_start = row_splits[0];
    K2_CHECK_EQ(cur_row_start, 0);
    K2_CHECK_EQ(row_splits[num_rows], num_elems);
    for (int32_t row = 0; row < num_rows; ++row) {
      int32_t next_row_start = row_splits[row + 1];
      for (; cur_row_start < next_row_start; ++cur_row_start)
        row_ids[cur_row_start] = row;
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    if (1) {
      mgpu::context_t *mgpu_allocator = GetModernGpuAllocator(c);
      mgpu::load_balance_search(num_elems, row_splits, num_rows, row_ids,
                                *mgpu_allocator);
    } else {
      // TODO: Will probably just delete this branch at some point.

      // The following algorithm isn't particularly adapted to GPU hardware in
      // terms of coalesced reads and writes and so on, but it has reasonable
      // asymptotic time complexity (assuming all kernels run in parallel),
      // specifically: O(log(largest(row_splits[i+1]-row_splits[i])))

      K2_EVAL(
          c, num_elems + 1, lambda_init_minus_one,
          (int32_t i)->void { row_ids[i] = -1; });

      K2_EVAL(
          c, num_elems + 1, lambda_phase_one, (int32_t i)->void {
            int32_t this_row_split = row_splits[i],
                    next_row_split =
                        (i < num_rows ? row_splits[i + 1] : this_row_split + 1);
            if (this_row_split < next_row_split) row_ids[this_row_split] = i;
            // we have to fill in row_ids[this_row_split],
            // row_ids[this_row_split+1]... row_ids[next_row_split-1] with the
            // same value but that could be a long loop. Instead we write at
            // this_row_split and all indexes this_row_split < i <
            // next_row_split such that i is the result of rounding up
            // this_row_split to (something)*2^n, for n = 1, 2, 3, ... this will
            // take time logarithmic in (next_row_split - this_row_split). we
            // can then fill in the gaps with a logarithmic-time loop, by
            // looking for a value that's not (-1) by rounding the current index
            // down to successively higher powers of 2.
            for (int32_t power = 0, j = this_row_split;
                 j + (1 << power) < next_row_split; power++) {
              if (j & (1 << power)) {
                j += (1 << power);
                // we know that j is now < next_row_split, because we checked "j
                // + (1<<power) < next_row_split" in the loop condition. Note,
                // we don't want a loop-within-a-loop because of how SIMT
                // works...
                row_ids[j] = i;
              }
            }
          });

      // could do the next line for num_elems+1, but the element at `num_elems`
      // will already be set.
      K2_EVAL(
          c, num_elems, lambda_phase_two, (int32_t j)->void {
            int32_t row_index = row_ids[j];
            if (row_index != -1) return;
            int32_t power = 0, j2 = j;
            for (; row_index != -1; power++) {
              if (j2 & (1 << power)) {
                j2 -= (1 << power);
                row_index = row_ids[j2];
              }
              assert(power < 31);
            }
            row_ids[j] = row_ids[j2];
          });
    }
  }
}


// see declaration in utils.h for documentation.
void RowIdsToRowSplits(ContextPtr c, int32_t num_elems, const int32_t *row_ids,
                       bool no_empty_rows, int32_t num_rows,
                       int32_t *row_splits) {
  NVTX_RANGE(K2_FUNC);
  // process corner case first
  if (num_elems == 0) {
    K2_EVAL(
        c, num_rows + 1, lambda_set_values,
        (int32_t i)->void { row_splits[i] = 0; });
    return;
  }
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    int32_t cur_row = -1;
    for (int32_t i = 0; i < num_elems; i++) {
      int32_t row = row_ids[i];
      K2_CHECK_GE(row, cur_row);
      while (cur_row < row) {
        cur_row++;
        row_splits[cur_row] = i;
      }
    }
    // cur_row  must be >= 0 here as num_elems > 0
    K2_CHECK_GE(cur_row, 0);
    while (cur_row < num_rows) {
      row_splits[++cur_row] = num_elems;
    }
  } else {
    K2_CHECK_EQ(d, kCuda);
    auto lambda_set_row_splits = [=] __device__(int32_t i) {
      if (i == num_rows)
        row_splits[i] = num_elems;
      else
        row_splits[i] = i;
    };
    EvalDevice(c, num_rows + 1, lambda_set_row_splits);

    mgpu::context_t *mgpu_allocator = GetModernGpuAllocator(c);
    mgpu::sorted_search<mgpu::bounds_lower>(
        row_splits, num_rows, row_ids, num_elems, row_splits,
        LessThan<int32_t>(), *mgpu_allocator);
  }
}

}  // namespace k2
