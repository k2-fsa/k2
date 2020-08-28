// k2/csrc/cuda/utils.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/cuda/utils.h"

namespace k2 {


/*

  See declaration of RowSplitsToRowIds() in utils.h.  This is implementation notes.

    Suppose the range we need to fill with a
    particular number (say, x) is from 1010 to 10000 inclusive (binary) The
    first kernel writes x to positions 1010, 1100, 10000; the significance of
    that sequence is we keep adding the smallest number we can add to get
    another zero at the end of the binary representation, until we exceed the
    range we're supposed to fill.  The second kernel: for a given index into x
    that is must fill (say, 1111), it asks "is the index currently here already
    the right one?", which it can test using the function is_valid_index()
    below; if it's not already corret, it searches in a sequence of positions:
    1110, 1100, 1000, 0000, like our sequence above but going downwards, again
    getting more zeros at the end of the binary representation, until it finds
    the correct value in the array at the searched position; then it copies the
    discovered value the original position requested (here, 1111).


    First kernel pseudocode: for each index 'i' into 't', it does:
      for (int n=0, j = t[i]; j < t[i+1]; n++) {
         x[j] = i;
         if (j & (1<<n))  j += (1 << n);
      }
    Second kernel pseudocode: for each element of x, it searches for the right index.  Suppose we're
    given num_indexes == length(n) == length(t) - 1.  Define is_valid_index as follows:
       // returns true if j is the value that we should be putting at position 'i' in x:
       // that is, if t[j] <= i < t[j+1].
       bool is_valid_index(i, j) {
          return (j >= 0 && j < num_indexes && t[j] <= i && i < t[j+1]);
       }
       // We suppose we are given i (the position into x that we're responsible for
       // setting:
       orig_i = i;
       for (int n=0; !is_valid_index(i, x[i]); n++) {
         if (i & (1<<n))  i -= (1 << n);
       }
       x[orig_i] = x[i];
*/
void RowSplitsToRowIds(ContextPtr &c, int32_t num_rows, const int32_t *row_splits,
                       int32_t num_elems, int32_t *row_ids) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    cur_row_start = row_splits[0];
    CHECK_EQ(cur_row_start, 0);
    for (int32_t row = 0; row < num_rows; row++) {
      int32_t next_row_start = row_splits[row+1];
      for (; cur_row_start < next_row_start; ++cur_row_start)
        row_ids[cur_row_start] = row;
    }
    row_ids[num_elems] = num_rows;
  } else {
    // TODO: this uses 3 kernels, which is definitely overkill for cases where num_elems
    // is quite small.  We should write a separate kernel for the case when num_rows+1
    // and num_elems+1 are smaller than some cutoff, e.g. 1024.

    // The following algorithm isn't particularly adapted to GPU hardware in
    // terms of coalesced reads and writes and so on, but it has reasonable
    // asymptotic time complexity (assuming all kernels run in parallel),
    // specifically: O(log(largest(row_splits[i+1]-row_splits[i])))
    assert(d == kCuda);
    auto lambda_init_minus_one = [=] __host__ __device__ (int32_t i) {
                                   row_ids[i] = -1;
                                 };
    Eval(c, num_elems + 1, lambda_init_minus_one);

    auto lambda_phase_one = [=] __host__ __device__ (int32_t i) {
         int32_t this_row_split = row_splits[i],
             next_row_split = (i < num_rows ? row_splits[i+1] : this_row_split+1);
         if (this_row_split < next_row_split)
           row_ids[this_row_split] = i;
         // we have to fill in row_ids[this_row_split], row_ids[this_row_split+1]...
         // row_ids[next_row_split-1] with the same value but that could be a long loop.
         // Instead we write at this_row_split and all indexes this_row_split < i < next_row_split
         // such that i is the result of rounding up this_row_split to  (something)*2^n,
         // for n = 1, 2, 3, ...
         // this will take time logarithmic in (next_row_split - this_row_split).
         // we can then fill in the gaps with a logarithmic-time loop, by looking for a value
         // that's not (-1) by rounding the current index down to successively higher
         // powers of 2.
         for (int32_t power=0, j=this_row_split; j + (1<<power) < next_row_split; power++) {
           if (j & (1<<power)) {
             j += (1 << power);
             // we know that j is now < next_row_split, because we checked "j +
             // (1<<power) < next_row_split" in the loop condition.
             // Note, we don't want a loop-within-a-loop because of how SIMT works...
             row_ids[j] = i;
           }
         }
    };
    Eval(c, num_elems + 1, lambda_phase_one);

    auto lambda_phase_two = [=] __host__ __device__ (int32_t j) {
       int32_t row_index = row_ids[j];
       if (row_index != -1)
         return;
       int32_t power = 0, j2 = j;
       for (; row_index != -1; power++) {
         if (j2 & (1 << power)) {
           j2 -= (1 << power);
           row_index = row_ids[j2];
         }
         assert(power < 31);
       }
       row_ids[j] = row_ids[j2];
    };
    // could do the next line for num_elems+1, but the element at `num_elems`
    // will already be set.
    Eval(c, num_elems, lambda_phase_two);

  }
}


// see declaration in utils.h for documentation.
void RowIdsToRowSplits(ContextPtr &c, int32_t num_elems, const T *row_ids,
                       bool no_empty_rows, int32_t num_rows, T *row_splits) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    int32_t cur_row = 0;
    // NOTE: we could simplify this code by making the row_ids have one extra
    // element, with the last element being equal to num_rows.
    for (int32_t i = 0; i <= num_elems; i++) {
      int32_t row = row_ids[i];
      CHECK_GE(row, cur_row);
      while (cur_row < row) {
        row_splits[cur_row] = i;
        cur_row++;
      }
    }
    row_splits[num_rows] = num_elems;
    assert(row_ids[num_elems] == num_rows);
  } else {
    if (no_empty_rows) {
      auto lambda_simple = [=] __host__ __device__ (int32_t i) {
         int32_t this_row = row_ids[i],
             prev_row = (i == 0 ? -1 : row_ids[i-1]);
         if (this_row > prev_row)
           row_splits[this_row] = i;
      };
      Eval(c, num_elems + 1, lambda_simple);
      return;
    }

    // The following algorithm isn't particularly adapted to GPU hardware in
    // terms of coalesced reads and writes and so on, but it has reasonable
    // asymptotic time complexity (assuming all kernels run in parallel),
    // specifically: O(log(largest(row_split[i+1]-row_split[i])))
    assert(d == kCuda);
    auto lambda_init_minus_one = [=] __host__ __device__ (int32_t i) {
                                   row_splits[i] = -1;
                                 };
    Eval(c, num_rows + 1, lambda_init_minus_one);

   auto lambda_phase_one = [=] __host__ __device__ (int32_t i) {
       int32_t this_row = row_ids[i],
           prev_row = (i == 0 ? -1 : row_ids[i-1]);
       if (this_row > prev_row) {
         K2_CHECK_LE(this_row, num_rows);
         row_splits[this_row] = i;
       }

       // if there are gaps, we also need to ensure that row_splits[prev_row+1]
       // through row_splits[this_row-1] are also set to i.  Since this range
       // could in principle be large, we only write a subset of elements (whose
       // count grows logarithmically with the range).
             next_row_split = row_splits[i+1];
         row_ids[this_row_split] = i;
         for (int32_t power=0, j=this_row_split; j + (1<<power) <= next_row_split; power++) {
           if (j & (1<<power)) {
             j += (1 << power);
             row_ids[j] = i;
           }
         }
      };
    Eval(c, num_elems + 1, lambda_phase_one);

    auto lambda_phase_two = [=] __host__ __device__ (int32_t j) {
         // this kernel finds the locations of any -1's, and aims to find the
         // most recent value that isn't -1 to replace the -1 with.  Due to how
         // we wrote the non(-1) values in lambda_phase_one, we can do this in
         // time that's logarithmic in the length of region that needs to be
         // filled in.  basically we keep turning trailing ones to zeros in the
         // binary representation of 'i' until we find a location that doesn't
         // have -1 in it.
         // Note: this is the same as the lambda_phase_two in
         // RowIdsToRowSplits(), but with different variable names.
         int32_t elem_index = row_splits[j];
         if (elem_index != -1)
           return;
         int32_t power = 0, j2 = j;
         for (; elem_index != -1; power++) {
           if (j2 & (1 << power)) {
             j2 -= (1 << power);
             elem_index = row_splits[j2];
           }
           assert(power < 31);
         }
         row_splits[j] = row_splits[j2];
    };
    // could do the next line for num_rows+1, but the element at `num_rows`
    // will already be set.
    Eval(c, num_rows, lambda_phase_two);
  }
}





}  // namespace k2
