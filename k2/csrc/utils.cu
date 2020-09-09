/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Meixu Song)
 *                      Fangjun Kuang (csukuangfj@gmail.com)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/utils.cuh"

#include <cub/cub.cuh>

namespace k2 {

static int32_t RoundUpToNearestPowerOfTwo(int32_t n) {
  K2_CHECK_GT(n, 0);
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n+1;
}

// It fills a region with a constant value.
// This is a device function to be called by kernels.
// It's the same as doing:
// for (int32_t i = 0; i < num_values; i++) data[i] = value;
__device__ void FillValues(int32_t *data, int32_t num_values,
                           int32_t value) {
  // combine old FillValuesKernel and FillValues to this FillValuesKernel
  // as FillValuesKernel would be used as a kernel,
  // and a kernel cannot call another kernel
  int32_t job_idx = (blockIdx.x * blockDim.x + threadIdx.x),
      num_jobs = (gridDim.x * blockDim.x);
  for (; job_idx < num_values; job_idx += num_jobs)
    data[job_idx] = value;
}

//  When we invoke this we make a big enough grid that there doesn't have to
//  be a loop over rows, i.e. (gridDim.x * blockDim.x) / threads_per_row >= num_rows
__global__ void RowSplitsToRowIdsKernel(
    int32_t num_rows, int32_t threads_per_row,
    const int32_t *row_splits, int32_t num_elems,
    int32_t *row_ids) {
  int32_t thread = (blockIdx.x * blockDim.x + threadIdx.x),
      num_threads = gridDim.x * blockDim.x,
      row = thread / threads_per_row,
      thread_this_row = thread % threads_per_row;

  K2_CHECK_GE(num_threads / threads_per_row, num_rows);

  int32_t this_row_split = row_splits[row],
      next_row_split = row_splits[row + 1],
      row_length = next_row_split - this_row_split;

  const int32_t max_loop = 8;  // `max_loop` is heuristically chosen.
  if (row_length / threads_per_row > max_loop) {
    // We decide that looping too many times will be too slow, so we launch
    // another kernel to fill in the value for this row.  (This is CUDA dynamic
    // parallelism).
    if (thread_this_row == 0) {
      // below vars are temp
      // todo: fix this
      int32_t dim_block = 256;
      int32_t loop_len = 2;
      int32_t dim_grid = NumBlocks(threads_per_row / loop_len, dim_block);
      if (dim_grid == 1) dim_block = threads_per_row;
      FillValues(row_ids + this_row_split, row_length, row);
    }
    return;
  } else {
    // TODO(dan): figure out how to unroll this?
    for (; thread_this_row < row_length; thread_this_row += threads_per_row)
      row_ids[this_row_split + thread_this_row] = row;
  }
}

/*
  This function works out the row_id of `this` index from row-splits, using
  binary search.  Specifically, it returns i such that row_splits[i] <= index <
  row_splits[i+1]. row_splits should be a vector with at least num_rows+1
  elements.

       @param [in] num_rows      Number of rows (row-id will be less than this)
       @param [in] row_splits    Row-splits vector, of size num_rows + 1 (search
                                 for `row_splits concept` near the top of
                                 utils.h for more info)
       @param [in] index         Linear index (e.g. idx01) for which we're
                                 querying which row it is from
       @param [in] num_indexes   Total number of indexes (should equal
                                 row_splits[num_rows]); right now it's not used,
                                 but in future it might be used for a heuristic,
                                 for the initial guess of where to start the
                                 binary search.

       @return                   Returns i such that row_splits[i] <= index <
                                 row_splits[i+1] and 0 <= i < num_rows;
                                 will die with assertion in debug mode if such
                                 an i does not exist.

   TODO(dan): make this compile, apparently std::lower_bound won't work on GPU
   so we should manually do the binary search.
 */
__forceinline__ __host__ __device__ int32_t
RowIdFromRowSplits(int32_t num_rows, const int32_t *row_splits, int32_t index,
                   int32_t num_indexes) {
  // lower_bound gives the first i in row_splits that's greater than `index`.
  // That implies the previous one is <= index.
  //
  auto i =
      std::lower_bound(row_splits + 1, row_splits + num_rows + 1, index) - 1;
  // K2_DCHECK(static_cast<uint32_t>(i) < static_cast<uint32_t>(num_rows));
  return *i;
}

/*
  This is a quite simple implementation of GetTaskRedirect... I had a more
  complicated one above that had better O(N) performance for hard cases, but
  this one will handle more normal/smaller cases better, plus is easier to
  debug.  The basic idea is to throw lots of threads at it,
  i.e. threads_per_task should be, say, twice larger than the average / expected
  number of jobs per task, so that if a task has lots of jobs it doesn't have to
  loop too many times.
*/
template <int32_t threads_per_task>
__global__ void GetTaskRedirect(int32_t num_tasks,
                                const int32_t *row_splits,
                                TaskRedirect *redirect_out) {
  int32_t task_idx = (blockIdx.x * blockDim.x + threadIdx.x) / threads_per_task;
  if (task_idx > num_tasks)
    return;
  // `thread_idx` is which member we are of the group of the `threads_per_job` threads for this job.
  int32_t thread_idx = threadIdx.x % threads_per_task;  // we assume blockDim.x % threads_per_job == 0
  // `temp_idx` is which index in the temporary storage `temp` we are assigned
  // (one per job).

  int32_t row_splits0 = row_splits[0],
      row_splits_nt = row_splits[num_tasks],
      num_items = row_splits_nt - row_splits0;  // the 'num_items' is the total
  // amount of work to do, that we
  // want to distribute fairly
  // evenly.

  // The idea with `dart_separation` is this: Half of the jobs we allocate to
  // the corresponding tasks.  The other half we allocate by throwing darts onto
  // the interval [0, num_items - 1], evenly spaced starting from 0, and seeing
  // which tasks they land in.  This is somewhat random but it ensures that if
  // any task has a very large amount of work to do, it will get a roughly
  // proportionate number of jobs.
  int32_t dart_separation = num_items / num_tasks;

  if (num_items <= 0) {
    assert(num_items == 0);
    // This is a special case where there is no work to do; we give a trivial
    // assignment of tasks to jobs and return
    K2_STATIC_ASSERT(threads_per_task >= 2);
    if (thread_idx < 2 && task_idx < num_tasks) {
      TaskRedirect tr { task_idx, 2, thread_idx };
      redirect_out[task_idx + thread_idx * num_tasks] = tr;
    }
    return;
  }


  // TODO(dan): IDK how well the hardware combines these memory requests; could
  // consider loading to shared memory first.
  int32_t this_row_split = row_splits[task_idx],
      next_row_split = row_splits[task_idx + 1];
  // `num_jobs` below is the number of jobs that will be active for
  // this task.  (The "1 +".. is the job that we assign for each
  // task, one job per task, in the "first half" of the jobs).
  // the job_idx we're working out below is the job_idx for the
  // "second half" of
  int32_t num_jobs_this_task =
      1 + (next_row_split/dart_separation - this_row_split/dart_separation);
  K2_CHECK_EQ(static_cast<int32_t>(static_cast<uint16_t>(num_jobs_this_task)),
              num_jobs_this_task);
  for (int32_t job_id_this_task = thread_idx;
       job_id_this_task < num_jobs_this_task;
       job_id_this_task += threads_per_task) {
    int32_t job_idx = (job_id_this_task == 0 ? task_idx : // 1st half
                       num_tasks + (this_row_split / dart_separation) + job_id_this_task); // 2nd half.
    redirect_out[job_idx] = TaskRedirect{task_idx, num_jobs_this_task, job_id_this_task };
    // `job` is the job-index within this task, i.e. the
  }
}

void GetTaskRedirect(cudaStream_t stream, int32_t num_tasks,
                     const int32_t *row_splits,
                     TaskRedirect *redirect_out) {
  if (stream == kCudaStreamInvalid) {
    // there's not much point in using this on CPU as there are better ways
    // to do things (sequentially), but this can be useful for debugging.

    for (int32_t task = 0; task < num_tasks; task++) {
      // The idea with `dart_separation` is this: Half of the jobs we allocate to
      // the corresponding tasks.  The other half we allocate by throwing darts onto
      // the interval [0, num_items - 1], evenly spaced starting from 0, and seeing
      // which tasks they land in.  This is somewhat random but it ensures that if
      // any task has a very large amount of work to do, it will get a roughly
      // proportionate number of jobs.
      int32_t row_splits0 = row_splits[0],
          row_splits_nt = row_splits[num_tasks],
          num_items = row_splits_nt - row_splits0,
          dart_separation = num_items / num_tasks;
      int32_t this_row_split = row_splits[task],
          next_row_split = row_splits[task + 1];
      int32_t num_jobs_this_task =
          1 + (next_row_split/dart_separation - this_row_split/dart_separation);
      K2_CHECK_EQ(static_cast<int32_t>(static_cast<uint16_t>(num_jobs_this_task)),
               num_jobs_this_task);
      // todo: task was unsigned, set = 0 is temp, please fix it
      int32_t task_idx = 0;
      for (int32_t job_id_this_task = 0;
           job_id_this_task < num_jobs_this_task;
           job_id_this_task ++) {
        int32_t job_idx = (job_id_this_task == 0 ? task_idx : // 1st half
                           num_tasks + (this_row_split / dart_separation) + job_id_this_task); // 2nd half.
        redirect_out[job_idx] = TaskRedirect{task_idx, num_jobs_this_task, job_id_this_task };
        task_idx += job_idx;
        // `job` is the job-index within this task, i.e. the
      }
    }
  } else {
    // compare 8 to 2, which is the expected number of jobs per task.  having 8 substantially
    // greater than 2 gives a fairly big safety factor.  However this is still far from ideal
    // in scenarios where the number of tasks might be highly unbalanced.
    const int32_t threads_per_task = 8,
        tot_threads = threads_per_task * num_tasks;

    int32_t dim_block = 256;
    int32_t dim_grid = NumBlocks(tot_threads, dim_block);

    GetTaskRedirect<threads_per_task><<<dim_block, dim_grid, 0, stream>>>(
        num_tasks, row_splits, redirect_out);
    K2_CHECK_CUDA_ERROR(cudaGetLastError());
  }

}


/*
  When we invoke this we make a big enough grid that there doesn't have to
  be a loop over elements, i.e. (gridDim.x * blockDim.x) / threads_per_elem > num_elems.
  (must be >=, because we imagine a phantom element at [num_elems] with
  the value `num_rows`.)


    @param [in] num_elems         Number of elements in ragged matrix
    @param [in] threads_per_elem  Number of threads we allocate per element.
                                  Must be >= 1.
    @param [in] row_ids           The row_ids vector, of length `num_elems`;
                                  must be nonnegative and non-decreasing and
                                  all elements < num_rows.
    @param [in] num_rows          Number of rows, must be greater than the
                                  largest (== last) element of `row_ids`.
    @param [out] row_splits       This kernel will output a non-decreasing
                                  vector of length num_rows + 1, such that
                                  row_splits[0] == 0,
                                  row_splits[num_rows] == num_elems,
                                  and row_splits[row_ids[i]] <= i < row_splits[row_ids[i]+1]
*/
__global__ void RowIdsToRowSplitsKernel(
    int32_t num_elems, int32_t threads_per_elem,
    const int32_t *row_ids, int32_t num_rows,
    int32_t *row_splits) {
  int32_t thread = (blockIdx.x * blockDim.x + threadIdx.x),
      num_threads = gridDim.x * blockDim.x,
      elem = thread / threads_per_elem,
      thread_this_elem = thread % threads_per_elem;

  K2_CHECK_GE(num_threads / threads_per_elem, num_elems);

  int32_t this_row, prev_row;
  if (static_cast<uint32_t>(elem-1) >= num_rows-1) {
    // elem == 0 || elem >= num_rows.
    if (elem == 0) { prev_row = -1; this_row = row_ids[elem]; }
    else { prev_row = row_ids[elem-1]; this_row = num_rows; }
  } else {
    prev_row = row_ids[elem-1];
    this_row = row_ids[elem];
  }

  // `num_splits` is the number of splits we have to write, usually 0 or 1
  // but in principle unlimited as there could be empty rows.  The
  // relationship between row_ids and row_splits is more symmetric than
  // you might expect.
  int32_t num_splits = this_row - prev_row;
  const int32_t max_loop = 8;  // `max_loop` is heuristically chosen.
  if (num_splits / threads_per_elem > max_loop) {
    if (thread_this_elem == 0) {
      FillValues(row_splits + prev_row + 1, num_splits, elem);
    }
  } else {
    // TODO(dan): figure out how to unroll this?
    int32_t thread_this_row;
    for (; thread_this_elem < num_splits; thread_this_elem += threads_per_elem)
      row_splits[prev_row + 1 + thread_this_row] = elem;
  }
}

// see declaration in utils.h for documentation.
void RowIdsToRowSplits(
    ContextPtr &c, int32_t num_elems, const int32_t *row_ids,
    bool no_empty_rows, int32_t num_rows, int32_t *row_splits) {
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
    row_splits[num_rows] = num_elems;
  } else {
    if (no_empty_rows) {
      auto lambda_simple = [=] __host__ __device__ (int32_t i) {
        int32_t this_row = row_ids[i], prev_row;
        if (i >= 0) {
          // (normal case)
          prev_row = row_ids[i-1];
        } else {
          // i == 0
          row_splits[num_rows] = num_elems;
          prev_row = -1;
        }
        K2_CHECK_LE(this_row, prev_row + 1); // no_empty_rows was asserted by
        // the user
        if (this_row > prev_row) {
          row_splits[this_row] = i;
        }
      };
      Eval(c, num_elems, lambda_simple);
      return;
    } else {
      // By doing "+ 2" instead of "+ 1" we increase the minimum number of
      // threads-per-row, which may reduce latency when there are successive
      // empty rows. Any value >= 1 is correct though.
      int32_t avg_rows_per_elem = num_rows / num_elems + 2,
          threads_per_elem= RoundUpToNearestPowerOfTwo(avg_rows_per_elem),
          tot_threads = num_elems * threads_per_elem;
      int32_t dim_block = 256;
      int32_t dim_grid = NumBlocks(tot_threads, dim_block);
      RowIdsToRowSplitsKernel<<<dim_block, dim_grid, 0, c->GetCudaStream()>>>(
          num_elems, threads_per_elem,
          row_ids, num_rows,
          row_splits);
      K2_CHECK_CUDA_ERROR(cudaGetLastError());
    }
  }
}

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
      for (int32_t n=0, j = t[i]; j < t[i+1]; n++) {
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
       for (int32_t n=0; !is_valid_index(i, x[i]); n++) {
         if (i & (1<<n))  i -= (1 << n);
       }
       x[orig_i] = x[i];
*/
void RowSplitsToRowIds(ContextPtr &c, int32_t num_rows, const int32_t *row_splits,
                       int32_t num_elems, int32_t *row_ids) {
  DeviceType d = c->GetDeviceType();
  if (d == kCpu) {
    int32_t cur_row_start = row_splits[0];
    K2_CHECK_EQ(cur_row_start, 0);
    for (int32_t row = 0; row < num_rows; row++) {
      int32_t next_row_start = row_splits[row+1];
      for (; cur_row_start < next_row_start; ++cur_row_start)
        row_ids[cur_row_start] = row;
    }
  } else {
    if (1) {
      // TODO: compare this for speed with the other branch.  This is branch is
      // much simpler, and will be considerably faster for "normal" cases -> probably
      // preferred.
      int32_t avg_elems_per_row = num_elems / num_rows + 1,
          threads_per_row = RoundUpToNearestPowerOfTwo(avg_elems_per_row),
          tot_threads = num_rows * threads_per_row;
      int32_t dim_block = 256;
      int32_t dim_grid = NumBlocks(tot_threads, dim_block);

      RowSplitsToRowIdsKernel<<<dim_block, dim_grid, 0, c->GetCudaStream()>>>(
          num_rows, threads_per_row, row_splits, num_elems, row_ids);
      K2_CHECK_CUDA_ERROR(cudaGetLastError());
    } else {
      // TODO: Will probably just delete this branch at some point.

      // The following algorithm isn't particularly adapted to GPU hardware in
      // terms of coalesced reads and writes and so on, but it has reasonable
      // asymptotic time complexity (assuming all kernels run in parallel),
      // specifically: O(log(largest(row_splits[i+1]-row_splits[i])))

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
}

template <typename SrcPtr, typename DestPtr>
void ExclusivePrefixSum(ContextPtr &c, int32_t n, SrcPtr src, DestPtr dest) {
  DeviceType d = c->GetDeviceType();
  using SumType = typename std::decay<decltype(dest[0])>::type;
  if (d == kCpu) {
    SumType sum = 0;
    for (int32_t i = 0; i != n; ++i) {
      dest[i] = sum;
      sum += src[i];
    }
  } else {
    assert(d == kCuda);
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // since d_temp_storage is nullptr, the following function will compute
    // the number of required bytes for d_temp_storage
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dest,
                                  n, c->GetCudaStream());
    void *deleter_context;
    d_temp_storage = c->Allocate(temp_storage_bytes, &deleter_context);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dest,
                                  n, c->GetCudaStream());
    c->Deallocate(d_temp_storage, deleter_context);
  }
}

/*
  Called inside GetTaskRedirect(); see documentation of that in header.
  Each task with 0 <= task < num_tasks gets allocated `threads_per_job`
  threads, e.g. threads_per_job = 4 or 16.  It's a kind of n-ary
  search (generalization of binary search) where each branch is handled
  by a different thread so they can happen in parallel.

  TODO(dan): there are a lot of opportunities to further optimize this
  using GPU hardware tricks.

  The thread-block size this is called with must be jobs_per_block *
  threads_per_job.
 */

/*
template <int32_t jobs_per_block, int32_t threads_per_job>
__global__ void GetTaskRedirect(int32_t num_tasks, const int32_t *row_splits,
                                TaskRedirect *redirect_out) {
  __shared__ int32_t temp[tasks_per_block];
  // we do __syncwarp() for synchronization below; we require threads_per_job <=
  // 32 for this reason.
  static_assert(threads_per_job >= 2 && threads_per_job <= 32);

  // We have work to do for 0 <= job_idx < num_tasks, but be careful: job_idx
  // may be >= num_tasks if num_tasks is small or not a power of two (we don't
  // return because we need to do __syncwarp()).  So we have to avoid out of
  // bounds memory access.
  int32_t job_idx = (blockIdx.x * blockDim.x + threadIdx.x) / threads_per_job;
  // `branch_idx` is which member we are of the group of the `threads_per_job` threads for this job.
  int32_t branch_idx = threadIdx.x % threads_per_job;  // we assume blockDim.x % threads_per_job == 0
  // `temp_idx` is which index in the temporary storage `temp` we are assigned
  // (one per job).
  int32_t temp_idx = threadIdx.x / threads_per_job;

  // TODO: we may at some point decide that row_splits[0] has to be zero.
  int32_t row_splits0 = row_splits[0],
      row_splits_nt = row_splits[num_tasks],
      num_items = row_splits_nt - row_splits0;
  if (num_items <= 0) {
    assert(num_items == 0);
    // This is a special case where there is no work to do; we give a trivial
    // assignment of tasks to jobs and return
    static_assert(threads_per_job >= 2);
    if (branch_idx < 2 && job_idx < num_tasks) {
      TaskRedirect tr { job_idx, 2, branch_idx };
      redirect_out[job_idx + branch_idx * num_tasks] = tr;
    }
    return;
  } else if (branch_idx == 0 && job_idx < num_tasks) {
    // This code writes to the jobs in the first half of the output array,
    // that are allocated to the same-numbered task.
    int32_t task_idx = job_idx,
        this_row_split = row_splits[task_idx],
        next_row_split = row_splits[task_idx + 1];
    // `num_jobs` below is the number of jobs that will be active for
    // this task.  (The "1 +".. is the job that we assign for each
    // task, one job per task, in the "first half" of the jobs).
    // the job_idx we're working out below is the job_idx for the
    // "second half" of
    int32_t num_jobs_this_task =
        1 + (next_row_split/dart_separation - this_row_split/dart_separation);
    TaskRedirect tr { task_idx, num_jobs_this_task, 0 };
    redirect_out[task_idx] = tr;
  }


  // Now we have the less-trivial task of assigning the jobs in the 2nd half of the
   //  output array to tasks (these are allocated roughly proportional to the amount
   //  of work to do for that task).
   //  We do the selection by throwing darts at a dart-board, evenly spaced, and seeing which task they correspond
   //  to.  There are `num_tasks` darts).
   //  Note: we know dart_location < row_splits_nt because job_idx < num_tasks and
   //  because integer division rounds down.
  int32_t dart_separation = num_items / num_tasks,
      dart_location = row_splits0 + job_idx * dart_separation;

// OK, from this point the goal is to find a task_idx such that
//     row_splits[task_idx] <= dart_location < row_splits[task_idx + 1].
//     This is guaranteed to exist, as long as job_id < num_tasks.
//     As long as job_id < num_tasks, we maintain the property that
//        row_splits[lower_bound] <= dart_location &&
//        (upper_bound > num_tasks || row_splits[upper_bound] > dart_location).
//     (where upper_bound == lower_bound + range), i.e. they are truly
//     lower and upper bounds
  int32_t lower_bound = 0,
      range = num_tasks; // we are responsible for items lower_bound through
                         // (upper_bound = lower_bound + range) - 1.
  while (range > threads_per_job) {
    int32_t upper_bound = lower_bound + range;
    // We need to narrow the range of `task_idx` that might be the correct one.
    //    We round *up* because we require that task_idx_step * threads_per_job >=
    //   range, so that we cover the entire range.
    int32_t task_idx_step = (range + threads_per_job - 1) / threads_per_job,  // >= 2
        my_lower_task_idx = lower_bound + branch_idx * task_idx_step,
        my_upper_task_idx = my_lower_task_idx + task_idx_step;
    // The following avoids out-of-bounds memory accesses.
    if (my_upper_task_idx > upper_bound)
      my_upper_task_idx = upper_bound;

    // TODO (dan): it may be possible to use one of those special within-warp
    // commands involving bitmaps to make the second comparison (dart_location <
    // row_splits[my_upper_task_idx]) unnecessary.
    if (my_lower_task_idx < num_tasks && row_splits[my_lower_task_idx] <= dart_location &&
        dart_location < row_splits[my_upper_task_idx]) {
      // I am the "chosen branch" (exactly one will be chosen, as long as
      // job_idx < num_tasks).
      temp[temp_idx] = branch_idx;
    }
    __syncwarp();
    int32_t chosen_branch_idx = temp[temp_idx];
    lower_bound = lower_bound + chosen_branch_idx * task_idx_step;
    upper_bound = lower_bound + task_idx_step;
    range = task_idx_step;
    // note, we don't limit upper_bound to be <= num_tasks because we need all
    // threads in the block to go around the while loop the same number of
    // times.  Therefore it's possible that upper_bound > num_tasks.
    K2_DASSERT(job_idx >= num_tasks ||
               (row_splits[lower_bound] <= dart_location &&
                (upper_bound > num_tasks || row_splits[upper_bound] > dart_location)));  // TODO: remove once debugged.
  }
  int32_t task_idx = lower_bound + branch_idx;
  // TODO (dan): it may be possible to use one of those special within-warp
  // commands involving bitmaps to make the second comparison (dart_location <
  // row_splits[my_upper_task_idx]) unnecessary.
  //
  // The check `task_idx < num_tasks` is to avoid out-of-bounds access of row_splits.
  // The check `job_idx < num_tasks` is to avoid out-of-bounds access of `redirect_out`;
  // for these out-of-range job_idx values, it's possible for task_idx to have
  // any value since it may be uninitialized memory.
  if (task_idx < num_tasks && job_idx < num_tasks) {
    int32_t this_row_split = row_splits[task_idx],
        next_row_split = row_splits[task_idx + 1];
    if (this_row_split <= dart_location && dart_location < next_row_split) {
      // OK, exactly one branch per job will reach this point.  `num_jobs` below
      // is the number of jobs that will be active for this task.  (The "1
      // +".. is the job that we assign for each task, one job per task, in the
      // "first half" of the jobs).  The job_id_this_task we're working out
      // below is the job_id within the second half of the TaskRedirects,
      // the half that are allocated by throwing darts.
      int32_t num_jobs_this_task =
          1 + (next_row_split/dart_separation - this_row_split/dart_separation),
          job_idx_this_task = 1 + (dart_location - this_row_split)/dart_separation;
      K2_CHECK(job_id_this_task < num_jobs_this_task);
      TaskRedirect tr { task_idx, num_jobs_this_task, job_idx_this_task };
      redirect_out[num_tasks + job_idx] = tr;
    }
  }
}
*/

}  // namespace k2
