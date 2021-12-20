/**
 * @copyright
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
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

#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>

#include "k2/csrc/utils.h"  // for LogAdd
#include "k2/python/csrc/torch/mutual_information.h"

namespace k2 {

/*
  Forward of mutual_information.  Each thread block computes blocks of the 'p'
  array of (s, t) shape equal to (BLOCK_SIZE, BLOCK_SIZE), e.g. (32, 32).
  Thread-blocks loop over such blocks, but they might loop only once if there is
  not that much data to process.  We sequentially launch thread groups in
  such a way that thread-blocks within a group do not depend on each other
  (see the "iter" parameter).  The blocks of the 'image' (i.e. of the p matrix)
  that each group handles are arranged in a diagonal.

  Template args:
      scalar_t: the floating-point type, e.g. float, double; maybe eventually
                half, although I think we don't support LogAdd for half yet.
      BLOCK_SIZE: an integer power of two no greater than 32 (this limitation
                is because we assume BLOCK_SIZE + 1 <= 64 in some data-loading
                code).
  Args:
      px:     Tensor of shape [B][S][T + 1]; contains the log-odds ratio of
              generating the next x in the sequence, i.e.
              xy[b][s][t] is the log of
                 p(x_s | x_0..x_{s-1}, y_0..y_{s-1}) / p(x_s),
             i.e. the log-prob of generating x_s given subsequences of lengths
             (s, t), divided by the prior probability of generating x_s.  (See
              mutual_information.py for more info).
      py:     The log-odds ratio of generating the next y in the sequence.
              Shape [B][S + 1][T]
      p:      This function writes to p[b][s][t] the mutual information between
              sub-sequences of x and y of length s and t respectively, from the
              b'th sequences in the batch.  Its shape is [B][S + 1][T + 1].
              Concretely, this function implements the following recursion,
              in the case where s_begin == t_begin == 0:

               p[b,0,0] = 0.0
               p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                  p[b,s,t-1] + py[b,s,t-1])          (eq. 0)
                       if s > 0 or t > 0,
                       treating values with any -1 index as -infinity.
              .. if `boundary` is set, we start fom p[b,s_begin,t_begin]=0.0.
   boundary:  If set, a tensor of shape [B][4] of type int64_t, which
              contains, where for each batch element b, boundary[b] equals
              [s_begin, t_begin, s_end, t_end]
              which are the beginning and end (i.e. one-past-the-last) of the
              x and y sequences that we should process.  Otherwise, must be
              a tensor of shape [0][0] of type int64_t; the values will
              default to (0, 0, S, T).
     ans: a tensor `ans` of shape [B], where this function will set
            ans[b] = p[b][s_end][t_end],
            with s_end and t_end being (S, T) if `boundary` was specified,
            and (boundary[b][2], boundary[b][3]) otherwise.
            `ans` represents the mutual information between each pair of
            sequences (i.e. x[b] and y[b], although the sequences are not
            supplied directy to this function).

   The block-dim and grid-dim must both be 1-dimensional, and the block-dim must
   be at least 128.
*/
template <typename scalar_t,
          int BLOCK_SIZE>  // e.g. BLOCK_SIZE == 16 or 32.
__global__ void mutual_information_kernel(
    // B, S, T + 1, i.e. batch, x_seq_length, y_seq_length + 1
    torch::PackedTensorAccessor32<scalar_t, 3> px,
    torch::PackedTensorAccessor32<scalar_t, 3> py,  // B, S + 1, T.
    // B, S + 1, T + 1.  This is an output.
    torch::PackedTensorAccessor32<scalar_t, 3> p,
    // B, 4;  or 0, 0 if boundaries are the defaults (0, 0, S, T)
    torch::PackedTensorAccessor32<int64_t, 2> boundary,
    torch::PackedTensorAccessor32<scalar_t, 1> ans,  // [B]
    int iter) {  // This kernel is sequentially called with 'iter' = 0, 1, 2 and
                 // so on, up to num_iters - 1 where num_iters = num_s_blocks +
                 // num_t_blocks - 1 num_s_blocks = S / BLOCK_SIZE + 1
                 // num_t_blocks = T / BLOCK_SIZE + 1
                 // so that each group depends on the previous group...
  const int B = px.size(0), S = px.size(1), T = py.size(2);
  // num_s_blocks and num_t_blocks are the number of blocks we need to cover the
  // array of size (S, T) with blocks of this size, in the s and t directions
  // respectively.
  // You can read the following expressions as simplifications of, for example,
  // num_s_blocks = ((S + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
  // i.e. rounding-up division of (S + 1) by BLOCK_SIZE, and the same for (T +
  // 1).
  const int num_s_blocks = S / BLOCK_SIZE + 1;
  //, num_t_blocks = T / BLOCK_SIZE + 1;

  // num_blocks_this_iter is an upper bound on the number of blocks of size
  // (BLOCK_SIZE by BLOCK_SIZE) that might be active on this iteration (`iter`).
  // These iterations start from the bottom left of the image so that on iter ==
  // 0 we process only one block with block-index (0, 0) then on iter == 1 we
  // process block-indexes (1, 0) and (0, 1); and then on iter==2 we process (2,
  // 0), (1, 1) and (0, 2); and so on.  We also will never have more than
  // `num_s_blocks` blocks (We'll never have more than num_t_blocks either, but
  // the numbering we use corresponds to s and not t, so when we hit the
  // num_t_blocks limit, the blocks with the lowest s indexes would just not be
  // active and we'll 'continue' in the loop below).
  int num_blocks_this_iter = min(iter + 1, num_s_blocks);

  // For the block with s_block_begin == 0 and t_block_begin == 0 (for
  // easy illustration), px_buf[s][t] will contain exp(px[s - 1][t]); or 0
  // for out-of-range indexes into px.
  // Likewise, py_buf[s][t] will contain exp(py[s][t - 1]).
  __shared__ scalar_t px_buf[BLOCK_SIZE][BLOCK_SIZE],
      py_buf[BLOCK_SIZE][BLOCK_SIZE];

  // p_buf[s][t] == p[s+s_block_begin-1][t+t_block_begin-1]
  // 1st row/col of p_buf correspond to the previously computed blocks (lower
  // `iter`), or to negative indexes into p.  So, for the origin block,
  // p_buf[s][t] corresponds to p[s - 1][t - 1]; or -inf for
  // out-of-range values.
  __shared__ scalar_t p_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied; or (0, 0, S, T) otherwise.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = T;
  }

  // batch_block_iter iterates over batch elements (index b) and block
  // indexes in the range [0..num_blocks_this_iter-1], combining both
  // batch and block indexes.
  for (int batch_block_iter = blockIdx.x;
       batch_block_iter < B * num_blocks_this_iter;
       batch_block_iter += gridDim.x) {
    int block = batch_block_iter / B,
        b = batch_block_iter % B;  // b is the index into the batch

    // Note: `block` can be no greater than `iter` because num_blocks_this_iter
    // <= iter + 1, i.e. iter >= num_blocks_this_iter - 1; and
    // block < num_blocks_this_iter, so iter - block >= 0.
    int s_block_begin = block * BLOCK_SIZE,
        t_block_begin = (iter - block) * BLOCK_SIZE;
    bool is_origin_block = (s_block_begin + t_block_begin == 0);

    __syncthreads();

    if (boundary.size(0) != 0 && threadIdx.x < 4)
      boundary_buf[threadIdx.x] = boundary[b][threadIdx.x];

    __syncthreads();

    int s_begin = boundary_buf[0], t_begin = boundary_buf[1],
        s_end = boundary_buf[2], t_end = boundary_buf[3];

    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block (the block of `p`
    // that we will write), no greater than (BLOCK_SIZE, BLOCK_SIZE) but
    // possibly less than that if we are towards the end of the sequence.  The
    // last element in the output matrix p that we need to write is (s_end,
    // t_end), i.e. the one-past-the-end index is (s_end + 1, t_end + 1).
    int block_S = min(BLOCK_SIZE, s_end + 1 - s_block_begin),
        block_T = min(BLOCK_SIZE, t_end + 1 - t_block_begin);

    if (block_S <= 0 || block_T <= 0) continue;

    // Load px_buf and py_buf.  We exponentiate; the assumption is that they
    // most likely won't overflow or underflow, but if they do overflow we'll
    // detect it later; we'll also detect certain kinds of underflow.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // comparing as unsigned int makes sure the index is nonnegative.
      // Caution: if s_begin > 0 or t_begin > 0 we may end up loading some px
      // and py values that are outside the proper boundaries that we need, but
      // the corresponding p_buf values will end up being 0 so this won't
      // matter.
      scalar_t this_px = -INFINITY;
      if (s > s_begin && s <= s_end && t <= t_end) this_px = px[b][s - 1][t];
      px_buf[s_in_block][t_in_block] = this_px;

      scalar_t this_py = -INFINITY;
      if (t > t_begin && t <= t_end && s <= s_end) this_py = py[b][s][t - 1];
      py_buf[s_in_block][t_in_block] = this_py;
    }

    // Load the 1st row and 1st column of p_buf (except element[0][0] is not
    // needed).  This is the context from previously computed blocks of the
    // image.  Remember: p_buf[s][t] will correspond to p[s + s_block_begin -
    // 1][t + t_block_begin - 1]
    if (threadIdx.x <= BLOCK_SIZE) {
      // s_in_p_buf are simply the indexes into p_buf
      int s_in_p_buf = threadIdx.x, t_in_p_buf = 0,
          s = s_in_p_buf + s_block_begin - 1,
          t = t_in_p_buf + t_block_begin - 1;

      scalar_t this_p = -INFINITY;
      if (s >= s_begin && s <= s_end && t >= t_begin && t <= t_end)
        this_p = p[b][s][t];
      /*printf("p[%d][%d][%d] = %f, threadIdx.x = %d, px = %f, py = %f\n", b, s,
         t, (float)this_p, (int)threadIdx.x,
          (float)px_buf[s_in_p_buf][t_in_p_buf],
         (float)py_buf[s_in_p_buf][t_in_p_buf]); */
      p_buf[s_in_p_buf][t_in_p_buf] = this_p;
    } else if (static_cast<unsigned int>(static_cast<int>(threadIdx.x) - 64) <=
               static_cast<unsigned int>(BLOCK_SIZE)) {
      // Another warp handles the other leg.  Checking as unsigned
      // tests that threadIdx.x - 64 is both >= 0 and <= BLOCK_SIZE
      int s_in_p_buf = 0, t_in_p_buf = static_cast<int>(threadIdx.x) - 64,
          s = s_in_p_buf + s_block_begin - 1,
          t = t_in_p_buf + t_block_begin - 1;

      scalar_t this_p = -INFINITY;
      if (s >= s_begin && s <= s_end && t >= t_begin && t <= t_end)
        this_p = p[b][s][t];
      /*printf("p[%d][%d][%d] = %f, threadIdx.x = %d, px = %f, py = %f\n", b, s,
        t, (float)this_p, (int)threadIdx.x,
        (float)px_buf[s_in_p_buf][t_in_p_buf],
        (float)py_buf[s_in_p_buf][t_in_p_buf]);*/
      p_buf[s_in_p_buf][t_in_p_buf] = this_p;
    }

    __syncthreads();

    // from here to the next __syncthreads(), only the 1st warp should be active
    // so we shouldn't need to synchronize.  (implicit within-warp
    // synchronization).

    if (threadIdx.x == 0) {
      // This if-statement is an optimization and modification of the loop below
      // for the value i == 0, i.e. inner-iteration == 0.  The modification is
      // to set p_buf to 1.0 = exp(0.0) if this is the "origin block",
      // i.e. s == s_begin, t == t_begin.  This corresponds to the
      // probability of the pair of sequences of length (0, 0).
      p_buf[1][1] =
          (is_origin_block ? 0.0
                           : LogAdd<scalar_t>()(p_buf[0][1] + px_buf[0][0],
                                                p_buf[1][0] + py_buf[0][0]));
    }

    scalar_t p_buf_s1_t;  // This is for an optimization to avoid one
                          // shared-memory read/write in the loop below.  it
                          // represents p_buf[s + 1][t]; the first time we
                          // access this, it will be for t == 0, except for
                          // thread 0 when we first need it for t == 1.
    if (threadIdx.x < BLOCK_SIZE) {
      int s = threadIdx.x;
      p_buf_s1_t = p_buf[s + 1][threadIdx.x == 0 ? 1 : 0];
    }

    int s = threadIdx.x;
    for (int i = 1; i < block_S + block_T - 1; ++i) {
      __syncwarp();
      // i is the inner iteration, which corresponds to the (s + t) indexes of
      // the elements within the block that we write.  So i == 0 writes
      // positions (s, t) == (0, 0) (but we treated i == 0 as a special case
      // above); i == 1 writes (0, 1) and (1, 0); i == 2 writes (0, 2), (1, 1)
      // and (2, 1); and so on.  Note: not many threads participate in this
      // part, only up to BLOCK_SIZE at most.  Unfortunately we couldn't figure
      // out a very meaningful way for more threads to do work, that looked like
      // it would really spead things up.
      // So this kernel does (2 * BLOCK_SIZE) iterations, which may seem a lot,
      // but we do at least do the I/O in an efficient way and keep the
      // inner loop simple and fast (e.g. no exp() or log()).
      int t = i - s;
      if (s < block_S &&
          static_cast<unsigned int>(t) < static_cast<unsigned int>(block_T)) {
        // p_buf is indexed by s + 1 and t + 1 because it has an extra initial
        // row and column for context from previous blocks.  Taking into account
        // the way these buffers relate to the tensors p, px and py,
        // can be interpreted as follows,
        // writing sbb for s_block_begin and tbb for t_block_begin:
        //
        //   p[b][s+sbb][t+tbb] = LogAdd(p[b][s+sbb-1][t+tbb] +
        //   px[s+sbb-1][t+tbb],
        //                               p[b][s+sbb][t+tbb-1] +
        //                               py[s+sbb][t+tbb-1]
        //
        // where you can see that apart from the offsets of tbb and sbb, this is
        // the same as the recursion defined for p in
        // mutual_information.py:mutual_information_recursion(); and (eq. 0)
        // above.
#if 0
        p_buf[s + 1][t + 1] = LogAdd<scalar_t>()(
            p_buf[s][t + 1] + px_buf[s][t], p_buf[s + 1][t] + py_buf[s][t]);

        /*printf("threadIdx.x = %d, i = %d, s = %d, t = %d, p_buf[s+1][t+1] =
          %f, p_buf[s][t+1] = %f, " "px_buf[s][t] = %f, p_buf[s + 1][t] = %f,
          py_buf[s][t] = %f\n", (int)threadIdx.x, i, s, t,
          (float)p_buf[s+1][t+1], (float)p_buf[s][t+1], (float)px_buf[s][t],
          (float)p_buf[s+1][t], (float)py_buf[s][t]);*/
#else
        // This is an optimization of the statement above (the other half of
        // this #if/#else) where we keep p_buf[s + 1][t] in a register to avoid
        // the need for a load from shared memory.
        p_buf_s1_t = LogAdd<scalar_t>()(p_buf[s][t + 1] + px_buf[s][t],
                                        p_buf_s1_t + py_buf[s][t]);
        // The next time this thread reads p_buf_s1_t, t will be one greater,
        // so p_buf_s1_t will contain p_buf[s + 1][t].  The first time this
        // thread uses p_buf_s1_t is when t == 0, except for thread 0 where
        // the 1st item accessed is for s == 0, t == 1.
        p_buf[s + 1][t + 1] = p_buf_s1_t;
#endif
        // We don't need to do __syncthreads() in this loop because all the
        // threads that are active are in the same warp.  (However, in future,
        // if NVidia changes some things, we might need to sync here).
      }
    }
    __syncthreads();

    // Write out the data to p; check that nothing has gone out of numerical
    // range, and write 'panic' flag if it has.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      if (s_in_block < block_S && t_in_block < block_T) {
        scalar_t this_p = p_buf[s_in_block + 1][t_in_block + 1];
        p[b][s][t] = this_p;
      }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      // Write `ans`, if this is the final (top-right) block in its sequence
      // Logically, the following equation corresponds to:
      //   ans[b] = p[b][s_end][t_end]
      if (s_block_begin + block_S - 1 == s_end &&
          t_block_begin + block_T - 1 == t_end) {
        // you could read block_S below as block_S - 1 + 1, meaning,
        // it's the last index in a block of size block_S, but the indexes into
        // p_buf have a "+ 1".  Likewise for block_T.
        ans[b] = p_buf[block_S][block_T];
      }
    }
  }
}

/*
  Backward of mutual_information.

  If we were to write the forward pass in non-log space, it would be (ignoring
  edge cases), as follows... we'll prefix all the variable names with e, e.g.
ep, to clarify that it's the exp of the actual argument p:

         ep[b][s][t] = ep[b][s - 1][t] * epx[b][s - 1][t] +
                       ep[b][s][t - 1] * epy[b][s][t - 1].    (eq. 1)

(A)
  First we consider the part of the backprop that requires recursion or
iteration, i.e. the part involving only gradients of ep.
 This is: ep_grad[b][s - 1][t] += ep_grad[b][s][t] * epx[b][s - 1][t]
          ep_grad[b][s][t - 1] += ep_grad[b][s][t] * epy[b][s][t - 1].

  .. and if we add 1 to the s index of the first equation above and 1 to the
     t index of the second equation, we can see that:

          ep_grad[b][s][t] = ep_grad[b][s + 1][t] * epx[b][s][t] +
                             ep_grad[b][s][t + 1] * epy[b][s][t].

  Now, if ep = exp(p), and y is the loss function we are backprop'ing,
  then ep_grad == dy/dep == dy/dp
       dp/dep == dy/dp / (dep/dp) == dy/dp / exp(p)
              == dy/dp / ep.  == p_grad / ep.
       I.e. ep_grad = p_grad / ep.

  So we can write the above as:
      p_grad[b][s][t] / ep[b][s][t]
          = p_grad[b][s + 1][t] / ep[b][s + 1][t] * epx[b][s][t] +
            p_grad[b][s][t + 1] / ep[b][s][t + 1] * epy[b][s][t].

  Or, rearranging:
      p_grad[b][s][t] =
        p_grad[b][s + 1][t] * exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t]) +
        p_grad[b][s][t + 1] * exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1]).
                                                                       (eq. 2)

 (B)  The following is the backprop for epx and epy from (eq. 1):

        epx_grad[b][s - 1][t] += ep_grad[b][s][t] * ep[b][s - 1][t]
        epy_grad[b][s][t - 1] += ep_grad[b][s][t] * ep[b][s][t - 1]

   .. adding 1 to the s indexes in the 1st equation and to the t indexes in the
2nd:

        epx_grad[b][s][t] = ep_grad[b][s + 1][t] * ep[b][s][t]
        epy_grad[b][s][t] = ep_grad[b][s][t + 1] * ep[b][s][t]

    Using, similar to the above, ep_grad = p_grad / ep, and similarly,
    epx_grad = px_grad / epx and epy_grad = py_grad / epy, and writing exp(p)
for p and so on, the above becomes:

      px_grad[b][s][t] / exp(px[b][s][t]) =
        p_grad[b][s + 1][t] / exp(p[b][s + 1][t]) * exp(p[b][s][t])
      py_grad[b][s][t] / exp(py[b][s][t]) =
        p_grad[b][s][t + 1] / exp(p[b][s][t + 1]) * exp(p[b][s][t])
    Rearranging:
      px_grad[b][s][t]  =
        p_grad[b][s + 1][t] * exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t])
                                                                       (eq. 3a)
      py_grad[b][s][t]  =
        p_grad[b][s][t + 1] * exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1])
                                                                       (eq. 3b)


   Defining terms that are common to (eq. 2) and (eqs. 3a,3b), write:

   xderiv[b][s][t] := exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t])  (eq. 4)
   yderiv[b][s][t] := exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1])  (eq. 5)

   .. and note that these quantities are <= 1 so there is no problem doing
   the exponentiation.  So the recursion can be simplified as from eqs. (2, 3a,
3b), as:

       p_grad[b][s][t]  = p_grad[b][s + 1][t] * xderiv[b][s][t] +
                          p_grad[b][s][t + 1] * yderiv[b][s][t]         (eq. 6)
       px_grad[b][s][t] = p_grad[b][s + 1][t] * xderiv[b][s][t]         (eq. 7)
       py_grad[b][s][t] = p_grad[b][s][t + 1] * yderiv[b][s][t]         (eq. 8)

  (It might seem like we could just reuse px_grad and py_grad for (eq. 6), but
it's not clear to me that this is the best strategy since that would require an
extra write to shared memory within the loop that's the limiting factor.)

  The backward pass will be slightly different from the forward pass in terms of
  how we store and index p (and p_grad), because for writing a particular block
  of p_grad, we need context on the top and right instead of the bottom and
  left.  So there are offsets of 1.
 */
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mutual_information_backward_kernel(
    // B, S, T + 1, i.e. batch, x_seq_length, y_seq_length + 1
    torch::PackedTensorAccessor32<scalar_t, 3> px,
    torch::PackedTensorAccessor32<scalar_t, 3> py,  // B, S + 1, T.
    // B, S + 1, T + 1.  Produced in forward pass.
    torch::PackedTensorAccessor32<scalar_t, 3> p,
    // [B].  This is an input.
    torch::PackedTensorAccessor32<scalar_t, 1> ans_grad,
    // B, S + 1, T + 1.   This is a temporary.
    torch::PackedTensorAccessor32<scalar_t, 3> p_grad,
    torch::PackedTensorAccessor32<scalar_t, 3> px_grad,  // B, S, T + 1.
    torch::PackedTensorAccessor32<scalar_t, 3> py_grad,  // B, S + 1, T.
    // B, 4;  or 0, 0 if boundaries are the defaults (0, 0, S, T)
    torch::PackedTensorAccessor32<int64_t, 2> boundary,
    int iter,  // This kernel is sequentially called with 'iter' = num_iters
               // - 1, num_iters - 2, .. 0, where num_iters can be taken to
               // be any sufficiently large number but will actually be:
               // num_s_blocks + num_t_blocks - 1 where num_s_blocks = S /
               // BLOCK_SIZE + 1 and num_t_blocks = T / BLOCK_SIZE + 1
    bool overwrite_ans_grad) {  // If overwite_ans_grad == true, this function
                                // will overwrite ans_grad with a value which,
                                // if everything is working correctly, should be
                                // identical or very close to the value of
                                // ans_grad that was passed in.
  const int B = px.size(0), S = px.size(1), T = py.size(2);

  // For statements that are the same as the forward pass, we are omitting some
  // comments.  We'll focus, in the comments, on differences from the forward
  // pass.
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            // num_t_blocks = T / BLOCK_SIZE + 1,
      num_blocks_this_iter = min(iter + 1, num_s_blocks);

  // px_buf and py_buf are used temporarily to store the px and py values,
  // but then modified to store the "xderiv" and "yderiv" values defined
  // in (eq. 5) and (eq. 6) above.  For out-of-range values, we'll write 0.0
  // here.
  //  Initially (before xderiv/yderiv are written):
  //   px_buf[s][t] contains px[s+s_block_begin][t+t_block_begin];
  //   py_buf[s][t] contains py[s+s_block_begin][t+t_block_begin].
  // Later (see eq. 4 and eq. 5):
  //  px_buf[s][t] contains
  //         exp(p[b][ss][tt] + px[b][ss][tt] - p[b][ss + 1][tt]),
  //  py_buf[s][t] contains
  //         exp(p[b][ss][tt] + py[b][ss][tt] - p[b][ss][tt + 1]
  // where ss == s + s_block_begin, tt = t + t_block_begin.
  // Unlike in the forward code, there is no offset of 1 in the indexes.
  __shared__ scalar_t px_buf[BLOCK_SIZE][BLOCK_SIZE],
      py_buf[BLOCK_SIZE][BLOCK_SIZE];

  // p_buf is initially used to store p, and then (after we are done putting
  // xderiv and yderiv into px_buf and py_buf) it is repurposed to store
  // p_grad.
  //
  // Unlike in the forward pass, p_buf has the same numbering as px_buf and
  // py_buf, it's not offset by 1: e.g., for the origin block, p_buf[0][0]
  // refers to p[0][0] and not p[-1][-1].  The p_buf block is larger by 1 than
  // the block for px_buf and py_buf; unlike in the forward pass, we store
  // context on the top and right, not the bottom and left, i.e. the elements at
  // (one past the largest indexes in the block).
  //
  // For out-of-range elements of p_buf, we'll put zero.
  __shared__ scalar_t p_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied; or (0, 0, S, T) if not.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = T;
  }

  // batch_block_iter iterates over both batch elements (index b), and block
  // indexes in the range [0..num_blocks_this_iter-1].  The order here
  // doesn't matter, since there are no interdependencies between these
  // blocks (they are on a diagonal).
  for (int batch_block_iter = blockIdx.x;
       batch_block_iter < B * num_blocks_this_iter;
       batch_block_iter += gridDim.x) {
    int block = batch_block_iter / B, b = batch_block_iter % B;
    int s_block_begin = block * BLOCK_SIZE,
        t_block_begin = (iter - block) * BLOCK_SIZE;

    if (threadIdx.x < 4 && boundary.size(0) != 0)
      boundary_buf[threadIdx.x] = boundary[b][threadIdx.x];
    __syncthreads();

    int s_begin = boundary_buf[0], t_begin = boundary_buf[1],
        s_end = boundary_buf[2], t_end = boundary_buf[3];
    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block, no greater than
    // (BLOCK_SIZE, BLOCK_SIZE) but possibly less than that if we are towards
    // the end of the sequence.
    // The last element of the output matrix p_grad we write is (s_end, t_end),
    // i.e. the one-past-the-end index of p_grad is (s_end + 1, t_end + 1).
    int block_S = min(BLOCK_SIZE, s_end + 1 - s_block_begin),
        block_T = min(BLOCK_SIZE, t_end + 1 - t_block_begin);

    if (block_S <= 0 || block_T <= 0) continue;

    // Load px_buf and py_buf.  At this point we just set them to the px and py
    // for this block.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // We let px and py default to -infinity if they are out of range, which
      // will cause xderiv and yderiv for out-of-range values to be zero, and
      // cause correct behavior in edge cases (for the top and right blocks).
      // The issue is that p and p_grad are of larger size than px and py.
      scalar_t this_px = -INFINITY;
      if (s < s_end && t <= t_end) this_px = px[b][s][t];
      px_buf[s_in_block][t_in_block] = this_px;
      scalar_t this_py = -INFINITY;
      if (s <= s_end && t < t_end) this_py = py[b][s][t];
      py_buf[s_in_block][t_in_block] = this_py;
    }
    __syncthreads();

    // load p.
    for (int i = threadIdx.x; i < (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1);
         i += blockDim.x) {
      int s_in_block = i / (BLOCK_SIZE + 1), t_in_block = i % (BLOCK_SIZE + 1),
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // Setting 0.0 for out-of-bounds elements of p, together with setting
      // -INFINITY for out-of-bounds elements of px_buf and py_buf, will
      // ensure that we do the right thing in top and right edge cases,
      // i.e. that no derivatives will be propagated from out-of-bounds points
      // because the corresponding xderiv and yderiv values will be zero.
      scalar_t this_p = 0.0;
      if (s <= s_end && t <= t_end) this_p = p[b][s][t];
      // if this_p is -inf, replace with large finite negative value, to avoid
      // NaN's below.
      // TODO: use a value that would work correctly in half precision
      if (this_p < -1.0e+30) this_p = -1.0e+30;
      p_buf[s_in_block][t_in_block] = this_p;
    }
    __syncthreads();

    // Set xderiv and yderiv; see (eq. 4) and (eq. 5).
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      // We can apply this formula to the entire block even if we are processing
      // a partial block; we have ensured that x_buf and y_buf contain
      // -infinity, and p contains 0, for out-of-range elements, so we'll get
      // x_buf and y_buf containing 0 after applying the followin formulas.
      int s = i / BLOCK_SIZE, t = i % BLOCK_SIZE;
      // Mathematically the following is doing:
      //   xderiv[b][s][t] := exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t])
      // (with an offset on the s and t indexes)
      px_buf[s][t] = exp(p_buf[s][t] + px_buf[s][t] - p_buf[s + 1][t]);
      // Mathematically the following is doing:
      //   yderiv[b][s][t] := exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1])
      // (with an offset on the s and t indexes)
      py_buf[s][t] = exp(p_buf[s][t] + py_buf[s][t] - p_buf[s][t + 1]);
    }

    __syncthreads();

    // Load p_grad for the top and right elements in p_buf: i.e. for elements
    // p_buf[s][t] where s == block_S (exclusive-or) t == block_T.  We don't
    // need to load the top-right corner [block_S][block_T]; that location will
    // never be accessed.
    // These are the p_grad values computed by previous instances of this kernel
    // If this is one of the top or right blocks, some or all of the p_grad
    // values we'd be reading here will be out of range, and we use zeros
    // to ensure no gradient gets propagated from those positions.
    if (threadIdx.x < block_S) {
      int s_in_block = threadIdx.x, t_in_block = block_T,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      p_buf[s_in_block][t_in_block] =
          (s <= s_end && t <= t_end ? p_grad[b][s][t] : 0.0);
    } else if (static_cast<unsigned int>(static_cast<int>(threadIdx.x) - 64) <
               static_cast<unsigned int>(block_T)) {
      // casting to unsigned before the comparison tests for both negative and
      // out-of-range values of (int)threadIdx.x - 64.
      int s_in_block = block_S, t_in_block = static_cast<int>(threadIdx.x) - 64,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      p_buf[s_in_block][t_in_block] =
          (s <= s_end && t <= t_end ? p_grad[b][s][t] : 0.0);
    }

    __syncthreads();

    //  The highest-numbered value in p_buf that we need (corresponding,
    // of course, to p_grad), is:
    //    p_buf[block_S - 1][block_T - 1],
    // and the inner iteration number (i) on which we set this is the sum of
    // these indexes, i.e.  (block_S - 1) + (block_T - 1).
    bool is_final_block = (s_block_begin + block_S == s_end + 1 &&
                           t_block_begin + block_T == t_end + 1);

    int first_iter = block_S + block_T - 2;
    if (is_final_block) {
      // The following statement corresponds to:
      // p_grad[b][s_end][t_end] = ans_grad[b]
      // Normally this element of p_buf would be set by the first iteration of
      // the loop below, so if it's set this way we have to decrement first_iter
      // to prevent it from being overwritten.
      p_buf[block_S - 1][block_T - 1] = ans_grad[b];
      --first_iter;
    }

    {
      int s = threadIdx.x;
      for (int i = first_iter; i >= 0; --i) {
        __syncwarp();
        int t = i - s;
        if (s < block_S &&
            static_cast<unsigned int>(t) < static_cast<unsigned int>(block_T)) {
          // The following statement is really operating on the gradients;
          // it corresponds, with offsets of s_block_begin and t_block_begin
          // on the indexes, to (eq. 6) defined above, i.e.:
          //   p_grad[b][s][t]  = p_grad[b][s + 1][t] * xderiv[b][s][t] +
          //                      p_grad[b][s][t + 1] * yderiv[b][s][t]
          p_buf[s][t] =
              (p_buf[s + 1][t] * px_buf[s][t] + p_buf[s][t + 1] * py_buf[s][t]);
        }
      }
    }

    __syncthreads();

    // Write out p_grad, px_grad and py_grad.
    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, t = t_in_block + t_block_begin;
      // s_end and t_end are the one-past-the-end of the (x,y) sequences, but
      // the one-past-the-end element of p_grad would be (s_end + 1, t_end + 1).
      if (t <= t_end && s <= s_end) {
        p_grad[b][s][t] = p_buf[s_in_block][t_in_block];

        if (s < s_end) {  // write px_grad, which is of shape [B][S][T + 1]
          // From (eq. 7):
          // px_grad[b][s][t] = p_grad[b][s + 1][t] * xderiv[b][s][t]
          px_grad[b][s][t] = (p_buf[s_in_block + 1][t_in_block] *
                              px_buf[s_in_block][t_in_block]);
        }
        if (t < t_end) {  // write py_grad, which is of shape [B][S + 1][T]
          // from (eq. 8):
          // py_grad[b][s][t] = p_grad[b][s][t + 1] * yderiv[b][s][t]
          py_grad[b][s][t] = (p_buf[s_in_block][t_in_block + 1] *
                              py_buf[s_in_block][t_in_block]);
        }
      }
    }

    if (threadIdx.x == 0 && s_block_begin == s_begin &&
        t_block_begin == t_begin && overwrite_ans_grad)
      ans_grad[b] = p_buf[0][0];
  }
}

// forward of mutual_information.  See """... """ comment of
// `mutual_information` in mutual_information.py for documentation of the
// behavior of this function.
torch::Tensor MutualInformationCuda(torch::Tensor px, torch::Tensor py,
                                    torch::Tensor boundary, torch::Tensor p) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(
      px.device().is_cuda() && py.device().is_cuda() && p.device().is_cuda(),
      "inputs must be CUDA tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0), S = px.size(1), T = px.size(2) - 1;
  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);
  TORCH_CHECK((boundary.size(0) == 0 && boundary.size(1) == 0) ||
              (boundary.size(0) == B && boundary.size(1) == 4));
  TORCH_CHECK(boundary.device().is_cuda() && boundary.dtype() == torch::kInt64);

  torch::Tensor ans = torch::empty({B}, opts);

  // num_threads and num_blocks and BLOCK_SIZE can be tuned.
  // (however, num_threads may not be less than 128).
  const int num_threads = 128, num_blocks = 256, BLOCK_SIZE = 32;

  // The blocks cover the 'p' matrix, which is of size (B, S+1, T+1),
  // so dividing by BLOCK_SIZE rounding up we get e.g.
  // (S+1 + BLOCK_SIZE-1) / BLOCK_SIZE == S / BLOCK_SIZE + 1
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            num_t_blocks = T / BLOCK_SIZE + 1,
            num_iters = num_s_blocks + num_t_blocks - 1;

  AT_DISPATCH_FLOATING_TYPES(
      px.scalar_type(), "mutual_information_cuda_stub", ([&] {
        for (int iter = 0; iter < num_iters; ++iter) {
          mutual_information_kernel<scalar_t, BLOCK_SIZE>
              <<<num_blocks, num_threads>>>(
                  px.packed_accessor32<scalar_t, 3>(),
                  py.packed_accessor32<scalar_t, 3>(),
                  p.packed_accessor32<scalar_t, 3>(),
                  boundary.packed_accessor32<int64_t, 2>(),
                  ans.packed_accessor32<scalar_t, 1>(), iter);
        }
      }));
  return ans;
}

// backward of mutual_information; returns (grad_px, grad_py)
// If overwrite_ans_grad == true, will overwrite ans_grad with a value which
// should be identical to the original ans_grad if the computation worked
// as it should.
std::vector<torch::Tensor> MutualInformationBackwardCuda(
    torch::Tensor px, torch::Tensor py, torch::Tensor boundary, torch::Tensor p,
    torch::Tensor ans_grad, bool overwrite_ans_grad) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(ans_grad.dim() == 1, "ans_grad must be 1-dimensional.");

  TORCH_CHECK(px.device().is_cuda() && py.device().is_cuda() &&
              p.device().is_cuda() && ans_grad.device().is_cuda() &&
              "inputs must be CUDA tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0), S = px.size(1), T = px.size(2) - 1;

  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);
  TORCH_CHECK((boundary.size(0) == 0 && boundary.size(1) == 0) ||
              (boundary.size(0) == B && boundary.size(1) == 4));
  TORCH_CHECK(boundary.device().is_cuda() && boundary.dtype() == torch::kInt64);
  TORCH_CHECK(ans_grad.size(0) == B);

  bool has_boundary = (boundary.size(0) != 0);

  torch::Tensor p_grad = torch::empty({B, S + 1, T + 1}, opts),
                px_grad = (has_boundary ? torch::zeros({B, S, T + 1}, opts)
                                        : torch::empty({B, S, T + 1}, opts)),
                py_grad = (has_boundary ? torch::zeros({B, S + 1, T}, opts)
                                        : torch::empty({B, S + 1, T}, opts));

  // num_threads and num_blocks and BLOCK_SIZE can be tuned.
  // (however, num_threads may not be less than 128).
  const int num_threads = 128, num_blocks = 256, BLOCK_SIZE = 32;

  // The blocks cover the 'p' matrix, which is of size (B, S+1, T+1),
  // so dividing by BLOCK_SIZE rounding up we get e.g.
  // (S+1 + BLOCK_SIZE-1) / BLOCK_SIZE == S / BLOCK_SIZE + 1
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            num_t_blocks = T / BLOCK_SIZE + 1,
            num_iters = num_s_blocks + num_t_blocks - 1;

  AT_DISPATCH_FLOATING_TYPES(
      px.scalar_type(), "mutual_information_backward_stub", ([&] {
        for (int iter = num_iters - 1; iter >= 0; --iter) {
          mutual_information_backward_kernel<scalar_t, BLOCK_SIZE>
              <<<num_blocks, num_threads>>>(
                  px.packed_accessor32<scalar_t, 3>(),
                  py.packed_accessor32<scalar_t, 3>(),
                  p.packed_accessor32<scalar_t, 3>(),
                  ans_grad.packed_accessor32<scalar_t, 1>(),
                  p_grad.packed_accessor32<scalar_t, 3>(),
                  px_grad.packed_accessor32<scalar_t, 3>(),
                  py_grad.packed_accessor32<scalar_t, 3>(),
                  boundary.packed_accessor32<int64_t, 2>(), iter,
                  overwrite_ans_grad);
        }
      }));
  return std::vector<torch::Tensor>({px_grad, py_grad});
}
}  // namespace k2
