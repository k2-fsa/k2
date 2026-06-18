/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Daniel Povey,
 *                                                   Wei Kang)
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

#include <cooperative_groups.h>

#include "k2/python/csrc/torch/levenshtein_distance.h"

namespace k2 {

/*
  Kernel of levenshtein_distance.  Each thread block computes blocks of the
  'ans' array of (s, t) shape equal to (BLOCK_SIZE, BLOCK_SIZE), e.g. (32, 32).
  Thread-blocks loop over such blocks, but they might loop only once if there is
  not that much data to process.  We sequentially launch thread groups in
  such a way that thread-blocks within a group do not depend on each other
  (see the "iter" parameter).  The blocks of the 'image' (i.e. of the ans
  matrix) that each group handles are arranged in a diagonal.

  Template args:
      BLOCK_SIZE: an integer power of two no greater than 32 (this limitation
                is because we assume BLOCK_SIZE + 1 <= 64 in some data-loading
                code).

    @param px  A two-dimensional tensor with the shape of ``[B][S]`` containing
               sequences. It's data type MUST be ``torch.int32``.
    @param py  A two-dimensional tensor with the shape of ``[B][U]`` containing
               sequences. It's data type MUST be ``torch.int32``.
               ``py`` and ``px`` should have the same batch size.
    @param boundary  If supplied, a torch.LongTensor of shape ``[B][4]``, where
                     each row contains ``[s_begin, u_begin, s_end, u_end]``,
                     with ``0 <= s_begin <= s_end <= S`` and
                     ``0 <= u_begin <= u_end <= U``
                     (this implies that empty sequences are allowed).
                     If not supplied, the values ``[0, 0, S, U]`` will be
                     assumed. These are the beginning and one-past-the-last
                     positions in the ``px`` and ``py`` sequences respectively,
                     and can be used if not all sequences are of the same
                     length.
    @param ans  This function writes to ans[b][s][u] the levenshtein distance
                between sub-sequences of ``px`` and ``py`` of length s and u
                respectively, from the b'th sequences in the batch.  Its shape
                is ``[B][S + 1][U + 1]``. Concretely, this function implements
                the following recursion, in the case where
                s_begin == u_begin == 0:

               ans[b, 0, u] = u
               ans[b, s, 0] = s
               ans[b, s, t] = min(min(ans[b, s-1, u] + 1, ans[b, s, u - 1] + 1),
                              ans[b, s-1, u-1] + (px[b, s] == py[b, u] ? 0 :1))

               if `boundary` is set, we start from ans[b,s_begin,t_begin]=0.
               The values in the positions out of the range of boundary are
               uninitialized, can be any random values.

   The block-dim and grid-dim must both be 1-dimensional, and the block-dim must
   be at least 128.
*/
template <int BLOCK_SIZE>  // e.g. BLOCK_SIZE == 16 or 32.
__global__ void levenshtein_distance_kernel(
    // B, S, i.e. batch, x_seq_length
    torch::PackedTensorAccessor32<int32_t, 2> px,
    // B, U, i.e. batch, y_seq_length
    torch::PackedTensorAccessor32<int32_t, 2> py,
    // B, 4,  or 0, 0 if boundaries are the defaults (0, 0, S, U)
    torch::PackedTensorAccessor32<int64_t, 2> boundary,
    torch::PackedTensorAccessor32<int32_t, 3> ans,  // [B, S + 1, U + 1]
    int iter) {  // This kernel is sequentially called with 'iter' = 0, 1, 2 and
                 // so on, up to num_iters - 1 where num_iters = num_s_blocks +
                 // num_s_blocks - 1 num_s_blocks = S / BLOCK_SIZE + 1
                 // num_u_blocks = U / BLOCK_SIZE + 1
                 // so that each group depends on the previous group...
  const int B = px.size(0), S = px.size(1), U = py.size(1);

  // num_s_blocks and num_u_blocks are the number of blocks we need to cover the
  // array of size (S, U) with blocks of this size, in the s and u directions
  // respectively.
  // You can read the following expressions as simplifications of, for example,
  // num_s_blocks = ((S + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
  // i.e. rounding-up division of (S + 1) by BLOCK_SIZE,
  // and the same for (U + 1).
  const int num_s_blocks = S / BLOCK_SIZE + 1;

  // num_blocks_this_iter is an upper bound on the number of blocks of size
  // (BLOCK_SIZE by BLOCK_SIZE) that might be active on this iteration (`iter`).
  // These iterations start from the bottom left of the image so that on iter ==
  // 0 we process only one block with block-index (0, 0) then on iter == 1 we
  // process block-indexes (1, 0) and (0, 1); and then on iter==2 we process (2,
  // 0), (1, 1) and (0, 2); and so on.  We also will never have more than
  // `num_s_blocks` blocks (We'll never have more than num_u_blocks either, but
  // the numbering we use corresponds to s and not u, so when we hit the
  // num_u_blocks limit, the blocks with the lowest s indexes would just not be
  // active and we'll 'continue' in the loop below).
  int num_blocks_this_iter = min(iter + 1, num_s_blocks);

  // px_buf[s] == px[s + s_block_begin]; py_buf[u] == py[u + u_block_begin]
  __shared__ int32_t px_buf[BLOCK_SIZE], py_buf[BLOCK_SIZE];

  // ans_buf[s][u] == ans[s+s_block_begin][u+u_block_begin]
  // 1st row/col of ans_buf correspond to the previously computed blocks (lower
  // `iter`)
  // Note: ans[s_begin][u] and ans[s][u_begin] are initial values, so actually
  // we will start from ans[s_begin + 1][u_begin + 1]
  __shared__ int32_t ans_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied; or (0, 0, S, U) otherwise.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = U;
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
        u_block_begin = (iter - block) * BLOCK_SIZE;

    __syncthreads();

    if (threadIdx.x < 4) boundary_buf[threadIdx.x] = boundary[b][threadIdx.x];

    __syncthreads();

    int s_begin = boundary_buf[0], u_begin = boundary_buf[1],
        s_end = boundary_buf[2], u_end = boundary_buf[3];

    s_block_begin += s_begin;
    u_block_begin += u_begin;

    // block_S and block_U are the actual sizes of this block (the block of
    // `ans` that we will write), no greater than (BLOCK_SIZE, BLOCK_SIZE) but
    // possibly less than that if we are towards the end of the sequence.  The
    // last element in the output matrix ans that we need to write is (s_end,
    // u_end), i.e. the one-past-the-end index is (s_end + 1, u_end + 1).
    int block_S = min(BLOCK_SIZE, s_end - s_block_begin),
        block_U = min(BLOCK_SIZE, u_end - u_block_begin);

    if (block_S < 0 || block_U < 0) continue;

    // Load px_buf and py_buf.
    if (threadIdx.x < BLOCK_SIZE) {
      int idx_in_block = threadIdx.x, s = idx_in_block + s_block_begin,
          u = idx_in_block + u_block_begin;

      int32_t this_px = -1;
      if (s >= s_begin && s < s_end) this_px = px[b][s];
      px_buf[idx_in_block] = this_px;

      int32_t this_py = -1;
      if (u >= u_begin && u < u_end) this_py = py[b][u];
      py_buf[idx_in_block] = this_py;
    }

    // Load the 1st row and 1st column of ans_buf.
    // This is the context from previously computed blocks of the
    // image.  ans_buf[s][u] will correspond to
    // ans[s + s_block_begin][u + u_block_begin]. ans matrix has a shape of
    // [S + 1][U + 1] and the first row (i.e. ans[s][u_begin]) and the first
    // column (i.e. ans[s_begin][u]) are initialized within this function, so
    // the corresponding elements in ans_buf will not load from ans.
    if (threadIdx.x <= BLOCK_SIZE) {
      // s_in_p_buf and u_in_pbuf are simply the indexes into ans_buf
      int s_in_ans_buf = threadIdx.x, u_in_ans_buf = 0,
          s = s_in_ans_buf + s_block_begin, u = u_in_ans_buf + u_block_begin;

      // The initial value for the first row, which means py is an empty
      // sequence.
      int32_t this_ans = s - s_begin;

      // Note: The condition is `s > s_begin` and `u > u_begin`, we will not
      // load the first row and first column from ans.
      if (s > s_begin && s <= s_end && u > u_begin && u <= u_end)
        this_ans = ans[b][s][u];

      // The condition is !(s_block_begin == s_begin && s_in_ans_buf == 0)
      // it means we won't write to 1st column when loading 1st row, so as not
      // to pollute ans_buf[0][0]
      if (s_block_begin != s_begin || s_in_ans_buf != 0)
        ans_buf[s_in_ans_buf][u_in_ans_buf] = this_ans;
    } else if (static_cast<unsigned int>(static_cast<int>(threadIdx.x) - 64) <=
               static_cast<unsigned int>(BLOCK_SIZE)) {
      // Another warp handles the other leg.  Checking as unsigned
      // tests that threadIdx.x - 64 is both >= 0 and <= BLOCK_SIZE
      int s_in_ans_buf = 0, u_in_ans_buf = static_cast<int>(threadIdx.x) - 64,
          s = s_in_ans_buf + s_block_begin, u = u_in_ans_buf + u_block_begin;

      int32_t this_ans = u - u_begin;

      if (s > s_begin && s <= s_end && u > u_begin && u <= u_end)
        this_ans = ans[b][s][u];

      // The condition is !(u_block_begin == u_begin && u_in_ans_buf == 0)
      // it means we won't write to 1st row when loading 1st column, so as not
      // to pollute ans_buf[0][0]
      if (u_block_begin != u_begin || u_in_ans_buf != 0)
        ans_buf[s_in_ans_buf][u_in_ans_buf] = this_ans;
    }

    // Initial the first element of the original block, as the code above would
    // not write to this position, so, treat it as a special case here.
    if (threadIdx.x == 0) {
      if (s_block_begin == s_begin && u_block_begin == u_begin)
        ans_buf[0][0] = 0;
    }

    __syncthreads();

    // from here to the next __syncthreads(), only the 1st warp should be active
    // so we shouldn't need to synchronize.  (implicit within-warp
    // synchronization).
    int s = threadIdx.x;
    for (int i = 0; i < block_S + block_U - 1; ++i) {
      __syncwarp();
      // i is the inner iteration, which corresponds to the (s + t) indexes of
      // the elements within the block that we write.  So i == 0 writes
      // positions (s, t) == (0, 0) (but we treated i == 0 as a special case
      // above); i == 1 writes (0, 1) and (1, 0); i == 2 writes (0, 2), (1, 1)
      // and (2, 1); and so on.  Note: not many threads participate in this
      // part, only up to BLOCK_SIZE at most.  Unfortunately we couldn't figure
      // out a very meaningful way for more threads to do work, that looked like
      // it would really speed things up.
      // So this kernel does (2 * BLOCK_SIZE) iterations, which may seem a lot,
      // but we do at least do the I/O in an efficient way.
      int u = i - s;
      if (s < block_S &&
          static_cast<unsigned int>(u) < static_cast<unsigned int>(block_U)) {
        // ans_buf is indexed by s + 1 and t + 1 because it has an extra initial
        // row and column for context from previous blocks.
        int32_t cost = px_buf[s] == py_buf[u] ? 0 : 1;
        ans_buf[s + 1][u + 1] =
            min(min(ans_buf[s][u + 1] + 1, ans_buf[s + 1][u] + 1),
                ans_buf[s][u] + cost);
        // We don't need to do __syncthreads() in this loop because all the
        // threads that are active are in the same warp.  (However, in future,
        // if NVidia changes some things, we might need to sync here).
      }
    }
    __syncthreads();

    // Write out the data to ans;

    // The left and bottom column, which means that py is empty or px is empty.
    if (threadIdx.x <= BLOCK_SIZE) {
      int idx_in_block = threadIdx.x, s = idx_in_block + s_block_begin,
          u = idx_in_block + u_block_begin;
      if (s_block_begin == s_begin && idx_in_block <= block_U)
        ans[b][s_begin][u] = ans_buf[0][idx_in_block];
      if (u_block_begin == u_begin && idx_in_block <= block_S)
        ans[b][s][u_begin] = ans_buf[idx_in_block][0];
    }

    for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE, u_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin, u = u_in_block + u_block_begin;
      if (s_in_block < block_S && u_in_block < block_U) {
        int32_t this_ans = ans_buf[s_in_block + 1][u_in_block + 1];
        ans[b][s + 1][u + 1] = this_ans;
      }
    }
  }
}

torch::Tensor LevenshteinDistanceCuda(
    torch::Tensor px, torch::Tensor py,
    torch::optional<torch::Tensor> opt_boundary) {
  TORCH_CHECK(px.dim() == 2, "px must be 2-dimensional");
  TORCH_CHECK(py.dim() == 2, "py must be 2-dimensional.");
  TORCH_CHECK(px.device().is_cuda() && py.device().is_cuda(),
              "inputs must be CUDA tensors");

  TORCH_CHECK(px.dtype() == torch::kInt32 && py.dtype() == torch::kInt32,
              "The dtype of inputs must be kInt32");

  auto opts = torch::TensorOptions().dtype(px.dtype()).device(px.device());

  const int B = px.size(0), S = px.size(1), U = py.size(1);
  TORCH_CHECK(B == py.size(0), "px and py must have same batch size");

  auto boundary = opt_boundary.value_or(
      torch::tensor({0, 0, S, U},
                    torch::dtype(torch::kInt64).device(px.device()))
          .reshape({1, 4})
          .expand({B, 4}));
  TORCH_CHECK(boundary.size(0) == B && boundary.size(1) == 4);
  TORCH_CHECK(boundary.device().is_cuda() && boundary.dtype() == torch::kInt64);

  torch::Tensor ans = torch::empty({B, S + 1, U + 1}, opts);

  // num_threads and num_blocks and BLOCK_SIZE can be tuned.
  // (however, num_threads may not be less than 128).
  const int num_threads = 128, num_blocks = 256, BLOCK_SIZE = 32;

  // The blocks cover the 'ans' matrix, which is of size (B, S+1, U+1),
  // so dividing by BLOCK_SIZE rounding up we get e.g.
  // (S + 1 + BLOCK_SIZE-1) / BLOCK_SIZE == S / BLOCK_SIZE + 1
  const int num_s_blocks = S / BLOCK_SIZE + 1,
            num_u_blocks = U / BLOCK_SIZE + 1,
            num_iters = num_s_blocks + num_u_blocks - 1;

  for (int iter = 0; iter < num_iters; ++iter) {
    levenshtein_distance_kernel<BLOCK_SIZE><<<num_blocks, num_threads>>>(
        px.packed_accessor32<int32_t, 2>(), py.packed_accessor32<int32_t, 2>(),
        boundary.packed_accessor32<int64_t, 2>(),
        ans.packed_accessor32<int32_t, 3>(), iter);
  }
  return ans;
}

}  // namespace k2
