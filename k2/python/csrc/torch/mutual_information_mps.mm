/**
 * Copyright      2024  k2-fsa Authors
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

// Native Metal implementation of mutual_information forward and backward.
// Mirrors the blocked antidiagonal wavefront pattern of mutual_information_cuda.cu.
// Only compiled when K2_WITH_MPS is defined (Apple + MPS build).

#ifdef K2_WITH_MPS

// mutual_information.h pulls in torch/extension.h which includes
// aten_interned_strings.h. OperationUtils.h defines
// TORCH_ASSERT_ONLY_METHOD_OPERATORS which would trigger a #error in
// aten_interned_strings.h if included first. So mutual_information.h
// must come before the MPS headers.
#include "k2/python/csrc/torch/mutual_information.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

// ─────────────────────────────────────────────────────────────────────────────
// Metal Shading Language kernel source (embedded, runtime-compiled once)
// ─────────────────────────────────────────────────────────────────────────────
//
// Design mirrors mutual_information_cuda.cu:
//   • BLOCK_SIZE 32 tiles on the (s,t) grid
//   • Outer diagonal loop driven from C++ (one Metal dispatch per `iter`)
//   • 128 threads per threadgroup; cooperative load / scatter
//   • Inner antidiagonal computed by the first SIMD group (threads 0–31)
//   • threadgroup_barrier / simdgroup_barrier replace __syncthreads / __syncwarp
// ─────────────────────────────────────────────────────────────────────────────
static const char kMIKernelSrc[] = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ── helpers ──────────────────────────────────────────────────────────────────
inline float log_add(float a, float b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    float hi = max(a, b);
    // exp(lo - hi) is in (0, 1] so log(1 + exp(...)) is numerically stable
    return hi + log(1.0f + exp(min(a, b) - hi));
}

inline float safe_exp_mps(float x) {
    if (isnan(x) || isinf(x)) return 0.0f;
    float r = exp(x);
    return (isnan(r) || isinf(r)) ? 0.0f : r;
}

// ── BLOCK_SIZE must match the C++ constant (32) ───────────────────────────────
constant int BSIZE = 32;

// ── Parameter structs (must exactly match the C++ side) ──────────────────────
struct MIFwdParams {
    int B, S, T;
    int px_stride_b, px_stride_s;   // px strides; t stride == 1
    int py_stride_b, py_stride_s;   // py strides; t stride == 1
    int p_stride_b,  p_stride_s;    // p  strides; t stride == 1
    int iter;
    int num_blocks_this_iter;
    int t_offset;   // 0 if !modified, -1 if modified
};

struct MIBwdParams {
    int B, S, T;
    int px_stride_b, px_stride_s;
    int py_stride_b, py_stride_s;
    int p_stride_b,  p_stride_s;
    int iter;
    int num_blocks_this_iter;
    int neg_t_offset;  // 0 if !modified, 1 if modified
    int has_boundary;
};

// ── Forward kernel ────────────────────────────────────────────────────────────
// Each threadgroup handles multiple (batch, block) pairs via the outer loop.
// Thread layout: 128 threads / threadgroup.
// Inner antidiagonal uses only threads 0..BSIZE-1 (first SIMD group, size 32).
kernel void mi_forward(
    device const float*     px          [[buffer(0)]],  // [B][S][T+1 or T]
    device const float*     py          [[buffer(1)]],  // [B][S+1][T]
    device       float*     p           [[buffer(2)]],  // [B][S+1][T+1]  (in/out)
    device const int*       boundary    [[buffer(3)]],  // [B][4] int32
    device       float*     ans         [[buffer(4)]],  // [B]
    constant MIFwdParams&   params      [[buffer(5)]],
    uint tg_id    [[threadgroup_position_in_grid]],
    uint tg_count [[threadgroups_per_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    // Threadgroup-resident buffers
    threadgroup float px_buf[BSIZE][BSIZE];          // 4 KB
    threadgroup float py_buf[BSIZE][BSIZE];          // 4 KB
    threadgroup float p_buf[BSIZE + 1][BSIZE + 1];  // ~4.4 KB
    threadgroup int   bnd[4];

    const int B   = params.B, S = params.S, T = params.T;
    const int t_o = params.t_offset;
    const int iter = params.iter;
    const int nblk = params.num_blocks_this_iter;

    for (int bbi = (int)tg_id; bbi < B * nblk; bbi += (int)tg_count) {
        int blk = bbi / B, b = bbi % B;
        int s_bb = blk * BSIZE;           // s_block_begin (before adding s_begin)
        int t_bb = (iter - blk) * BSIZE;  // t_block_begin (before adding t_begin)

        // ── Load boundary for batch element b ─────────────────────────────
        if (tid == 0) { bnd[0] = 0; bnd[1] = 0; bnd[2] = S; bnd[3] = T; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) bnd[tid] = boundary[b * 4 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int s_begin = bnd[0], t_begin = bnd[1];
        int s_end   = bnd[2], t_end   = bnd[3];
        s_bb += s_begin;
        t_bb += t_begin;

        int block_S = min(BSIZE, s_end + 1 - s_bb);
        int block_T = min(BSIZE, t_end + 1 - t_bb);
        if (block_S <= 0 || block_T <= 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        bool is_origin = (s_bb == s_begin && t_bb == t_begin);

        // ── Cooperative load of px_buf and py_buf ─────────────────────────
        // 128 threads cover 1024 = BSIZE*BSIZE elements in 8 passes
        for (int i = (int)tid; i < BSIZE * BSIZE; i += 128) {
            int si = i / BSIZE, ti = i % BSIZE;
            int sg = si + s_bb, tg_g = ti + t_bb;
            int t_off = tg_g + t_o;

            float pxv = -INFINITY;
            if (sg > s_begin && sg <= s_end && t_off >= t_begin && tg_g <= t_end)
                pxv = px[b * params.px_stride_b + (sg - 1) * params.px_stride_s + t_off];
            px_buf[si][ti] = pxv;

            float pyv = -INFINITY;
            if (tg_g > t_begin && tg_g <= t_end && sg <= s_end)
                pyv = py[b * params.py_stride_b + sg * params.py_stride_s + (tg_g - 1)];
            py_buf[si][ti] = pyv;
        }

        // ── Load border of p_buf (first column + first row) ───────────────
        // First column (s_in_p = 0..BSIZE, t_in_p = 0): threads 0..BSIZE
        if (tid <= (uint)BSIZE) {
            int si = (int)tid;
            int s = si + s_bb - 1, t = t_bb - 1;
            float pv = -INFINITY;
            if (s >= s_begin && s <= s_end && t >= t_begin && t <= t_end)
                pv = p[b * params.p_stride_b + s * params.p_stride_s + t];
            p_buf[si][0] = pv;
        }
        // First row (s_in_p = 0, t_in_p = 0..BSIZE): threads 64..64+BSIZE
        // Unsigned cast trick mirrors CUDA: tests both >= 0 and <= BSIZE
        {
            uint u = tid - 64u;  // wraps to large value for tid < 64
            if (u <= (uint)BSIZE) {
                int ti = (int)u;
                int s = s_bb - 1, t = ti + t_bb - 1;
                float pv = -INFINITY;
                if (s >= s_begin && s <= s_end && t >= t_begin && t <= t_end)
                    pv = p[b * params.p_stride_b + s * params.p_stride_s + t];
                p_buf[0][ti] = pv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Thread 0: initialize p_buf[1][1] ─────────────────────────────
        if (tid == 0) {
            p_buf[1][1] = is_origin ? 0.0f :
                log_add(p_buf[0][1 + t_o] + px_buf[0][0],
                        p_buf[1][0]        + py_buf[0][0]);
        }

        // ── Inner antidiagonal sweep (threads 0..BSIZE-1 only) ────────────
        // simdgroup_barrier syncs the first SIMD group (threads 0..31)
        // which is the only group active in this loop.
        int s = (int)tid;
        for (int i = 1; i < block_S + block_T - 1; ++i) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            int t = i - s;
            if (s < block_S && t >= 0 && t < block_T) {
                p_buf[s + 1][t + 1] = log_add(
                    p_buf[s][t + 1 + t_o] + px_buf[s][t],
                    p_buf[s + 1][t]        + py_buf[s][t]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Write p_buf results back to global p ──────────────────────────
        for (int i = (int)tid; i < BSIZE * BSIZE; i += 128) {
            int si = i / BSIZE, ti = i % BSIZE;
            if (si < block_S && ti < block_T) {
                int sg = si + s_bb, tg_g = ti + t_bb;
                p[b * params.p_stride_b + sg * params.p_stride_s + tg_g] =
                    p_buf[si + 1][ti + 1];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Thread 0: write ans if this is the final (top-right) block ────
        if (tid == 0 &&
            s_bb + block_S - 1 == s_end &&
            t_bb + block_T - 1 == t_end) {
            ans[b] = p_buf[block_S][block_T];
        }
    }
}

// ── Backward kernel ───────────────────────────────────────────────────────────
// Mirrors mutual_information_backward_kernel in mutual_information_cuda.cu.
// Inputs: px, py, p (forward output), ans_grad
// Outputs: px_grad, py_grad  (p_grad computed internally and accumulated)
kernel void mi_backward(
    device const float*     px          [[buffer(0)]],  // [B][S][T+1 or T]
    device const float*     py          [[buffer(1)]],  // [B][S+1][T]
    device const float*     p           [[buffer(2)]],  // [B][S+1][T+1]
    device       float*     ans_grad    [[buffer(3)]],  // [B]
    device       float*     p_grad      [[buffer(4)]],  // [B][S+1][T+1]
    device       float*     px_grad     [[buffer(5)]],  // [B][S][T+1 or T]
    device       float*     py_grad     [[buffer(6)]],  // [B][S+1][T]
    device const int*       boundary    [[buffer(7)]],  // [B][4] int32
    constant MIBwdParams&   params      [[buffer(8)]],
    uint tg_id    [[threadgroup_position_in_grid]],
    uint tg_count [[threadgroups_per_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    // px_buf / py_buf: initially px/py values, then overwritten with term1/term2
    threadgroup float px_buf[BSIZE][BSIZE];
    threadgroup float py_buf[BSIZE][BSIZE];
    // p_buf: (BSIZE+1)×(BSIZE+1), first used for p values, then repurposed for p_grad.
    // Unlike forward, indexing is NOT offset by 1; context is on TOP and RIGHT.
    threadgroup float p_buf[BSIZE + 1][BSIZE + 1];
    threadgroup int   bnd[4];

    const int B = params.B, S = params.S, T = params.T;
    const int neg_t_o = params.neg_t_offset;  // 0 if !modified, 1 if modified
    const int iter = params.iter;
    const int nblk = params.num_blocks_this_iter;

    for (int bbi = (int)tg_id; bbi < B * nblk; bbi += (int)tg_count) {
        int blk = bbi / B, b = bbi % B;
        int s_bb = blk * BSIZE;
        int t_bb = (iter - blk) * BSIZE;

        // ── Boundary ──────────────────────────────────────────────────────
        if (tid == 0) { bnd[0] = 0; bnd[1] = 0; bnd[2] = S; bnd[3] = T; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) bnd[tid] = boundary[b * 4 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int s_begin = bnd[0], t_begin = bnd[1];
        int s_end   = bnd[2], t_end   = bnd[3];
        s_bb += s_begin;
        t_bb += t_begin;

        int block_S = min(BSIZE, s_end + 1 - s_bb);
        int block_T = min(BSIZE, t_end + 1 - t_bb);
        if (block_S <= 0 || block_T <= 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // ── Load px_buf and py_buf ────────────────────────────────────────
        for (int i = (int)tid; i < BSIZE * BSIZE; i += 128) {
            int si = i / BSIZE, ti = i % BSIZE;
            int sg = si + s_bb, tg_g = ti + t_bb;

            float pxv = -INFINITY;
            if (sg < s_end && tg_g <= t_end)
                pxv = px[b * params.px_stride_b + sg * params.px_stride_s + tg_g];
            px_buf[si][ti] = pxv;

            float pyv = -INFINITY;
            if (sg <= s_end && tg_g < t_end)
                pyv = py[b * params.py_stride_b + sg * params.py_stride_s + tg_g];
            py_buf[si][ti] = pyv;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Load p_buf from global p  (size (BSIZE+1)×(BSIZE+1)) ─────────
        for (int i = (int)tid; i < (BSIZE + 1) * (BSIZE + 1); i += 128) {
            int si = i / (BSIZE + 1), ti = i % (BSIZE + 1);
            int sg = si + s_bb, tg_g = ti + t_bb;
            float pv = 0.0f;
            if (sg <= s_end && tg_g <= t_end) {
                pv = p[b * params.p_stride_b + sg * params.p_stride_s + tg_g];
                if (pv < -1.0e+30f) pv = -1.0e+30f;
            }
            p_buf[si][ti] = pv;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute term1 (xderiv) and term2 (yderiv) in-place ───────────
        // term1[s][t] = safe_exp(p[s][t] + px[s][t] - p[s+1][t-t_offset])
        //             = safe_exp(p_buf[s][t] + px_buf[s][t] - p_buf[s+1][t+neg_t_o])
        // term2[s][t] = safe_exp(p[s][t] + py[s][t] - p[s][t+1])
        for (int i = (int)tid; i < BSIZE * BSIZE; i += 128) {
            int si = i / BSIZE, ti = i % BSIZE;
            float xd = safe_exp_mps(p_buf[si][ti] + px_buf[si][ti]
                                    - p_buf[si + 1][ti + neg_t_o]);
            float yd = safe_exp_mps(p_buf[si][ti] + py_buf[si][ti]
                                    - p_buf[si][ti + 1]);
            px_buf[si][ti] = xd;
            py_buf[si][ti] = yd;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Load p_grad for top+right border of this block ────────────────
        // p_buf[s][block_T] for s in [0..block_S]: threads 0..block_S
        if (tid <= (uint)block_S) {
            int si = (int)tid, sg = si + s_bb, tg_g = block_T + t_bb;
            p_buf[si][block_T] = (sg <= s_end && tg_g <= t_end)
                                 ? p_grad[b * params.p_stride_b + sg * params.p_stride_s + tg_g]
                                 : 0.0f;
        }
        // p_buf[block_S][t] for t in [0..block_T-1]: use unsigned trick for threads 64..64+block_T-1
        {
            uint u = tid - 64u;
            if (u < (uint)block_T) {
                int ti = (int)u, sg = block_S + s_bb, tg_g = ti + t_bb;
                p_buf[block_S][ti] = (sg <= s_end && tg_g <= t_end)
                                     ? p_grad[b * params.p_stride_b + sg * params.p_stride_s + tg_g]
                                     : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Handle final block: seed p_grad[s_end][t_end] with ans_grad ──
        bool is_final = (s_bb + block_S == s_end + 1 && t_bb + block_T == t_end + 1);
        int first_iter = block_S + block_T - 2;
        if (is_final) {
            if (tid == 0) p_buf[block_S - 1][block_T - 1] = ans_grad[b];
            --first_iter;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Inner reverse antidiagonal sweep ─────────────────────────────
        int s = (int)tid;
        for (int i = first_iter; i >= 0; --i) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            int t = i - s;
            if (s < block_S && t >= 0 && t < block_T) {
                // p_grad[s,t] = p_grad[s+1,t-t_offset]*term1[s,t] + p_grad[s,t+1]*term2[s,t]
                //             = p_buf[s+1][t+neg_t_o] * px_buf[s][t] + p_buf[s][t+1] * py_buf[s][t]
                p_buf[s][t] = p_buf[s + 1][t + neg_t_o] * px_buf[s][t]
                            + p_buf[s][t + 1]             * py_buf[s][t];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Write p_grad, px_grad, py_grad ───────────────────────────────
        for (int i = (int)tid; i < BSIZE * BSIZE; i += 128) {
            int si = i / BSIZE, ti = i % BSIZE;
            int sg = si + s_bb, tg_g = ti + t_bb;
            if (tg_g <= t_end && sg <= s_end) {
                p_grad[b * params.p_stride_b + sg * params.p_stride_s + tg_g] =
                    p_buf[si][ti];

                // px_grad: shape [B][S][T+1] if !modified, [B][S][T] if modified
                // condition: sg < s_end && tg_g <= t_end - neg_t_o
                if (sg < s_end && tg_g <= t_end - neg_t_o) {
                    // px_grad[b][sg][tg_g] = p_grad[sg+1][tg_g+neg_t_o] * term1[sg][tg_g]
                    px_grad[b * params.px_stride_b + sg * params.px_stride_s + tg_g] =
                        p_buf[si + 1][ti + neg_t_o] * px_buf[si][ti];
                }

                // py_grad: shape [B][S+1][T]
                if (tg_g < t_end) {
                    py_grad[b * params.py_stride_b + sg * params.py_stride_s + tg_g] =
                        p_buf[si][ti + 1] * py_buf[si][ti];
                }
            }
        }

        // Thread 0: optionally overwrite ans_grad[b] with recomputed value
        // (origin block: p_buf[0][0] == p_grad[s_begin][t_begin])
        if (tid == 0 && s_bb == s_begin && t_bb == t_begin)
            ans_grad[b] = p_buf[0][0];
    }
}
)MSL";

// ─────────────────────────────────────────────────────────────────────────────
// C++ wrapper: compile kernel once, cache pipeline states
// ─────────────────────────────────────────────────────────────────────────────
namespace {

struct MIPipelineCache {
    id<MTLComputePipelineState> fwd = nil;
    id<MTLComputePipelineState> bwd = nil;
};

// Struct layouts must exactly match MSL structs above.
struct MIFwdParams {
    int B, S, T;
    int px_stride_b, px_stride_s;
    int py_stride_b, py_stride_s;
    int p_stride_b,  p_stride_s;
    int iter;
    int num_blocks_this_iter;
    int t_offset;
};

struct MIBwdParams {
    int B, S, T;
    int px_stride_b, px_stride_s;
    int py_stride_b, py_stride_s;
    int p_stride_b,  p_stride_s;
    int iter;
    int num_blocks_this_iter;
    int neg_t_offset;
    int has_boundary;
};

MIPipelineCache* GetOrBuildPipelines() {
    static dispatch_once_t token;
    static MIPipelineCache* cache = nullptr;
    dispatch_once(&token, ^{
        cache = new MIPipelineCache();
        id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();

        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion2_4;
        NSString* src = [NSString stringWithUTF8String:kMIKernelSrc];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            NSLog(@"k2 MPS: Metal compile error: %@", err.localizedDescription);
            return;
        }

        id<MTLFunction> fwd_fn = [lib newFunctionWithName:@"mi_forward"];
        if (!fwd_fn) {
            NSLog(@"k2 MPS: Metal function 'mi_forward' not found in compiled library");
            return;
        }
        id<MTLFunction> bwd_fn = [lib newFunctionWithName:@"mi_backward"];
        if (!bwd_fn) {
            NSLog(@"k2 MPS: Metal function 'mi_backward' not found in compiled library");
            return;
        }

        cache->fwd = [device newComputePipelineStateWithFunction:fwd_fn error:&err];
        if (!cache->fwd)
            NSLog(@"k2 MPS: pipeline (fwd) error: %@", err.localizedDescription);

        cache->bwd = [device newComputePipelineStateWithFunction:bwd_fn error:&err];
        if (!cache->bwd)
            NSLog(@"k2 MPS: pipeline (bwd) error: %@", err.localizedDescription);
    });
    return cache;
}

// Encode one kernel dispatch onto PyTorch's MPS command buffer.
void EncodeForward(id<MTLComputePipelineState> pso,
                   id<MTLCommandBuffer> cmdbuf,
                   const torch::Tensor& px,
                   const torch::Tensor& py,
                   torch::Tensor& p,
                   const torch::Tensor& boundary,  // [B][4] int32, contiguous
                   torch::Tensor& ans,
                   const MIFwdParams& params,
                   int num_threadgroups) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:pso];

    auto bind = [&](int idx, const torch::Tensor& t) {
        [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
                offset:t.storage_offset() * t.element_size()
               atIndex:idx];
    };
    bind(0, px);
    bind(1, py);
    bind(2, p);
    bind(3, boundary);
    bind(4, ans);
    [enc setBytes:&params length:sizeof(params) atIndex:5];

    MTLSize tg_size  = {128, 1, 1};
    MTLSize grid_sz  = {(NSUInteger)num_threadgroups, 1, 1};
    [enc dispatchThreadgroups:grid_sz threadsPerThreadgroup:tg_size];
    [enc endEncoding];
}

void EncodeBackward(id<MTLComputePipelineState> pso,
                    id<MTLCommandBuffer> cmdbuf,
                    const torch::Tensor& px,
                    const torch::Tensor& py,
                    const torch::Tensor& p,
                    torch::Tensor& ans_grad,
                    torch::Tensor& p_grad,
                    torch::Tensor& px_grad,
                    torch::Tensor& py_grad,
                    const torch::Tensor& boundary,  // [B][4] int32
                    const MIBwdParams& params,
                    int num_threadgroups) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:pso];

    auto bind = [&](int idx, const torch::Tensor& t) {
        [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
                offset:t.storage_offset() * t.element_size()
               atIndex:idx];
    };
    bind(0, px);  bind(1, py);  bind(2, p);
    bind(3, ans_grad);
    bind(4, p_grad);  bind(5, px_grad);  bind(6, py_grad);
    bind(7, boundary);
    [enc setBytes:&params length:sizeof(params) atIndex:8];

    MTLSize tg_size = {128, 1, 1};
    MTLSize grid_sz = {(NSUInteger)num_threadgroups, 1, 1};
    [enc dispatchThreadgroups:grid_sz threadsPerThreadgroup:tg_size];
    [enc endEncoding];
}

}  // namespace

namespace k2 {

// ─────────────────────────────────────────────────────────────────────────────
// MutualInformationMps
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor MutualInformationMps(torch::Tensor px, torch::Tensor py,
                                   torch::optional<torch::Tensor> opt_boundary,
                                   torch::Tensor p) {
    TORCH_CHECK(px.dim() == 3 && py.dim() == 3 && p.dim() == 3);
    TORCH_CHECK(px.scalar_type() == torch::kFloat,
                "MutualInformationMps only supports float32; got ",
                px.scalar_type());

    // Ensure contiguous layout (required for stride-1 t-dimension assumption)
    auto px_c = px.contiguous();
    auto py_c = py.contiguous();
    auto p_c  = p.contiguous();

    const int B = px_c.size(0), S = px_c.size(1), T = py_c.size(2);
    const bool modified = (px_c.size(2) == (int64_t)T);

    const int BLOCK_SIZE = 32;
    const int num_s_blocks = S / BLOCK_SIZE + 1;
    const int num_t_blocks = T / BLOCK_SIZE + 1;
    const int num_iters    = num_s_blocks + num_t_blocks - 1;

    // Expand or create boundary tensor as int32 on MPS
    torch::Tensor boundary_i32;
    if (opt_boundary.has_value()) {
        boundary_i32 = opt_boundary.value().to(torch::kInt32).contiguous();
    } else {
        boundary_i32 = torch::tensor({0, 0, S, T},
                           torch::TensorOptions().dtype(torch::kInt32).device(px.device()))
                           .reshape({1, 4}).expand({B, 4}).contiguous();
    }

    auto ans = torch::empty({B},
        torch::TensorOptions().dtype(px.scalar_type()).device(px.device()));

    MIPipelineCache* cache = GetOrBuildPipelines();
    TORCH_CHECK(cache && cache->fwd, "k2 MPS: failed to build MI forward pipeline");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    // Flush any open PyTorch encoder before creating our own.
    // COMMIT_AND_CONTINUE ends the current encoder, commits the command buffer,
    // and starts a fresh one — so our [cmdbuf computeCommandEncoder] calls below
    // won't collide with PyTorch's internal encoder.
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
    id<MTLCommandBuffer> cmdbuf = stream->commandBuffer();

    const int num_threadgroups = 256;  // tunable; matches CUDA num_blocks

    for (int iter = 0; iter < num_iters; ++iter) {
        int num_blocks_this_iter = std::min(iter + 1, num_s_blocks);
        MIFwdParams params{};
        params.B = B; params.S = S; params.T = T;
        params.px_stride_b = (int)px_c.stride(0);
        params.px_stride_s = (int)px_c.stride(1);
        params.py_stride_b = (int)py_c.stride(0);
        params.py_stride_s = (int)py_c.stride(1);
        params.p_stride_b  = (int)p_c.stride(0);
        params.p_stride_s  = (int)p_c.stride(1);
        params.iter = iter;
        params.num_blocks_this_iter = num_blocks_this_iter;
        params.t_offset = modified ? -1 : 0;

        EncodeForward(cache->fwd, cmdbuf, px_c, py_c, p_c, boundary_i32, ans,
                      params, num_threadgroups);
    }

    // If p was not already contiguous we need to copy the result back
    if (!p.is_contiguous()) p.copy_(p_c);

    // Commit asynchronously (PyTorch's MPS stream flushes on next sync point)
    stream->synchronize(at::mps::SyncType::NONE);

    return ans;
}

// ─────────────────────────────────────────────────────────────────────────────
// MutualInformationBackwardMps
// ─────────────────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> MutualInformationBackwardMps(
    torch::Tensor px, torch::Tensor py,
    torch::optional<torch::Tensor> opt_boundary,
    torch::Tensor p, torch::Tensor ans_grad,
    bool overwrite_ans_grad) {

    TORCH_CHECK(px.scalar_type() == torch::kFloat,
                "MutualInformationBackwardMps only supports float32; got ",
                px.scalar_type());

    auto px_c       = px.contiguous();
    auto py_c       = py.contiguous();
    auto p_c        = p.contiguous();
    auto ans_grad_c = ans_grad.contiguous();  // will be modified in-place if overwrite

    const int B = px_c.size(0), S = px_c.size(1), T = py_c.size(2);
    const bool modified = (px_c.size(2) == (int64_t)T);
    const bool has_boundary = opt_boundary.has_value();

    torch::Tensor boundary_i32;
    if (has_boundary) {
        boundary_i32 = opt_boundary.value().to(torch::kInt32).contiguous();
    } else {
        boundary_i32 = torch::tensor({0, 0, S, T},
                           torch::TensorOptions().dtype(torch::kInt32).device(px.device()))
                           .reshape({1, 4}).expand({B, 4}).contiguous();
    }

    auto opts = torch::TensorOptions().dtype(px.scalar_type()).device(px.device());
    int T1 = T + (modified ? 0 : 1);
    torch::Tensor p_grad  = torch::empty({B, S + 1, T + 1}, opts);
    torch::Tensor px_grad = has_boundary ? torch::zeros({B, S, T1}, opts)
                                         : torch::empty({B, S, T1}, opts);
    torch::Tensor py_grad = has_boundary ? torch::zeros({B, S + 1, T}, opts)
                                         : torch::empty({B, S + 1, T}, opts);

    const int BLOCK_SIZE = 32;
    const int num_s_blocks = S / BLOCK_SIZE + 1;
    const int num_t_blocks = T / BLOCK_SIZE + 1;
    const int num_iters    = num_s_blocks + num_t_blocks - 1;

    MIPipelineCache* cache = GetOrBuildPipelines();
    TORCH_CHECK(cache && cache->bwd, "k2 MPS: failed to build MI backward pipeline");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
    id<MTLCommandBuffer> cmdbuf = stream->commandBuffer();

    const int num_threadgroups = 256;

    for (int iter = num_iters - 1; iter >= 0; --iter) {
        int num_blocks_this_iter = std::min(iter + 1, num_s_blocks);
        MIBwdParams params{};
        params.B = B; params.S = S; params.T = T;
        params.px_stride_b = (int)px_c.stride(0);
        params.px_stride_s = (int)px_c.stride(1);
        params.py_stride_b = (int)py_c.stride(0);
        params.py_stride_s = (int)py_c.stride(1);
        params.p_stride_b  = (int)p_c.stride(0);
        params.p_stride_s  = (int)p_c.stride(1);
        params.iter = iter;
        params.num_blocks_this_iter = num_blocks_this_iter;
        params.neg_t_offset = modified ? 1 : 0;
        params.has_boundary = has_boundary ? 1 : 0;

        EncodeBackward(cache->bwd, cmdbuf, px_c, py_c, p_c, ans_grad_c,
                       p_grad, px_grad, py_grad, boundary_i32, params,
                       num_threadgroups);
    }

    if (overwrite_ans_grad && !ans_grad.is_contiguous())
        ans_grad.copy_(ans_grad_c);

    stream->synchronize(at::mps::SyncType::NONE);

    return {px_grad, py_grad};
}

}  // namespace k2

#endif  // K2_WITH_MPS
