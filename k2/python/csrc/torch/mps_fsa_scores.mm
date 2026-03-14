/**
 * Copyright      2026  k2-fsa Authors
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

// Objective-C++ — must be compiled with clang as .mm on macOS.

// k2 headers first: prevents TORCH_ASSERT_ONLY_METHOD_OPERATORS conflict
// with ATen/native/mps/OperationUtils.h (same fix as mutual_information_mps.mm).
#include "k2/python/csrc/torch/mps_fsa_scores.h"
#include "k2/csrc/mps_utils.h"         // AsMpsTensor / MpsRegistryView
#include "k2/csrc/ragged_ops.h"         // RaggedAxis0Splitter
#include "k2/csrc/pytorch_context.h"    // kMps

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>  // getMTLBufferStorage

#include <limits>

// ---------------------------------------------------------------------------
// Embedded Metal Shading Language kernel source
// ---------------------------------------------------------------------------
// Two kernels: log semiring (logadd) and tropical semiring (max).
// Each thread handles one entering arc in the current BFS batch.
//
// Layout note: FsaArc must match k2's Arc struct exactly (16 bytes):
//   { int src_state, int dest_state, int label, float score }
//
// Atomic float CAS pattern: we store float bits in device atomic_int;
// as_type<float>/as_type<int> perform bitwise reinterpretation without
// any value conversion (equivalent to memcpy / union in C).
static const char *kFsaKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Must match k2::Arc layout: 4 × 4 bytes = 16 bytes, no padding.
struct FsaArc {
    int   src_state;
    int   dest_state;
    int   label;
    float score;
};

// Numerically stable log(exp(a) + exp(b)) without log1p (not in Metal stdlib).
inline float log_add(float a, float b) {
    if (isinf(a) && a < 0.0f) return b;
    if (isinf(b) && b < 0.0f) return a;
    float hi = max(a, b);
    float lo = min(a, b);
    // lo - hi in (-inf, 0], so exp(...) in [0, 1], log(1+...) in [0, log2].
    return hi + log(1.0f + exp(lo - hi));
}

// ---------------------------------------------------------------------------
// fsa_forward_log: log semiring (logsumexp) forward pass.
//
// For each entering arc in the current BFS batch (one thread per arc):
//   candidate = state_scores[src] + arc.score
//   state_scores[dst] = log(exp(state_scores[dst]) + exp(candidate))
//
// state_scores_in and state_scores_out alias the same buffer; reads are from
// previously-committed states (BFS layers guarantee disjoint src/dst sets).
// ---------------------------------------------------------------------------
kernel void fsa_forward_log(
    device const int*      entering_arc_ids [[buffer(0)]],  // batch arc indices
    device const FsaArc*   arcs             [[buffer(1)]],  // all FSA arcs
    device const int*      arc_to_src       [[buffer(2)]],  // fsas row_ids2 (global src state)
    device const float*    scores_in        [[buffer(3)]],  // state_scores (read)
    device float*          scores_out       [[buffer(4)]],  // state_scores (write)
    constant int&          n_arcs           [[buffer(5)]],
    uint                   gid [[thread_position_in_grid]])
{
    if ((int)gid >= n_arcs) return;

    int   arc_idx  = entering_arc_ids[gid];
    int   src      = arc_to_src[arc_idx];          // global source state
    // dest_state in Arc is LOCAL; convert to global via the FSA's state offset.
    // offset = global_src - local_src = arc_to_src[arc_idx] - arcs[arc_idx].src_state
    int   dst      = arcs[arc_idx].dest_state + (src - arcs[arc_idx].src_state);
    float arc_w    = arcs[arc_idx].score;
    float src_s    = scores_in[src];

    if (isinf(src_s) && src_s < 0.0f) return;  // -inf source: no contribution

    float candidate = src_s + arc_w;
    if (isnan(candidate)) return;  // guard: NaN arc scores must not spin the CAS

    // Atomic logadd: CAS loop over bit pattern of the float.
    device atomic_int* slot =
        reinterpret_cast<device atomic_int*>(scores_out + dst);
    int old_bits = atomic_load_explicit(slot, memory_order_relaxed);
    int new_bits;
    do {
        float old_val = as_type<float>(old_bits);
        float new_val = log_add(old_val, candidate);
        new_bits = as_type<int>(new_val);
        if (old_bits == new_bits) return;  // no change (convergence guard)
    } while (!atomic_compare_exchange_weak_explicit(
        slot, &old_bits, new_bits,
        memory_order_relaxed, memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// fsa_forward_tropical: tropical semiring (max) forward pass.
//
// candidate = state_scores[src] + arc.score
// state_scores[dst] = max(state_scores[dst], candidate)
// ---------------------------------------------------------------------------
kernel void fsa_forward_tropical(
    device const int*      entering_arc_ids [[buffer(0)]],
    device const FsaArc*   arcs             [[buffer(1)]],
    device const int*      arc_to_src       [[buffer(2)]],  // global src state
    device const float*    scores_in        [[buffer(3)]],
    device float*          scores_out       [[buffer(4)]],
    constant int&          n_arcs           [[buffer(5)]],
    uint                   gid [[thread_position_in_grid]])
{
    if ((int)gid >= n_arcs) return;

    int   arc_idx  = entering_arc_ids[gid];
    int   src      = arc_to_src[arc_idx];
    // dest_state in Arc is LOCAL; convert to global via FSA offset.
    int   dst      = arcs[arc_idx].dest_state + (src - arcs[arc_idx].src_state);
    float arc_w    = arcs[arc_idx].score;
    float src_s    = scores_in[src];

    if (isinf(src_s) && src_s < 0.0f) return;

    float candidate = src_s + arc_w;
    if (isnan(candidate)) return;  // guard: NaN arc scores must not spin the CAS

    // Atomic max via CAS (float comparison, not int comparison).
    device atomic_int* slot =
        reinterpret_cast<device atomic_int*>(scores_out + dst);
    int cand_bits = as_type<int>(candidate);
    int old_bits  = atomic_load_explicit(slot, memory_order_relaxed);
    while (true) {
        float old_val = as_type<float>(old_bits);
        if (old_val >= candidate) return;  // already at least as large
        if (atomic_compare_exchange_weak_explicit(
                slot, &old_bits, cand_bits,
                memory_order_relaxed, memory_order_relaxed))
            return;  // CAS succeeded; old_bits updated on failure → retry
    }
}
// ===========================================================================
// Priority 6 — Associative-scan (Hillis-Steele) prefix kernels
// ===========================================================================
// For a single-FSA FsaVec with N states, the BFS-level batches have the
// structure: batch_sizes[d] = #arcs entering dest_state_local == d.  Each
// state d corresponds to exactly one matrix M[d] of size N×N in the tropical
// semiring (max-plus).  A Hillis-Steele inclusive prefix scan over M[0..N-1]
// gives all prefix products P[s] = M[s] ⊗ … ⊗ M[0] in ⌈log₂N⌉ steps rather
// than N sequential encoder calls.  State score: α[s] = P[s][s][0].
//
// M[d] semantics (tropical):
//   row == col && col != d → 0.0   (pass-through for other states)
//   row == d               → max arc weight for arcs src→d (or -inf if none)
//   else                   → -INFINITY
// M[0] (start state, never entered): identity (row == col → 0, else -inf).

// assoc_scan_init: initialize T_pow2 matrices of size N×N.
//   mat < t_actual: actual level d — pass-through rows except row==d (all -inf)
//     Special case d==0: identity.
//   mat >= t_actual: padding — identity.
kernel void assoc_scan_init(
    device float*  M        [[buffer(0)]],  // [T_pow2 × N × N]
    constant int&  N        [[buffer(1)]],
    constant int&  t_pow2   [[buffer(2)]],
    constant int&  t_actual [[buffer(3)]],
    uint3          gid      [[thread_position_in_grid]])
{
    int mat = (int)gid.z;
    int row = (int)gid.y;
    int col = (int)gid.x;
    if (mat >= t_pow2 || row >= N || col >= N) return;
    float val;
    if (mat >= t_actual || mat == 0) {
        // Padding or start state: identity
        val = (row == col) ? 0.0f : -INFINITY;
    } else {
        // Actual level d: pass-through for states != d, -inf for state d row
        val = (row == col && row != mat) ? 0.0f : -INFINITY;
    }
    M[mat * N * N + row * N + col] = val;
}

// assoc_scan_build_level: write arc weights into matrix M_level (= M[d]).
// Uses atomic CAS max so concurrent arcs entering the same (dst,src) pair
// correctly keep the largest weight.
kernel void assoc_scan_build_level(
    device const int*      arc_ids [[buffer(0)]],
    device const FsaArc*   arcs    [[buffer(1)]],
    device float*          M_level [[buffer(2)]],  // [N × N] slice for level d
    constant int&          n_arcs  [[buffer(3)]],
    constant int&          N       [[buffer(4)]],
    uint                   gid     [[thread_position_in_grid]])
{
    if ((int)gid >= n_arcs) return;
    int arc_idx = arc_ids[gid];
    int src = arcs[arc_idx].src_state;
    int dst = arcs[arc_idx].dest_state;
    float w  = arcs[arc_idx].score;
    if (isnan(w)) return;  // guard: NaN arc scores must not spin the CAS
    device atomic_int* slot =
        reinterpret_cast<device atomic_int*>(M_level + dst * N + src);
    int w_bits   = as_type<int>(w);
    int old_bits = atomic_load_explicit(slot, memory_order_relaxed);
    while (true) {
        if (as_type<float>(old_bits) >= w) return;
        if (atomic_compare_exchange_weak_explicit(
                slot, &old_bits, w_bits,
                memory_order_relaxed, memory_order_relaxed))
            return;
    }
}

// assoc_scan_prefix_step: one Hillis-Steele step (tropical semiring).
//   For mat >= step_d: buf_out[mat] = buf_in[mat] ⊗ buf_in[mat - step_d]
//   For mat <  step_d: buf_out[mat] = buf_in[mat]   (copy)
// Grid: (N, N, T_pow2)
kernel void assoc_scan_prefix_step(
    device const float* buf_in  [[buffer(0)]],
    device float*       buf_out [[buffer(1)]],
    constant int&       N       [[buffer(2)]],
    constant int&       t_pow2  [[buffer(3)]],
    constant int&       step_d  [[buffer(4)]],
    uint3               gid     [[thread_position_in_grid]])
{
    int mat = (int)gid.z;
    int row = (int)gid.y;
    int col = (int)gid.x;
    if (mat >= t_pow2 || row >= N || col >= N) return;
    int base = mat * N * N + row * N + col;
    if (mat < step_d) {
        buf_out[base] = buf_in[base];
        return;
    }
    // B[mat][row][col] = max_k  A[mat][row][k] + A[mat-step_d][k][col]
    float best = -INFINITY;
    for (int k = 0; k < N; k++) {
        float a = buf_in[mat * N * N + row * N + k];
        float b = buf_in[(mat - step_d) * N * N + k * N + col];
        if (!isinf(a) && !isinf(b))
            best = max(best, a + b);
    }
    buf_out[base] = best;
}

// assoc_scan_extract: alpha[s] = prefix_buf[s][s][0]  for s = 0 .. N-1.
kernel void assoc_scan_extract(
    device const float* prefix_buf [[buffer(0)]],  // [T_pow2 × N × N]
    device float*       alpha      [[buffer(1)]],  // [N]
    constant int&       N          [[buffer(2)]],
    uint                gid        [[thread_position_in_grid]])
{
    int s = (int)gid;
    if (s >= N) return;
    alpha[s] = prefix_buf[s * N * N + s * N + 0];
}

)MSL";

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------
struct FsaPipelines {
    // __unsafe_unretained: skip ARC release on static destructor so we don't
    // send messages to Metal objects after PyTorch has torn down the device.
    __unsafe_unretained id<MTLComputePipelineState> log_fwd           = nil;
    __unsafe_unretained id<MTLComputePipelineState> tropical_fwd      = nil;
    __unsafe_unretained id<MTLComputePipelineState> assoc_init        = nil;
    __unsafe_unretained id<MTLComputePipelineState> assoc_build_level = nil;
    __unsafe_unretained id<MTLComputePipelineState> assoc_prefix_step = nil;
    __unsafe_unretained id<MTLComputePipelineState> assoc_extract     = nil;
};

static FsaPipelines GetOrBuildFsaPipelines(id<MTLDevice> device) {
    static FsaPipelines cache;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kFsaKernelSrc];
        id<MTLLibrary> lib =
            [device newLibraryWithSource:src options:nil error:&err];
        K2_CHECK(lib != nil) << "Metal FSA library compile error: "
            << [[err localizedDescription] UTF8String];

        auto make_pipeline = [&](const char *name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn =
                [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            K2_CHECK(fn != nil) << "Metal function not found: " << name;
            NSError *e2 = nil;
            id<MTLComputePipelineState> ps =
                [device newComputePipelineStateWithFunction:fn error:&e2];
            K2_CHECK(ps != nil) << "Pipeline state error for " << name << ": "
                << [[e2 localizedDescription] UTF8String];
            return ps;
        };

        cache.log_fwd           = make_pipeline("fsa_forward_log");
        cache.tropical_fwd      = make_pipeline("fsa_forward_tropical");
        cache.assoc_init        = make_pipeline("assoc_scan_init");
        cache.assoc_build_level = make_pipeline("assoc_scan_build_level");
        cache.assoc_prefix_step = make_pipeline("assoc_scan_prefix_step");
        cache.assoc_extract     = make_pipeline("assoc_scan_extract");
    });
    return cache;
}

// ---------------------------------------------------------------------------
// Helper: encode one batch kernel onto the current command buffer.
// ---------------------------------------------------------------------------
static void EncodeFsaForwardBatch(
    id<MTLCommandBuffer>       cmd_buf,
    id<MTLComputePipelineState> pipeline,
    // MTL buffers + byte offsets
    id<MTLBuffer> buf_arc_ids,   NSUInteger off_arc_ids,
    id<MTLBuffer> buf_arcs,      NSUInteger off_arcs,
    id<MTLBuffer> buf_src_ids,   NSUInteger off_src_ids,
    id<MTLBuffer> buf_scores,    NSUInteger off_scores,
    int32_t n_arcs)
{
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    K2_CHECK(enc != nil) << "Failed to create MTLComputeCommandEncoder";

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_arc_ids  offset:off_arc_ids  atIndex:0];
    [enc setBuffer:buf_arcs     offset:off_arcs     atIndex:1];
    [enc setBuffer:buf_src_ids  offset:off_src_ids  atIndex:2];
    [enc setBuffer:buf_scores   offset:off_scores   atIndex:3];  // read
    [enc setBuffer:buf_scores   offset:off_scores   atIndex:4];  // write (same)
    [enc setBytes:&n_arcs length:sizeof(int32_t) atIndex:5];

    // 256 threads per threadgroup — works well for arc-level parallelism.
    static const NSUInteger kThreadsPerGroup = 256;
    NSUInteger num_groups =
        ((NSUInteger)n_arcs + kThreadsPerGroup - 1) / kThreadsPerGroup;

    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
    [enc endEncoding];
}

// ---------------------------------------------------------------------------
// GetForwardScoresMps — public entry point
// ---------------------------------------------------------------------------
namespace k2 {
namespace mps_ops {

Array1<float> GetForwardScoresMps(FsaVec &fsas,
                                   Ragged<int32_t> &entering_arc_batches_cpu,
                                   bool log_semiring) {
    ContextPtr &c = fsas.Context();
    K2_CHECK_EQ(c->GetDeviceType(), kMps);

    int32_t num_fsas      = fsas.Dim0();
    int32_t num_states    = fsas.TotSize(1);
    int32_t num_arcs      = fsas.TotSize(2);
    int32_t num_batches   = entering_arc_batches_cpu.Dim0();
    int32_t total_arc_ids = entering_arc_batches_cpu.NumElements();

    const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();  // MPS ptr
    const int32_t *fsas_row_ids2   = fsas.RowIds(2).Data();      // MPS ptr
    const Arc     *arcs_ptr        = fsas.values.Data();          // MPS ptr

    // ------------------------------------------------------------------
    // 1. Allocate state_scores on MPS and initialise via ATen ops.
    //    Array1<float>(c, n, val) uses K2_EVAL → CPU sequential for MPS;
    //    we skip that and do the fill directly through ATen.
    // ------------------------------------------------------------------
    Array1<float> state_scores(c, num_states);  // uninitialized allocation

    auto scores_t = AsMpsTensor(state_scores.Data(),
                                 (int64_t)num_states, torch::kFloat);
    scores_t.fill_(-std::numeric_limits<float>::infinity());

    // Set start state of each non-empty FSA to 0.
    // row_splits1[i] is the global state index of FSA i's start state.
    if (num_fsas > 0) {
        auto row_splits_t = AsMpsTensor(fsa_row_splits1,
                                         (int64_t)(num_fsas + 1));  // int32
        auto starts   = row_splits_t.slice(0, 0, num_fsas);
        auto ends     = row_splits_t.slice(0, 1, num_fsas + 1);
        auto nonempty = starts.ne(ends);
        auto valid    = starts.masked_select(nonempty).to(torch::kInt64);
        scores_t.index_put_({valid}, 0.0f);
    }

    if (num_arcs == 0 || num_batches == 0 || total_arc_ids == 0)
        return state_scores;

    // ------------------------------------------------------------------
    // 2. Obtain Metal device, build (or reuse) pipelines.
    // ------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    K2_CHECK(device != nil) << "No Metal device available";
    FsaPipelines pipelines = GetOrBuildFsaPipelines(device);

    // ------------------------------------------------------------------
    // 3. Get persistent MTL buffers for the MPS-resident arrays.
    // ------------------------------------------------------------------
    // Arc structs: view as bytes so offset is in bytes directly.
    auto arcs_t = AsMpsTensor(arcs_ptr,
                               (int64_t)num_arcs * (int64_t)sizeof(Arc),
                               torch::kByte);
    id<MTLBuffer> buf_arcs   = at::native::mps::getMTLBufferStorage(arcs_t);
    NSUInteger    off_arcs   = (NSUInteger)(arcs_t.storage_offset());

    auto src_t = AsMpsTensor(fsas_row_ids2, (int64_t)num_arcs);  // int32
    id<MTLBuffer> buf_src    = at::native::mps::getMTLBufferStorage(src_t);
    NSUInteger    off_src    = (NSUInteger)(src_t.storage_offset() * 4);

    id<MTLBuffer> buf_scores = at::native::mps::getMTLBufferStorage(scores_t);
    NSUInteger    off_scores = (NSUInteger)(scores_t.storage_offset() * 4);

    // ------------------------------------------------------------------
    // 4. Copy all CPU entering-arc IDs to MPS once.
    //    RaggedAxis0Splitter (CPU context) gives per-batch arc_begin offsets
    //    into this flat array, so we index into it by byte offset.
    // ------------------------------------------------------------------
    const int32_t *cpu_arc_ids_ptr = entering_arc_batches_cpu.values.Data();
    torch::Tensor arc_ids_mps =
        torch::from_blob((void *)cpu_arc_ids_ptr, {(int64_t)total_arc_ids},
                         torch::TensorOptions()
                             .dtype(torch::kInt32)
                             .device(torch::kCPU))
            .to(at::Device(at::kMPS));
    id<MTLBuffer> buf_all_arc_ids =
        at::native::mps::getMTLBufferStorage(arc_ids_mps);
    NSUInteger base_off_arc_ids =
        (NSUInteger)(arc_ids_mps.storage_offset() * 4);

    id<MTLComputePipelineState> pipeline =
        log_semiring ? pipelines.log_fwd : pipelines.tropical_fwd;

    // ------------------------------------------------------------------
    // 5. Flush any pending PyTorch MPS encoder, then encode all batch
    //    kernels onto a single command buffer.  Metal executes them in
    //    submission order, which enforces BFS layer dependencies.
    // ------------------------------------------------------------------
    auto *stream = at::mps::getCurrentMPSStream();
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);

    id<MTLCommandBuffer> cmd_buf = stream->commandBuffer();
    K2_CHECK(cmd_buf != nil) << "Failed to get MPS command buffer";

    // entering_arc_batches_cpu has CPU context: RaggedAxis0Splitter works.
    RaggedAxis0Splitter<int32_t> splitter(entering_arc_batches_cpu);

    for (int32_t i = 0; i < num_batches; ++i) {
        int32_t arc_begin;
        Ragged<int32_t> batch = splitter.GetElement(i, &arc_begin);
        int32_t n_arcs_batch  = batch.NumElements();
        if (n_arcs_batch == 0) continue;

        // Byte offset into arc_ids_mps for this batch.
        NSUInteger off_arc_ids = base_off_arc_ids + (NSUInteger)(arc_begin * 4);

        EncodeFsaForwardBatch(
            cmd_buf, pipeline,
            buf_all_arc_ids, off_arc_ids,
            buf_arcs,        off_arcs,
            buf_src,         off_src,
            buf_scores,      off_scores,
            n_arcs_batch);
    }

    // Commit asynchronously; PyTorch will sync when the result is needed.
    stream->synchronize(at::mps::SyncType::NONE);

    return state_scores;
}

// ---------------------------------------------------------------------------
// GetForwardScoresMpsNative — zero-copy variant.
//
// Accepts sorted_arc_ids already resident on MPS: avoids the full FSA
// CPU copy required by GetForwardScoresMps.  The caller provides a CPU-side
// batch_sizes vector (one int per BFS level) derived cheaply from the
// arc.dest_state_local column, exploiting the k2 invariant that FSAs are
// topologically sorted (src_state_local < dest_state_local for every arc).
// ---------------------------------------------------------------------------
Array1<float> GetForwardScoresMpsNative(
    FsaVec &fsas,
    torch::Tensor sorted_arc_ids,
    const std::vector<int32_t> &batch_sizes,
    bool log_semiring) {
    ContextPtr &c = fsas.Context();
    K2_CHECK_EQ(c->GetDeviceType(), kMps);

    int32_t num_fsas   = fsas.Dim0();
    int32_t num_states = fsas.TotSize(1);
    int32_t num_arcs   = fsas.TotSize(2);

    const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();  // MPS ptr
    const int32_t *fsas_row_ids2   = fsas.RowIds(2).Data();      // MPS ptr
    const Arc     *arcs_ptr        = fsas.values.Data();          // MPS ptr

    // ------------------------------------------------------------------
    // 1. Allocate state_scores on MPS and initialise via ATen ops.
    // ------------------------------------------------------------------
    Array1<float> state_scores(c, num_states);
    auto scores_t = AsMpsTensor(state_scores.Data(),
                                 (int64_t)num_states, torch::kFloat);
    scores_t.fill_(-std::numeric_limits<float>::infinity());

    if (num_fsas > 0) {
        auto row_splits_t = AsMpsTensor(fsa_row_splits1,
                                         (int64_t)(num_fsas + 1));
        auto starts   = row_splits_t.slice(0, 0, num_fsas);
        auto ends     = row_splits_t.slice(0, 1, num_fsas + 1);
        auto nonempty = starts.ne(ends);
        auto valid    = starts.masked_select(nonempty).to(torch::kInt64);
        scores_t.index_put_({valid}, 0.0f);
    }

    int32_t total_arc_ids = (int32_t)sorted_arc_ids.numel();
    if (total_arc_ids == 0 || batch_sizes.empty())
        return state_scores;

    // ------------------------------------------------------------------
    // 2. Build (or reuse) Metal pipelines.
    // ------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    K2_CHECK(device != nil) << "No Metal device available";
    FsaPipelines pipelines = GetOrBuildFsaPipelines(device);

    // ------------------------------------------------------------------
    // 3. Get persistent MTL buffers for the MPS-resident arrays.
    // ------------------------------------------------------------------
    auto arcs_t = AsMpsTensor(arcs_ptr,
                               (int64_t)num_arcs * (int64_t)sizeof(Arc),
                               torch::kByte);
    id<MTLBuffer> buf_arcs   = at::native::mps::getMTLBufferStorage(arcs_t);
    NSUInteger    off_arcs   = (NSUInteger)(arcs_t.storage_offset());

    auto src_t = AsMpsTensor(fsas_row_ids2, (int64_t)num_arcs);
    id<MTLBuffer> buf_src    = at::native::mps::getMTLBufferStorage(src_t);
    NSUInteger    off_src    = (NSUInteger)(src_t.storage_offset() * 4);

    id<MTLBuffer> buf_scores = at::native::mps::getMTLBufferStorage(scores_t);
    NSUInteger    off_scores = (NSUInteger)(scores_t.storage_offset() * 4);

    // ------------------------------------------------------------------
    // 4. sorted_arc_ids is already on MPS — obtain buffer directly.
    //    No CPU→MPS copy needed (this is the key gain over GetForwardScoresMps).
    // ------------------------------------------------------------------
    id<MTLBuffer> buf_arc_ids =
        at::native::mps::getMTLBufferStorage(sorted_arc_ids);
    NSUInteger base_off_arc_ids =
        (NSUInteger)(sorted_arc_ids.storage_offset() * 4);

    id<MTLComputePipelineState> pipeline =
        log_semiring ? pipelines.log_fwd : pipelines.tropical_fwd;

    // ------------------------------------------------------------------
    // 5. Flush any pending PyTorch MPS encoder, then encode all batch
    //    kernels onto a single command buffer.
    // ------------------------------------------------------------------
    auto *stream = at::mps::getCurrentMPSStream();
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);

    id<MTLCommandBuffer> cmd_buf = stream->commandBuffer();
    K2_CHECK(cmd_buf != nil) << "Failed to get MPS command buffer";

    int32_t arc_cursor = 0;
    for (int32_t n_arcs_batch : batch_sizes) {
        if (n_arcs_batch > 0) {
            NSUInteger off_arc_ids =
                base_off_arc_ids + (NSUInteger)(arc_cursor * 4);
            EncodeFsaForwardBatch(
                cmd_buf, pipeline,
                buf_arc_ids, off_arc_ids,
                buf_arcs,    off_arcs,
                buf_src,     off_src,
                buf_scores,  off_scores,
                n_arcs_batch);
        }
        arc_cursor += n_arcs_batch;
    }

    stream->synchronize(at::mps::SyncType::NONE);
    return state_scores;
}

// ---------------------------------------------------------------------------
// GetForwardScoresMpsAssocScan — O(log N) associative-scan forward pass.
//
// Applies a Hillis-Steele inclusive prefix scan over N per-state transition
// matrices (one N×N matrix per dest_state_local), reducing the number of
// Metal compute encoder calls from N to ⌈log₂(N)⌉.
//
// Conditions for using this path (falls back to GetForwardScoresMpsNative):
//   • num_fsas == 1 (single FSA — multi-FSA requires independent scans)
//   • 4 ≤ num_states ≤ 128 (dense matrix fits GPU cache; below 4 sequential wins)
//   • !log_semiring (tropical / Viterbi only for now)
//
// Memory: two ping-pong float buffers of shape [T_pow2 × N × N] on MPS,
//   where T_pow2 = next power of 2 ≥ N.  Max: 2 × 128 × 128² × 4 = 16 MB.
// ---------------------------------------------------------------------------
Array1<float> GetForwardScoresMpsAssocScan(
    FsaVec &fsas,
    torch::Tensor sorted_arc_ids,
    const std::vector<int32_t> &batch_sizes,
    bool log_semiring)
{
    ContextPtr &c = fsas.Context();
    K2_CHECK_EQ(c->GetDeviceType(), kMps);

    int32_t num_fsas   = fsas.Dim0();
    int32_t num_states = fsas.TotSize(1);
    int32_t num_arcs   = fsas.TotSize(2);

    // --- threshold check — fall back to native sequential path ---
    if (log_semiring || num_fsas != 1 || num_states < 4 || num_states > 128) {
        return GetForwardScoresMpsNative(
            fsas, sorted_arc_ids, batch_sizes, log_semiring);
    }

    const int32_t N = num_states;

    // T_pow2: next power of 2 >= N (Hillis-Steele requires power-of-2 array).
    int32_t T_pow2 = 1;
    while (T_pow2 < N) T_pow2 <<= 1;

    const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();  // MPS ptr
    const Arc     *arcs_ptr        = fsas.values.Data();          // MPS ptr

    // ------------------------------------------------------------------
    // 1.  Allocate two ping-pong buffers on MPS (T_pow2 × N × N each).
    // ------------------------------------------------------------------
    int64_t mat_elems = (int64_t)T_pow2 * N * N;
    torch::Tensor buf_a = torch::empty(
        {mat_elems}, torch::TensorOptions().dtype(torch::kFloat).device(at::kMPS));
    torch::Tensor buf_b = torch::empty_like(buf_a);

    auto arcs_t = AsMpsTensor(arcs_ptr,
                               (int64_t)num_arcs * (int64_t)sizeof(Arc),
                               torch::kByte);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    K2_CHECK(device != nil) << "No Metal device available";
    FsaPipelines pipes = GetOrBuildFsaPipelines(device);

    // ------------------------------------------------------------------
    // 2.  Flush any pending ATen encoder, then begin encoding.
    // ------------------------------------------------------------------
    auto *stream = at::mps::getCurrentMPSStream();
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
    id<MTLCommandBuffer> cmd = stream->commandBuffer();
    K2_CHECK(cmd != nil) << "Failed to get MPS command buffer";

    id<MTLBuffer> mtl_a   = at::native::mps::getMTLBufferStorage(buf_a);
    id<MTLBuffer> mtl_b   = at::native::mps::getMTLBufferStorage(buf_b);
    id<MTLBuffer> buf_arcs = at::native::mps::getMTLBufferStorage(arcs_t);
    NSUInteger    off_arcs = (NSUInteger)arcs_t.storage_offset();  // byte offset

    id<MTLBuffer> buf_arc_ids =
        at::native::mps::getMTLBufferStorage(sorted_arc_ids);
    NSUInteger base_off_arc_ids =
        (NSUInteger)(sorted_arc_ids.storage_offset() * 4);

    // Helper: dispatch a 3-D grid (X × Y × Z) threadgroups of 1 thread each.
    // For small N this is fine; threadgroup size = 1 avoids occupancy issues.
    auto dispatch3 = [&](id<MTLComputePipelineState> pso,
                         NSUInteger X, NSUInteger Y, NSUInteger Z,
                         /* buffer bindings set by caller */ int) {
        // caller sets buffers before calling this lambda — not feasible as lambda.
        // Instead, inline the dispatch pattern at each call site below.
        (void)pso; (void)X; (void)Y; (void)Z;
    };
    (void)dispatch3;  // suppress unused warning; we inline below.

    // ------------------------------------------------------------------
    // 3.  assoc_scan_init: fill buf_a (both buffers, but only buf_a matters
    //     since buf_b gets fully overwritten by the first prefix step).
    // ------------------------------------------------------------------
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipes.assoc_init];
        [enc setBuffer:mtl_a offset:(NSUInteger)(buf_a.storage_offset() * 4) atIndex:0];
        int32_t n_val = N, tp_val = T_pow2, ta_val = N;  // t_actual = N states
        [enc setBytes:&n_val  length:4 atIndex:1];
        [enc setBytes:&tp_val length:4 atIndex:2];
        [enc setBytes:&ta_val length:4 atIndex:3];
        // Grid: (N, N, T_pow2) with threadgroup size (1,1,1).
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, (NSUInteger)N, (NSUInteger)T_pow2)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
    }

    // ------------------------------------------------------------------
    // 4.  assoc_scan_build_level: fill arc weights into each matrix M[d].
    //     sorted_arc_ids is already on MPS (zero-copy from Priority 4).
    // ------------------------------------------------------------------
    int32_t arc_cursor = 0;
    for (int32_t d = 0; d < N; ++d) {
        int32_t n_arcs_d = (d < (int32_t)batch_sizes.size()) ? batch_sizes[d] : 0;
        if (n_arcs_d > 0) {
            NSUInteger off_arc_ids_d = base_off_arc_ids + (NSUInteger)(arc_cursor * 4);
            // M[d] starts at buf_a offset: d * N * N floats
            NSUInteger off_m_d = (NSUInteger)(buf_a.storage_offset() * 4) +
                                 (NSUInteger)(d * N * N * 4);

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipes.assoc_build_level];
            [enc setBuffer:buf_arc_ids offset:off_arc_ids_d atIndex:0];
            [enc setBuffer:buf_arcs    offset:off_arcs       atIndex:1];
            [enc setBuffer:mtl_a       offset:off_m_d        atIndex:2];
            int32_t n_val = N;
            [enc setBytes:&n_arcs_d length:4 atIndex:3];
            [enc setBytes:&n_val    length:4 atIndex:4];
            NSUInteger tg = (NSUInteger)n_arcs_d;
            [enc dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }
        arc_cursor += n_arcs_d;
    }

    // ------------------------------------------------------------------
    // 5.  Hillis-Steele prefix scan: log2(T_pow2) steps.
    //     Ping-pong between buf_a (read) and buf_b (write), then swap.
    // ------------------------------------------------------------------
    id<MTLBuffer> cur_in  = mtl_a;
    id<MTLBuffer> cur_out = mtl_b;
    NSUInteger in_base  = (NSUInteger)(buf_a.storage_offset() * 4);
    NSUInteger out_base = (NSUInteger)(buf_b.storage_offset() * 4);

    for (int32_t step_d = 1; step_d < T_pow2; step_d <<= 1) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipes.assoc_prefix_step];
        [enc setBuffer:cur_in  offset:in_base  atIndex:0];
        [enc setBuffer:cur_out offset:out_base atIndex:1];
        int32_t n_val = N, tp_val = T_pow2, sd_val = step_d;
        [enc setBytes:&n_val  length:4 atIndex:2];
        [enc setBytes:&tp_val length:4 atIndex:3];
        [enc setBytes:&sd_val length:4 atIndex:4];
        // Grid: (N, N, T_pow2) — each thread handles one cell of one matrix.
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N,
                                              (NSUInteger)N,
                                              (NSUInteger)T_pow2)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        // Swap ping-pong.
        std::swap(cur_in,  cur_out);
        std::swap(in_base, out_base);
    }
    // After the loop, cur_in holds the final prefix products.

    // ------------------------------------------------------------------
    // 6.  Extract state scores: alpha[s] = prefix[s][s][0].
    // ------------------------------------------------------------------
    Array1<float> state_scores(c, num_states);
    auto scores_t = AsMpsTensor(state_scores.Data(),
                                 (int64_t)num_states, torch::kFloat);
    id<MTLBuffer> buf_scores = at::native::mps::getMTLBufferStorage(scores_t);
    NSUInteger    off_scores = (NSUInteger)(scores_t.storage_offset() * 4);

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipes.assoc_extract];
        [enc setBuffer:cur_in    offset:in_base  atIndex:0];
        [enc setBuffer:buf_scores offset:off_scores atIndex:1];
        int32_t n_val = N;
        [enc setBytes:&n_val length:4 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
    }

    stream->synchronize(at::mps::SyncType::NONE);
    return state_scores;
}

}  // namespace mps_ops
}  // namespace k2
