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

// MPS-accelerated GetForwardScores for float32 / log semiring.
// Only compiled when K2_WITH_MPS is defined.
//
// The Metal kernel dispatches one thread per entering arc per BFS batch.
// Each thread atomically updates state_scores[dst] via a CAS logadd loop,
// avoiding the intermediate entering_arc_batch_scores array used by the
// CPU sequential path.  Sequential BFS ordering is maintained by the
// Metal command queue's in-order execution — no CPU barriers needed.
#pragma once

#ifdef K2_WITH_MPS

// k2 headers must be included before Metal/MPS headers to prevent
// TORCH_ASSERT_ONLY_METHOD_OPERATORS conflicts with aten_interned_strings.h.
#include <vector>
#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include <torch/extension.h>  // NOLINT(build/include_order)

namespace k2 {
namespace mps_ops {

// GetForwardScoresMps: MPS-native forward pass for float32 (log or tropical).
//
// Dispatches a Metal compute kernel for each BFS batch.  The kernel
// evaluates  state_scores[dst] = logsumexp(state_scores[src] + arc.score)
// (log semiring) or max(state_scores[dst], state_scores[src] + arc.score)
// (tropical semiring) over all entering arcs in parallel using atomic CAS.
//
// `entering_arc_batches_cpu` must have CPU context — compute it on a CPU copy
// of the FSA.  `fsas` must have MPS context.  The arc IDs are copied to MPS
// once before the kernel loop.
Array1<float> GetForwardScoresMps(FsaVec &fsas,
                                   Ragged<int32_t> &entering_arc_batches_cpu,
                                   bool log_semiring);

// GetForwardScoresMpsNative: zero-copy MPS-native forward pass.
//
// Like GetForwardScoresMps but accepts pre-sorted arc IDs already on MPS and
// a CPU-side batch_sizes vector.  This avoids the full FSA CPU copy: the
// caller only transfers arc.dest_state (4 bytes × num_arcs) to CPU for
// sorting, then moves the sorted indices (4 bytes × num_arcs) back to MPS.
//
// `sorted_arc_ids` — int32 MPS tensor: arc indices sorted by BFS level.
// `batch_sizes`    — number of arcs per BFS level (may include zeros).
// `fsas`           — must have MPS context.
Array1<float> GetForwardScoresMpsNative(
    FsaVec &fsas,
    torch::Tensor sorted_arc_ids,
    const std::vector<int32_t> &batch_sizes,
    bool log_semiring);

// GetForwardScoresMpsAssocScan: O(log N) associative-scan forward pass.
//
// Uses a Hillis-Steele inclusive prefix scan over N per-state transition
// matrices (N×N each, tropical semiring) to compute all state forward scores
// in ⌈log₂N⌉ Metal encoder calls instead of N sequential calls.
//
// Falls back to GetForwardScoresMpsNative when:
//   • log_semiring is true (log semiring not yet implemented)
//   • num_fsas != 1 (multi-FSA not supported — would need independent scans)
//   • num_states < 4 or > 128 (outside the beneficial range)
//
// `sorted_arc_ids` — int32 MPS tensor: arc indices sorted by dest_state_local.
// `batch_sizes`    — number of arcs per dest_state_local value (one per state).
// `fsas`           — must have MPS context, exactly 1 FSA.
Array1<float> GetForwardScoresMpsAssocScan(
    FsaVec &fsas,
    torch::Tensor sorted_arc_ids,
    const std::vector<int32_t> &batch_sizes,
    bool log_semiring);

}  // namespace mps_ops
}  // namespace k2

#endif  // K2_WITH_MPS
