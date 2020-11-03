/**
 * @brief fsa_algo  Implementation of FSA algorithm wrappers from fsa_algo.h

 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/intersect.h"
#include "k2/csrc/host/topsort.h"
#include "k2/csrc/host_shim.h"

// this contains a subset of the algorithms in fsa_algo.h; currently it just
// contains one that are wrappings of the corresponding algorithms in
// host/.
namespace k2 {

bool RecursionWrapper(bool (*f)(Fsa &, Fsa *, Array1<int32_t> *), Fsa &src,
                      Fsa *dest, Array1<int32_t> *arc_map) {
  // src is actually an FsaVec.  Just recurse for now.
  int32_t num_fsas = src.shape.Dim0();
  std::vector<Fsa> srcs(num_fsas), dests(num_fsas);
  std::vector<Array1<int32_t>> arc_maps(num_fsas);
  for (int32_t i = 0; i < num_fsas; ++i) {
    srcs[i] = src.Index(0, i);
    // Recurse.
    if (!f(srcs[i], &(dests[i]), (arc_map ? &(arc_maps[i]) : nullptr)))
      return false;
  }
  *dest = Stack(0, num_fsas, dests.data());
  if (arc_map) *arc_map = Append(num_fsas, arc_maps.data());
  return true;
}

bool Connect(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map /*=nullptr*/) {
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapper(Connect, src, dest, arc_map);
  }

  k2host::Fsa host_fsa = FsaToHostFsa(src);
  k2host::Connection c(host_fsa);
  k2host::Array2Size<int32_t> size;
  c.GetSizes(&size);
  FsaCreator creator(size);
  k2host::Fsa host_dest_fsa = creator.GetHostFsa();
  int32_t *arc_map_data = nullptr;
  if (arc_map != nullptr) {
    *arc_map = Array1<int32_t>(src.Context(), size.size2);
    arc_map_data = arc_map->Data();
  }
  bool ans = c.GetOutput(&host_dest_fsa, arc_map_data);
  *dest = creator.GetFsa();
  return ans;
}

bool HostTopSort(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map /*=nullptr*/) {
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapper(HostTopSort, src, dest, arc_map);
  }

  k2host::Fsa host_fsa = FsaToHostFsa(src);
  k2host::TopSorter sorter(host_fsa);
  k2host::Array2Size<int32_t> size;
  sorter.GetSizes(&size);
  FsaCreator creator(size);
  k2host::Fsa host_dest_fsa = creator.GetHostFsa();
  int32_t *arc_map_data = nullptr;
  if (arc_map != nullptr) {
    *arc_map = Array1<int32_t>(src.Context(), size.size2);
    arc_map_data = arc_map->Data();
  }
  bool ans = sorter.GetOutput(&host_dest_fsa, arc_map_data);
  *dest = creator.GetFsa();
  return ans;
}

bool Intersect(FsaOrVec &a_fsas, FsaOrVec &b_fsas,
               bool treat_epsilons_specially, FsaVec *out,
               Array1<int32_t> *arc_map_a, Array1<int32_t> *arc_map_b) {
  K2_CHECK(a_fsas.NumAxes() >= 2 && a_fsas.NumAxes() <= 3);
  K2_CHECK(b_fsas.NumAxes() >= 2 && b_fsas.NumAxes() <= 3);
  ContextPtr c = a_fsas.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  if (a_fsas.NumAxes() == 2) {
    FsaVec a_fsas_vec = FsaToFsaVec(a_fsas);
    return Intersect(a_fsas_vec, b_fsas, treat_epsilons_specially, out,
                     arc_map_a, arc_map_b);
  }
  if (b_fsas.NumAxes() == 2) {
    FsaVec b_fsas_vec = FsaToFsaVec(b_fsas);
    return Intersect(a_fsas, b_fsas_vec, treat_epsilons_specially, out,
                     arc_map_a, arc_map_b);
  }

  int32_t num_fsas_a = a_fsas.Dim0(), num_fsas_b = b_fsas.Dim0();
  K2_CHECK_GT(num_fsas_a, 0);
  K2_CHECK_GT(num_fsas_b, 0);
  int32_t stride_a = 1, stride_b = 1;
  if (num_fsas_a != num_fsas_b) {
    if (num_fsas_a == 1) {
      stride_a = 0;
    } else if (num_fsas_b == 1) {
      stride_b = 0;
    } else {
      K2_CHECK_EQ(num_fsas_a, num_fsas_b);
    }
    // the check on the previous line will fail.
  }
  int32_t num_fsas = std::max(num_fsas_a, num_fsas_b);

  std::vector<std::unique_ptr<k2host::Intersection>> intersections(num_fsas);
  std::vector<k2host::Array2Size<int32_t>> sizes(num_fsas);
  for (int32_t i = 0; i < num_fsas; ++i) {
    k2host::Fsa host_fsa_a = FsaVecToHostFsa(a_fsas, i * stride_a),
                host_fsa_b = FsaVecToHostFsa(b_fsas, i * stride_b);
    intersections[i] = std::make_unique<k2host::Intersection>(
        host_fsa_a, host_fsa_b, treat_epsilons_specially);
    intersections[i]->GetSizes(&(sizes[i]));
  }
  FsaVecCreator creator(sizes);
  int32_t num_arcs = creator.NumArcs();

  if (arc_map_a) *arc_map_a = Array1<int32_t>(c, num_arcs);
  if (arc_map_b) *arc_map_b = Array1<int32_t>(c, num_arcs);

  // the following few lines will allow us to add suitable offsets to the
  // `arc_map`.
  Array1<int32_t> a_fsas_row_splits12 =
                      a_fsas.RowSplits(2)[a_fsas.RowSplits(1)],
                  b_fsas_row_splits12 =
                      b_fsas.RowSplits(2)[b_fsas.RowSplits(1)];
  const int32_t *a_fsas_row_splits12_data = a_fsas_row_splits12.Data(),
                *b_fsas_row_splits12_data = b_fsas_row_splits12.Data();

  bool ok = true;
  for (int32_t i = 0; i < num_fsas; ++i) {
    k2host::Fsa host_fsa_out = creator.GetHostFsa(i);
    int32_t arc_offset = creator.GetArcOffsetFor(i);
    int32_t *this_arc_map_a =
                (arc_map_a ? arc_map_a->Data() + arc_offset : nullptr),
            *this_arc_map_b =
                (arc_map_b ? arc_map_b->Data() + arc_offset : nullptr);
    bool ans = intersections[i]->GetOutput(&host_fsa_out, this_arc_map_a,
                                           this_arc_map_b);
    ok = ok && ans;
    int32_t this_num_arcs = creator.GetArcOffsetFor(i + 1) - arc_offset;
    if (arc_map_a) {
      int32_t arc_offset_a = a_fsas_row_splits12_data[i * stride_a];
      for (int32_t i = 0; i < this_num_arcs; i++)
        this_arc_map_a[i] += arc_offset_a;
    }
    if (arc_map_b) {
      int32_t arc_offset_b = b_fsas_row_splits12_data[i * stride_b];
      for (int32_t i = 0; i < this_num_arcs; i++)
        this_arc_map_b[i] += arc_offset_b;
    }
  }
  *out = creator.GetFsaVec();
  return ok;
}

Fsa LinearFsa(const Array1<int32_t> &symbols) {
  ContextPtr &c = symbols.Context();
  int32_t n = symbols.Dim(), num_states = n + 2, num_arcs = n + 1;
  Array1<int32_t> row_splits1 = Range(c, num_states + 1, 0),
                  row_ids1 = Range(c, num_arcs, 0);
  int32_t *row_splits1_data = row_splits1.Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();
  const int32_t *symbols_data = symbols.Data();
  auto lambda_set_arcs = [=] __host__ __device__(int32_t arc_idx01) -> void {
    int32_t src_state = arc_idx01, dest_state = arc_idx01 + 1,
            // -1 == kFinalSymbol
        symbol = (arc_idx01 < n ? symbols_data[arc_idx01] : -1);
    if (arc_idx01 < n) K2_CHECK_NE(symbol, -1);
    float score = 0.0;
    arcs_data[arc_idx01] = Arc(src_state, dest_state, symbol, score);
    // the final state has no leaving arcs.
    if (arc_idx01 == 0) row_splits1_data[num_states] = num_arcs;
  };
  Eval(c, num_arcs, lambda_set_arcs);
  return Ragged<Arc>(RaggedShape2(&row_splits1, &row_ids1, num_arcs), arcs);
}

FsaVec LinearFsas(Ragged<int32_t> &symbols) {
  K2_CHECK_EQ(symbols.NumAxes(), 2);
  ContextPtr &c = symbols.Context();

  // if there are n symbols, there are n+2 states and n+1 arcs.
  RaggedShape states_shape = ChangeSublistSize(symbols.shape, 2);

  int32_t num_states = states_shape.NumElements(),
          num_arcs = symbols.NumElements() + symbols.Dim0();

  // row_splits2 maps from state_idx01 to arc_idx012; row_ids2 does the reverse.
  // We'll set them in the lambda below.
  Array1<int32_t> row_splits2(c, num_states + 1), row_ids2(c, num_arcs);

  int32_t *row_ids2_data = row_ids2.Data(),
          *row_splits2_data = row_splits2.Data();
  const int32_t *row_ids1_data = states_shape.RowIds(1).Data(),
                *row_splits1_data = states_shape.RowSplits(1).Data(),
                *symbols_data = symbols.values.Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();
  auto lambda = [=] __host__ __device__(int32_t state_idx01) -> void {
    int32_t fsa_idx0 = row_ids1_data[state_idx01],
            state_idx0x = row_splits1_data[fsa_idx0],
            next_state_idx0x = row_splits1_data[fsa_idx0 + 1],
            idx1 = state_idx01 - state_idx0x;

    // the following works because each FSA has one fewer arcs than states.
    int32_t arc_idx0xx = state_idx0x - fsa_idx0,
            next_arc_idx0xx = next_state_idx0x - (fsa_idx0 + 1),
            // the following may look a bit wrong.. here, the idx1 is the same
            // as the idx12 if the arc exists, because each state has one arc
            // leaving it (except the last state).
        arc_idx012 = arc_idx0xx + idx1;
    // the following works because each FSA has one fewer symbols than arcs
    // (however it doesn't work for the last arc of each FSA; we check below.)
    int32_t symbol_idx01 = arc_idx012 - fsa_idx0;
    if (arc_idx012 < next_arc_idx0xx) {
      int32_t src_state = idx1, dest_state = idx1 + 1,
              symbol =
                  (arc_idx012 + 1 < next_arc_idx0xx ? symbols_data[symbol_idx01]
                                                    : -1);  // kFinalSymbol
      float score = 0.0;
      arcs_data[arc_idx012] = Arc(src_state, dest_state, symbol, score);
      row_ids2_data[arc_idx012] = state_idx01;
    } else {
      // The following ensures that the last element of row_splits1_data
      // (i.e. row_splits1[num_states]) is set to num_arcs.  It also writes
      // something unnecessary for the last state of each FSA but the last one,
      // which will cause 2 threads to write the same item to the same location.
      // Note that there is no arc with index `arc_idx01`, if you reach here.
      row_splits2_data[state_idx01 + 1] = arc_idx012;
    }
    row_splits2_data[state_idx01] = arc_idx012;
  };
  Eval(c, num_states, lambda);

  return Ragged<Arc>(
      RaggedShape3(&states_shape.RowSplits(1), &states_shape.RowIds(1),
                   num_states, &row_splits2, &row_ids2, num_arcs),
      arcs);
}

namespace {
struct ArcComparer {
  __host__ __device__ __forceinline__ bool operator()(const Arc &lhs,
                                                      const Arc &rhs) const {
    // Compares `label` first, then `dest_state`;
    // compare label as unsigned so -1 comes after other symbols, since some
    // algorithms may require epsilons to be first.
    if (lhs.label != rhs.label)
      return static_cast<uint32_t>(lhs.label) <
             static_cast<uint32_t>(rhs.label);
    else
      return lhs.dest_state < rhs.dest_state;
  }
};
}  // namespace

void ArcSort(Fsa *fsa) {
  if (fsa->NumAxes() < 2) return;  // it is empty
  SortSublists<Arc, ArcComparer>(fsa);
}

void ArcSort(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map /*= nullptr*/) {
  if (!src.values.IsValid()) return;

  if (arc_map != nullptr)
    *arc_map = Array1<int32_t>(src.Context(), src.NumElements());

  Fsa tmp(src.shape, src.values.Clone());
  SortSublists<Arc, ArcComparer>(&tmp, arc_map);
  *dest = tmp;
}

// TODO(fangjun): use the following method suggested by Dan
//
// ... incidentally, it's possible to further optimize this so the run
// time is less than linear, by using methods similar to what I use
// in GetStateBatches(); imagine computing a table that instead of
// the best traceback, is the best 2-step traceback; and then the 4-step
// traceback, and so on. There's no need for this right now, since the
// forward-pass algorithm is already at least linear-time in the length
// of this path. But we can consider it for the future.
Ragged<int32_t> ShortestPath(FsaVec &fsas,
                             const Array1<int32_t> &entering_arcs) {
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  const int32_t *entering_arcs_data = entering_arcs.Data();
  const Arc *arcs_data = fsas.values.Data();
  int32_t num_fsas = fsas.Dim0();
  int32_t num_states = fsas.TotSize(1);
  ContextPtr &context = fsas.Context();

  // allocate an extra element for ExclusiveSum
  Array1<int32_t> num_best_arcs_per_fsa(context, num_fsas + 1);
  int32_t *num_best_arcs_per_fsa_data = num_best_arcs_per_fsa.Data();
  const int32_t *row_splits1_data = fsas.RowSplits(1).Data();

  // -1 represents an invalid arc_index.
  // This extra array avoids an extra iteration over `entering_arcs`.
  Array1<int32_t> state_best_arc_index_array(context, num_states, -1);
  int32_t *state_best_arc_index_array_data = state_best_arc_index_array.Data();

  auto lambda_set_num_best_arcs = [=] __host__ __device__(int32_t fsas_idx0) {
    int32_t state_idx01 = row_splits1_data[fsas_idx0];
    int32_t state_idx01_next = row_splits1_data[fsas_idx0 + 1];

    if (state_idx01_next == state_idx01) {
      // this fsa is empty, so there is no best path available
      num_best_arcs_per_fsa_data[fsas_idx0] = 0;
      return;
    }

    int32_t final_state_idx01 = state_idx01_next - 1;
    int32_t cur_state = final_state_idx01;
    int32_t cur_index = entering_arcs_data[cur_state];
    int32_t num_arcs = 0;
    int32_t *p = state_best_arc_index_array_data + final_state_idx01;
    while (cur_index != -1) {
      *p = cur_index;
      --p;

      cur_state = arcs_data[cur_index].src_state + state_idx01;
      cur_index = entering_arcs_data[cur_state];
      ++num_arcs;
    }
    num_best_arcs_per_fsa_data[fsas_idx0] = num_arcs;
  };
  Eval(context, num_fsas, lambda_set_num_best_arcs);
  ExclusiveSum(num_best_arcs_per_fsa, &num_best_arcs_per_fsa);

  RaggedShape shape = RaggedShape2(&num_best_arcs_per_fsa, nullptr, -1);
  const int32_t *shape_row_splits1_data = shape.RowSplits(1).Data();
  const int32_t *shape_row_ids1_data = shape.RowIds(1).Data();

  const int32_t *ans_row_splits_data = shape.RowSplits(1).Data();
  Array1<int32_t> best_path_arc_indexes(context, shape.NumElements());
  int32_t *best_path_arc_indexes_data = best_path_arc_indexes.Data();

  auto lambda_set_best_arcs = [=] __host__ __device__(int32_t ans_idx01) {
    int32_t fsa_idx0 = shape_row_ids1_data[ans_idx01];
    int32_t ans_idx0x = shape_row_splits1_data[fsa_idx0];
    int32_t ans_idx1 = ans_idx01 - ans_idx0x;

    int32_t num_arcs_this_fsa = num_best_arcs_per_fsa_data[fsa_idx0 + 1] -
                                num_best_arcs_per_fsa_data[fsa_idx0];
    if (num_arcs_this_fsa == 0) return;

    int32_t final_state_idx01_this_fsa = row_splits1_data[fsa_idx0 + 1] - 1;

    const int32_t *p_start = state_best_arc_index_array_data +
                             final_state_idx01_this_fsa - num_arcs_this_fsa + 1;

    best_path_arc_indexes_data[ans_idx01] = p_start[ans_idx1];
  };
  Eval(context, shape.NumElements(), lambda_set_best_arcs);

  Ragged<int32_t> ans(shape, best_path_arc_indexes);
  return ans;
}


void AddEpsilonSelfLoops(FsaOrVec &src, FsaOrVec *dest,
                         Array1<int32_t> *arc_map) {
  ContextPtr c = src.Context();
  const int32_t *old_row_splits1_data = src.RowSplits(1).Data(),
      *old_row_ids1_data = src.RowIds(1).Data();
  const Arc *old_arcs_data = src.values.Data();
  if (src.NumAxes() == 2) {
    int32_t num_states = src.Dim0();
    if (num_states < 2) {
      K2_CHECK(num_states == 0);
      *dest = src;
      if (arc_map) *arc_map = Array1<int32_t>(c, 0);
      return;
    }

    int32_t old_num_arcs = src.TotSize(1),
        new_num_arcs = old_num_arcs + (num_states - 1);
    Array1<int32_t> new_row_splits(c, num_states + 1),
        new_row_ids(c, new_num_arcs);
    Array1<Arc> new_arcs(c, new_num_arcs);
    int32_t *new_row_splits1_data = new_row_splits.Data(),
        *new_row_ids1_data = new_row_ids.Data();
    Arc *new_arcs_data = new_arcs.Data();
    int32_t *arc_map_data = nullptr;
    if (arc_map) {
      *arc_map = Array1<int32_t>(c, new_num_arcs);
      arc_map_data = arc_map->Data();
    }
    ParallelRunner pr(c);
    {
      With w(pr.NewStream());
      auto lambda_copy_data = [=] __host__ __device__ (int32_t arc_idx01) -> void {
        int32_t state_idx0 = old_row_ids1_data[arc_idx01],
        new_arc_idx01 = arc_idx01 + 1 + state_idx0;
        // the "+1" above is because we put the self-loop first.
        new_row_ids1_data[new_arc_idx01] = state_idx0;
        new_arcs_data[new_arc_idx01] = old_arcs_data[arc_idx01];
        if (arc_map_data)
          arc_map_data[new_arc_idx01] = arc_idx01;
      };
      Eval(c, old_num_arcs, lambda_copy_data);
    }
    {
      With w(pr.NewStream());
      auto lambda_set_new_data = [=] __host__ __device__ (int32_t state_idx0) -> void {
        int32_t old_arc_idx0x = old_row_splits1_data[state_idx0],
        new_arc_idx0x = old_arc_idx0x + state_idx0;
        new_row_splits1_data[state_idx0] = new_arc_idx0x;
        if (state_idx0 + 1 < num_states) { // not final-state
          int32_t new_arc_idx01 = new_arc_idx0x;  // the 1st arc is the loop
          new_row_ids1_data[new_arc_idx01] = state_idx0;
          new_arcs_data[new_arc_idx01] = Arc(state_idx0, state_idx0, 0, 0.0);
          if (arc_map_data)
            arc_map_data[new_arc_idx01] = -1;
        } else {
          // Note: if num_states was zero we would have returned above, so we
          // dont have to worry about empty FSAs.
          new_row_splits1_data[num_states] = new_arc_idx0x;
        }
      };
      Eval(c, num_states, lambda_set_new_data);
    }
    pr.Finish();
    *dest = Ragged<Arc>(RaggedShape2(&new_row_splits, &new_row_ids, new_num_arcs),
                        new_arcs);
  } else {
    K2_CHECK(src.NumAxes() == 3);
    // Get a vector saying, for each FSA, whether it's nonempty.
    int32_t num_fsas = src.Dim0(),
        num_states = src.TotSize(1),
        old_num_arcs = src.TotSize(2);
    if (num_states == 0) {
      *dest = src;
      if (arc_map) *arc_map = Array1<int32_t>(c, 0);
      return;
    }
    Array1<int32_t> fsa_nonempty(c, num_fsas + 1);
    int32_t *fsa_nonempty_data = fsa_nonempty.Data();
    auto lambda_set_fsa_nonempty = [=] __host__ __device__ (int32_t fsa_idx0) -> void {
      fsa_nonempty_data[fsa_idx0] = (old_row_splits1_data[fsa_idx0+1] >
                                     old_row_splits1_data[fsa_idx0]);
    };
    Eval(c, num_fsas, lambda_set_fsa_nonempty);
    ExclusiveSum(fsa_nonempty, &fsa_nonempty);
    const int32_t *old_row_splits2_data = src.RowSplits(2).Data(),
        *old_row_ids2_data = src.RowIds(2).Data();
    int32_t num_nonempty_fsas = fsa_nonempty.Back(),
        new_num_arcs = old_num_arcs + num_states - num_nonempty_fsas;
    // we subtract `num_nonempty_fsas` because final-states don't get a self-loop.

    Array1<int32_t> new_row_splits2(c, num_states + 1),
        new_row_ids2(c, new_num_arcs);
    Array1<Arc> new_arcs(c, new_num_arcs);
    // fsa_idx0_mod_data maps from fsa_idx0 to a modified fsa_idx0 that
    // "doesn't count" FSAs with zero states.
    const int32_t *fsa_idx0_mod_data = fsa_nonempty_data;
    int32_t *new_row_splits2_data = new_row_splits2.Data(),
        *new_row_ids2_data = new_row_ids2.Data();
    Arc *new_arcs_data = new_arcs.Data();
    int32_t *arc_map_data = nullptr;
    if (arc_map) {
      *arc_map = Array1<int32_t>(c, new_num_arcs);
      arc_map_data = arc_map->Data();
    }
    ParallelRunner pr(c);
    {
      With w(pr.NewStream());
      auto lambda_copy_data = [=] __host__ __device__ (int32_t arc_idx012) -> void {
        int32_t state_idx01 = old_row_ids2_data[arc_idx012],
           fsa_idx0 = old_row_ids1_data[state_idx01],
           fsa_idx0_mod = fsa_idx0_mod_data[fsa_idx0],
          new_arc_idx012 = arc_idx012 + 1 + state_idx01 - fsa_idx0_mod;
        // The "+1" above is because we put the self-loop first.  The
        // "-fsa_idx0_mod" is because final-states don't get a self-loop.
        new_row_ids2_data[new_arc_idx012] = state_idx01;
        new_arcs_data[new_arc_idx012] = old_arcs_data[arc_idx012];
        if (arc_map_data)
          arc_map_data[new_arc_idx012] = arc_idx012;
      };
      Eval(c, old_num_arcs, lambda_copy_data);
    }
    {
      With w(pr.NewStream());
      auto lambda_set_new_data = [=] __host__ __device__ (int32_t state_idx01) -> void {
        int32_t
          fsa_idx0 = old_row_ids1_data[state_idx01],
          fsa_idx0_mod = fsa_idx0_mod_data[fsa_idx0],
          state_idx0x = old_row_splits1_data[fsa_idx0],
          next_state_idx0x = old_row_splits1_data[fsa_idx0 + 1],
          old_arc_idx01x = old_row_splits2_data[state_idx01];
        // Below the "+ state_idx01" is because each state gets a self-loop, and
        // the "- fsa_idx0_mod" is becaue final-states don't get a self-loop.
        int32_t new_arc_idx01x = old_arc_idx01x + state_idx01 - fsa_idx0_mod;
        // The self-loop arc is the first arc:
        int32_t new_arc_idx012 = new_arc_idx01x;
        new_row_splits2_data[state_idx01] = new_arc_idx01x;
        if (state_idx01 + 1 < next_state_idx0x) {  // not final-state
          new_row_ids2_data[new_arc_idx012] = state_idx01;
          int32_t state_idx1 = state_idx01 - state_idx0x;
          new_arcs_data[new_arc_idx012] = Arc(state_idx1, state_idx1, 0, 0.0);
          if (arc_map_data)
            arc_map_data[new_arc_idx012] = -1;
        } else if (state_idx01 + 1 == num_states) {
          // Note: if num_states was zero  we would have returned above, so we
          // dont have to worry about an empty FsaVec.
          new_row_splits2_data[num_states] = new_arc_idx01x;
        }
      };
      Eval(c, num_states, lambda_set_new_data);
    }
    pr.Finish();
    *dest = Ragged<Arc>(RaggedShape3(&src.RowSplits(1), &src.RowIds(1),
                                     num_states,
                                     &new_row_splits2, &new_row_ids2,
                                     new_num_arcs),
                        new_arcs);
  }
}



}  // namespace k2
