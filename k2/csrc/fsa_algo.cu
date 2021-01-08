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
#include <type_traits>
#include <utility>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/host/aux_labels.h"
#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/determinize.h"
#include "k2/csrc/host/intersect.h"
#include "k2/csrc/host/rmepsilon.h"
#include "k2/csrc/host/topsort.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/macros.h"

// this contains a subset of the algorithms in fsa_algo.h; currently it just
// contains one that are wrappings of the corresponding algorithms in
// host/.
namespace k2 {

bool RecursionWrapper(bool (*f)(Fsa &, Fsa *, Array1<int32_t> *), Fsa &src,
                      Fsa *dest, Array1<int32_t> *arc_map) {
  NVTX_RANGE(K2_FUNC);
  // src is actually an FsaVec.  Just recurse for now.
  int32_t num_fsas = src.shape.Dim0();
  std::vector<Fsa> srcs(num_fsas), dests(num_fsas);
  std::vector<Array1<int32_t>> arc_maps(num_fsas);
  int32_t tot_num_arcs = 0;
  for (int32_t i = 0; i < num_fsas; ++i) {
    srcs[i] = src.Index(0, i);
    // Recurse.
    if (!f(srcs[i], &(dests[i]),
           (arc_map != nullptr ? &(arc_maps[i]) : nullptr)))
      return false;
    if (arc_map != nullptr) {
      // convert arc indexes in arc_maps from idx2 to idx012
      arc_maps[i] = Plus(arc_maps[i], tot_num_arcs);
      tot_num_arcs += srcs[i].NumElements();
    }
  }
  *dest = Stack(0, num_fsas, dests.data());
  if (arc_map != nullptr) *arc_map = Append(num_fsas, arc_maps.data());
  return true;
}

bool Connect(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
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

bool TopSortHost(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapper(TopSortHost, src, dest, arc_map);
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

bool Intersect(FsaOrVec &a_fsas, int32_t properties_a, FsaOrVec &b_fsas,
               int32_t properties_b, bool treat_epsilons_specially, FsaVec *out,
               Array1<int32_t> *arc_map_a, Array1<int32_t> *arc_map_b) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(a_fsas.NumAxes() >= 2 && a_fsas.NumAxes() <= 3);
  K2_CHECK(b_fsas.NumAxes() >= 2 && b_fsas.NumAxes() <= 3);
  ContextPtr c = a_fsas.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  if (a_fsas.NumAxes() == 2) {
    FsaVec a_fsas_vec = FsaToFsaVec(a_fsas);
    return Intersect(a_fsas_vec, properties_a, b_fsas, properties_b,
                     treat_epsilons_specially, out, arc_map_a, arc_map_b);
  }
  if (b_fsas.NumAxes() == 2) {
    FsaVec b_fsas_vec = FsaToFsaVec(b_fsas);
    return Intersect(a_fsas, properties_a, b_fsas_vec, properties_b,
                     treat_epsilons_specially, out, arc_map_a, arc_map_b);
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
  if (properties_a < 0) {
    Array1<int32_t> properties_a_out(c, num_fsas_a);
    GetFsaVecBasicProperties(a_fsas, &properties_a_out, &properties_a);
  }
  if (properties_b < 0) {
    Array1<int32_t> properties_b_out(c, num_fsas_b);
    GetFsaVecBasicProperties(b_fsas, &properties_b_out, &properties_b);
  }
  bool arc_sorted = (properties_a & kFsaPropertiesArcSorted) &&
                    (properties_b & kFsaPropertiesArcSorted);
  K2_CHECK(arc_sorted) << "Both a_fsas and b_fsas should be arc-sorted";
  int32_t num_fsas = std::max(num_fsas_a, num_fsas_b);

  std::vector<std::unique_ptr<k2host::Intersection>> intersections(num_fsas);
  std::vector<k2host::Array2Size<int32_t>> sizes(num_fsas);
  for (int32_t i = 0; i < num_fsas; ++i) {
    k2host::Fsa host_fsa_a = FsaVecToHostFsa(a_fsas, i * stride_a),
                host_fsa_b = FsaVecToHostFsa(b_fsas, i * stride_b);
    intersections[i] = std::make_unique<k2host::Intersection>(
        host_fsa_a, host_fsa_b, treat_epsilons_specially, false);
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

// Will be used in RemoveEpsilon and Determinize below to process FsaVec input
// recursively.
void RecursionWrapper(void (*f)(FsaOrVec &, FsaOrVec *, Ragged<int32_t> *),
                      FsaOrVec &src, FsaOrVec *dest,
                      Ragged<int32_t> *arc_deriv) {
  NVTX_RANGE(K2_FUNC);
  // src is actually an FsaVec.  Just recurse for now.
  K2_CHECK_EQ(src.NumAxes(), 3);
  int32_t num_fsas = src.shape.Dim0();
  std::vector<Fsa> srcs(num_fsas), dests(num_fsas);
  std::vector<Ragged<int32_t>> arc_derivs(num_fsas);
  int32_t tot_num_arcs = 0;
  for (int32_t i = 0; i < num_fsas; ++i) {
    srcs[i] = src.Index(0, i);
    f(srcs[i], &(dests[i]), arc_deriv != nullptr ? &(arc_derivs[i]) : nullptr);
    if (arc_deriv != nullptr) {
      // convert arc indexes in arc_derivs from idx2 to idx012
      Array1<int32_t> &values = arc_derivs[i].values;
      values = Plus(values, tot_num_arcs);
      tot_num_arcs += srcs[i].NumElements();
    }
  }
  *dest = Stack(0, num_fsas, dests.data());
  if (arc_deriv != nullptr) *arc_deriv = Append(0, num_fsas, arc_derivs.data());
}

void RemoveEpsilon(FsaOrVec &src, FsaOrVec *dest,
                   Ragged<int32_t> *arc_derivs /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapper(RemoveEpsilon, src, dest, arc_derivs);
  }

  k2host::Fsa host_fsa = FsaToHostFsa(src);
  int32_t num_states = host_fsa.NumStates();
  K2_CHECK_EQ(num_states, src.Dim0());
  std::vector<double> max_forward_weights(num_states);
  std::vector<double> max_backward_weights(num_states);
  k2host::WfsaWithFbWeights max_wfsa(host_fsa, k2host::kMaxWeight,
                                     max_forward_weights.data(),
                                     max_backward_weights.data());
  // pass infinity as beam since we don't do pruning here.
  float beam = std::numeric_limits<float>::infinity();
  k2host::EpsilonsRemoverPrunedMax eps_remover(max_wfsa, beam);
  k2host::Array2Size<int32_t> fsa_size, arc_derivs_size;
  eps_remover.GetSizes(&fsa_size, &arc_derivs_size);
  FsaCreator fsa_creator(fsa_size);
  k2host::Fsa host_dest_fsa = fsa_creator.GetHostFsa();
  K2_STATIC_ASSERT(
      (std::is_same<k2host::MaxTracebackState::DerivType, int32_t>::value));
  Ragged2Creator<int32_t> ragged_creator(arc_derivs_size);
  k2host::Array2<int32_t *, int32_t> host_arc_derivs =
      ragged_creator.GetHostArray2();
  eps_remover.GetOutput(&host_dest_fsa, &host_arc_derivs);
  *dest = fsa_creator.GetFsa();
  if (arc_derivs != nullptr) *arc_derivs = ragged_creator.GetRagged2();
}

void Determinize(FsaOrVec &src, FsaOrVec *dest,
                 Ragged<int32_t> *arc_derivs /*=nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapper(Determinize, src, dest, arc_derivs);
  }
  k2host::Fsa host_fsa = FsaToHostFsa(src);
  int32_t num_states = host_fsa.NumStates();
  K2_CHECK_EQ(num_states, src.Dim0());
  int32_t max_step = -1;  // no limit
  k2host::DeterminizerMax determinizer(host_fsa, max_step);
  k2host::Array2Size<int32_t> fsa_size, arc_derivs_size;
  determinizer.GetSizes(&fsa_size, &arc_derivs_size);
  FsaCreator fsa_creator(fsa_size);
  k2host::Fsa host_dest_fsa = fsa_creator.GetHostFsa();
  K2_STATIC_ASSERT(
      (std::is_same<k2host::MaxTracebackState::DerivType, int32_t>::value));
  Ragged2Creator<int32_t> ragged_creator(arc_derivs_size);
  k2host::Array2<int32_t *, int32_t> host_arc_derivs =
      ragged_creator.GetHostArray2();
  determinizer.GetOutput(&host_dest_fsa, &host_arc_derivs);
  *dest = fsa_creator.GetFsa();
  if (arc_derivs != nullptr) *arc_derivs = ragged_creator.GetRagged2();
}

Fsa LinearFsa(const Array1<int32_t> &symbols) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = symbols.Context();
  int32_t n = symbols.Dim(), num_states = n + 2, num_arcs = n + 1;
  Array1<int32_t> row_splits1 = Range(c, num_states + 1, 0),
                  row_ids1 = Range(c, num_arcs, 0);
  int32_t *row_splits1_data = row_splits1.Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();
  const int32_t *symbols_data = symbols.Data();
  K2_EVAL(
      c, num_arcs, lambda_set_arcs, (int32_t arc_idx01)->void {
        int32_t src_state = arc_idx01, dest_state = arc_idx01 + 1,
                // -1 == kFinalSymbol
            symbol = (arc_idx01 < n ? symbols_data[arc_idx01] : -1);
        if (arc_idx01 < n) K2_CHECK_NE(symbol, -1);
        float score = 0.0;
        arcs_data[arc_idx01] = Arc(src_state, dest_state, symbol, score);
        // the final state has no leaving arcs.
        if (arc_idx01 == 0) row_splits1_data[num_states] = num_arcs;
      });
  return Ragged<Arc>(RaggedShape2(&row_splits1, &row_ids1, num_arcs), arcs);
}

FsaVec LinearFsas(Ragged<int32_t> &symbols) {
  NVTX_RANGE(K2_FUNC);
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
  K2_EVAL(
      c, num_states, lambda, (int32_t state_idx01)->void {
        int32_t fsa_idx0 = row_ids1_data[state_idx01],
                state_idx0x = row_splits1_data[fsa_idx0],
                next_state_idx0x = row_splits1_data[fsa_idx0 + 1],
                idx1 = state_idx01 - state_idx0x;

        // the following works because each FSA has one fewer arcs than states.
        int32_t arc_idx0xx = state_idx0x - fsa_idx0,
                next_arc_idx0xx = next_state_idx0x - (fsa_idx0 + 1),
                // the following may look a bit wrong.. here, the idx1 is the
                // same as the idx12 if the arc exists, because each state has
                // one arc leaving it (except the last state).
            arc_idx012 = arc_idx0xx + idx1;
        // the following works because each FSA has one fewer symbols than arcs
        // (however it doesn't work for the last arc of each FSA; we check
        // below.)
        int32_t symbol_idx01 = arc_idx012 - fsa_idx0;
        if (arc_idx012 < next_arc_idx0xx) {
          int32_t src_state = idx1, dest_state = idx1 + 1,
                  symbol = (arc_idx012 + 1 < next_arc_idx0xx
                                ? symbols_data[symbol_idx01]
                                : -1);  // kFinalSymbol
          float score = 0.0;
          arcs_data[arc_idx012] = Arc(src_state, dest_state, symbol, score);
          row_ids2_data[arc_idx012] = state_idx01;
        } else {
          // The following ensures that the last element of row_splits1_data
          // (i.e. row_splits1[num_states]) is set to num_arcs.  It also writes
          // something unnecessary for the last state of each FSA but the last
          // one, which will cause 2 threads to write the same item to the same
          // location. Note that there is no arc with index `arc_idx01`, if you
          // reach here.
          row_splits2_data[state_idx01 + 1] = arc_idx012;
        }
        row_splits2_data[state_idx01] = arc_idx012;
      });

  return Ragged<Arc>(
      RaggedShape3(&states_shape.RowSplits(1), &states_shape.RowIds(1),
                   num_states, &row_splits2, &row_ids2, num_arcs),
      arcs);
}

void ArcSort(Fsa *fsa) {
  NVTX_RANGE(K2_FUNC);
  if (fsa->NumAxes() < 2) return;  // it is empty
  SortSublists<Arc>(fsa);
}

void ArcSort(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  if (!src.values.IsValid()) return;

  if (arc_map != nullptr)
    *arc_map = Array1<int32_t>(src.Context(), src.NumElements());

  Fsa tmp(src.shape, src.values.Clone());
  SortSublists<Arc>(&tmp, arc_map);
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
  NVTX_RANGE(K2_FUNC);
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

  K2_EVAL(
      context, num_fsas, lambda_set_num_best_arcs, (int32_t fsas_idx0) {
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
      });
  ExclusiveSum(num_best_arcs_per_fsa, &num_best_arcs_per_fsa);

  RaggedShape shape = RaggedShape2(&num_best_arcs_per_fsa, nullptr, -1);
  const int32_t *shape_row_splits1_data = shape.RowSplits(1).Data();
  const int32_t *shape_row_ids1_data = shape.RowIds(1).Data();

  const int32_t *ans_row_splits_data = shape.RowSplits(1).Data();
  Array1<int32_t> best_path_arc_indexes(context, shape.NumElements());
  int32_t *best_path_arc_indexes_data = best_path_arc_indexes.Data();

  K2_EVAL(
      context, shape.NumElements(), lambda_set_best_arcs, (int32_t ans_idx01) {
        int32_t fsa_idx0 = shape_row_ids1_data[ans_idx01];
        int32_t ans_idx0x = shape_row_splits1_data[fsa_idx0];
        int32_t ans_idx1 = ans_idx01 - ans_idx0x;

        int32_t num_arcs_this_fsa = num_best_arcs_per_fsa_data[fsa_idx0 + 1] -
                                    num_best_arcs_per_fsa_data[fsa_idx0];
        if (num_arcs_this_fsa == 0) return;

        int32_t final_state_idx01_this_fsa = row_splits1_data[fsa_idx0 + 1] - 1;

        const int32_t *p_start = state_best_arc_index_array_data +
                                 final_state_idx01_this_fsa -
                                 num_arcs_this_fsa + 1;

        best_path_arc_indexes_data[ans_idx01] = p_start[ans_idx1];
      });

  Ragged<int32_t> ans(shape, best_path_arc_indexes);
  return ans;
}

void AddEpsilonSelfLoops(FsaOrVec &src, FsaOrVec *dest,
                         Array1<int32_t> *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  const int32_t *old_row_splits1_data = src.RowSplits(1).Data(),
                *old_row_ids1_data = src.RowIds(1).Data();
  const Arc *old_arcs_data = src.values.Data();
  if (src.NumAxes() == 2) {
    int32_t num_states = src.Dim0();
    if (num_states < 2) {
      K2_CHECK_EQ(num_states, 0);
      *dest = src;
      if (arc_map != nullptr) *arc_map = Array1<int32_t>(c, 0);
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
      K2_EVAL(
          c, old_num_arcs, lambda_copy_data, (int32_t arc_idx01)->void {
            int32_t state_idx0 = old_row_ids1_data[arc_idx01],
                    new_arc_idx01 = arc_idx01 + 1 + state_idx0;
            // the "+1" above is because we put the self-loop first.
            new_row_ids1_data[new_arc_idx01] = state_idx0;
            new_arcs_data[new_arc_idx01] = old_arcs_data[arc_idx01];
            if (arc_map_data) arc_map_data[new_arc_idx01] = arc_idx01;
          });
    }
    {
      With w(pr.NewStream());
      K2_EVAL(
          c, num_states, lambda_set_new_data, (int32_t state_idx0)->void {
            int32_t old_arc_idx0x = old_row_splits1_data[state_idx0],
                    new_arc_idx0x = old_arc_idx0x + state_idx0;
            new_row_splits1_data[state_idx0] = new_arc_idx0x;
            if (state_idx0 + 1 < num_states) {        // not final-state
              int32_t new_arc_idx01 = new_arc_idx0x;  // the 1st arc is the loop
              new_row_ids1_data[new_arc_idx01] = state_idx0;
              new_arcs_data[new_arc_idx01] =
                  Arc(state_idx0, state_idx0, 0, 0.0);
              if (arc_map_data) arc_map_data[new_arc_idx01] = -1;
            } else {
              // Note: if num_states was zero we would have returned above, so
              // we don't have to worry about empty FSAs.
              new_row_splits1_data[num_states] = new_arc_idx0x;
            }
          });
    }
    pr.Finish();
    *dest = Ragged<Arc>(
        RaggedShape2(&new_row_splits, &new_row_ids, new_num_arcs), new_arcs);
  } else {
    K2_CHECK_EQ(src.NumAxes(), 3);
    // Get a vector saying, for each FSA, whether it's nonempty.
    int32_t num_fsas = src.Dim0(), num_states = src.TotSize(1),
            old_num_arcs = src.TotSize(2);
    if (num_states == 0) {
      *dest = src;
      if (arc_map) *arc_map = Array1<int32_t>(c, 0);
      return;
    }
    Array1<int32_t> fsa_nonempty(c, num_fsas + 1);
    int32_t *fsa_nonempty_data = fsa_nonempty.Data();
    K2_EVAL(
        c, num_fsas, lambda_set_fsa_nonempty, (int32_t fsa_idx0)->void {
          fsa_nonempty_data[fsa_idx0] = (old_row_splits1_data[fsa_idx0 + 1] >
                                         old_row_splits1_data[fsa_idx0]);
        });
    ExclusiveSum(fsa_nonempty, &fsa_nonempty);
    const int32_t *old_row_splits2_data = src.RowSplits(2).Data(),
                  *old_row_ids2_data = src.RowIds(2).Data();
    int32_t num_nonempty_fsas = fsa_nonempty.Back(),
            new_num_arcs = old_num_arcs + num_states - num_nonempty_fsas;
    // we subtract `num_nonempty_fsas` because final-states don't get a
    // self-loop.

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
      K2_EVAL(
          c, old_num_arcs, lambda_copy_data, (int32_t arc_idx012)->void {
            int32_t state_idx01 = old_row_ids2_data[arc_idx012],
                    fsa_idx0 = old_row_ids1_data[state_idx01],
                    fsa_idx0_mod = fsa_idx0_mod_data[fsa_idx0],
                    new_arc_idx012 =
                        arc_idx012 + 1 + state_idx01 - fsa_idx0_mod;
            // The "+1" above is because we put the self-loop first.  The
            // "-fsa_idx0_mod" is because final-states don't get a self-loop.
            new_row_ids2_data[new_arc_idx012] = state_idx01;
            new_arcs_data[new_arc_idx012] = old_arcs_data[arc_idx012];
            if (arc_map_data) arc_map_data[new_arc_idx012] = arc_idx012;
          });
    }
    {
      With w(pr.NewStream());
      K2_EVAL(
          c, num_states, lambda_set_new_data, (int32_t state_idx01)->void {
            int32_t fsa_idx0 = old_row_ids1_data[state_idx01],
                    fsa_idx0_mod = fsa_idx0_mod_data[fsa_idx0],
                    state_idx0x = old_row_splits1_data[fsa_idx0],
                    next_state_idx0x = old_row_splits1_data[fsa_idx0 + 1],
                    old_arc_idx01x = old_row_splits2_data[state_idx01];
            // Below the "+ state_idx01" is because each state gets a self-loop,
            // and the "- fsa_idx0_mod" is because final-states don't get a
            // self-loop.
            int32_t new_arc_idx01x =
                old_arc_idx01x + state_idx01 - fsa_idx0_mod;
            // The self-loop arc is the first arc:
            int32_t new_arc_idx012 = new_arc_idx01x;
            new_row_splits2_data[state_idx01] = new_arc_idx01x;
            if (state_idx01 + 1 < next_state_idx0x) {  // not final-state
              new_row_ids2_data[new_arc_idx012] = state_idx01;
              int32_t state_idx1 = state_idx01 - state_idx0x;
              new_arcs_data[new_arc_idx012] =
                  Arc(state_idx1, state_idx1, 0, 0.0);
              if (arc_map_data) arc_map_data[new_arc_idx012] = -1;
            } else if (state_idx01 + 1 == num_states) {
              // Note: if num_states was zero  we would have returned above, so
              // we dont have to worry about an empty FsaVec.
              new_row_splits2_data[num_states] = new_arc_idx01x;
            }
          });
    }
    pr.Finish();
    *dest =
        Ragged<Arc>(RaggedShape3(&src.RowSplits(1), &src.RowIds(1), num_states,
                                 &new_row_splits2, &new_row_ids2, new_num_arcs),
                    new_arcs);
  }
}

Fsa Union(FsaVec &fsas, Array1<int32_t> *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsas.NumAxes(), 3);

  ContextPtr &context = fsas.Context();
  const int32_t *fsas_row_splits1_data = fsas.RowSplits(1).Data();
  const int32_t *fsas_row_splits2_data = fsas.RowSplits(2).Data();
  const int32_t *fsas_row_ids1_data = fsas.RowIds(1).Data();
  const int32_t *fsas_row_ids2_data = fsas.RowIds(2).Data();
  const Arc *arcs_data = fsas.values.Data();

  int32_t num_fsas = fsas.Dim0();
  int32_t num_states = fsas.TotSize(1);
  int32_t num_arcs = fsas.TotSize(2);

  // A new start state and a new final state are added (+2).
  // The final state of each fsa is removed (-num_fsas)
  int32_t num_out_states = num_states + 2 - num_fsas;
  int32_t out_final_state = num_out_states - 1;

  // For every fsa, a new arc is added from the new start state
  // to its original start state (+num_fsas)
  int32_t num_out_arcs = num_arcs + num_fsas;

  Array1<int32_t> out_row_ids(context, num_out_arcs);
  Array1<Arc> out_arcs(context, num_out_arcs);
  Array1<int32_t> tmp_arc_map(context, num_out_arcs, -1);
  int32_t *tmp_arc_map_data = tmp_arc_map.Data();

  int32_t *out_row_ids_data = out_row_ids.Data();
  Arc *out_arcs_data = out_arcs.Data();

  K2_EVAL(
      context, num_arcs, lambda_set_out, (int32_t fsas_arc_idx012) {
        int32_t fsas_state_idx01 = fsas_row_ids2_data[fsas_arc_idx012];
        int32_t fsas_idx0 = fsas_row_ids1_data[fsas_state_idx01];
        int32_t this_fsa_final_state_idx01 =
            fsas_row_splits1_data[fsas_idx0 + 1] - 1;

        K2_DCHECK_GT(this_fsa_final_state_idx01, fsas_state_idx01)
            << "We support only FSAs with at least two states at present";

        int32_t fsas_state_idx0x = fsas_row_splits1_data[fsas_idx0];
        int32_t fsas_state_idx1 = fsas_state_idx01 - fsas_state_idx0x;
        int32_t this_fsa_final_state_idx1 =
            this_fsa_final_state_idx01 - fsas_state_idx0x;

        int32_t fsas_arc_idx0xx = fsas_row_splits2_data[fsas_state_idx0x];

        // fsa0: +1 (a new start state)
        // fsa1: +0 (the final state of fsa0 is removed)
        // fsa2: -1 (the final state of fsa1 is removed)
        // fsa3: -2 (the final state of fsa2 is removed)
        int32_t state_offset = 1 - fsas_idx0;
        int32_t out_state_idx0 = fsas_state_idx01 + state_offset;

        int32_t out_arc_idx01 = fsas_arc_idx012 + num_fsas;
        out_row_ids_data[out_arc_idx01] = out_state_idx0;
        Arc arc = arcs_data[fsas_arc_idx012];

        K2_DCHECK_EQ(arc.src_state, fsas_state_idx1);

        if (arc.dest_state == this_fsa_final_state_idx1)
          arc.dest_state = out_final_state;
        else
          arc.dest_state = arc.dest_state - arc.src_state + out_state_idx0;

        arc.src_state = out_state_idx0;
        out_arcs_data[out_arc_idx01] = arc;
        tmp_arc_map_data[out_arc_idx01] = fsas_arc_idx012;

        if (fsas_arc_idx0xx == fsas_arc_idx012) {
          // add a new arc from the new start state to the start state
          // of this fsa
          //
          // WARNING: we cannot use fsas_state_idx01 here
          // since the start state may have no leaving arcs!
          Arc arc(0, fsas_state_idx0x + state_offset, 0, 0);
          out_arcs_data[fsas_idx0] = arc;
          out_row_ids_data[fsas_idx0] = 0;
        }
      });

  if (arc_map != nullptr) *arc_map = std::move(tmp_arc_map);
  Array1<int32_t> out_row_splits(context, num_out_states + 1);
  RowIdsToRowSplits(out_row_ids, &out_row_splits);
  RaggedShape shape = RaggedShape2(&out_row_splits, &out_row_ids, num_out_arcs);
  Fsa ans = Ragged<Arc>(shape, out_arcs);
  return ans;
}

Fsa Closure(Fsa &fsa, Array1<int32_t> *arc_map /* = nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsa.NumAxes(), 2) << "We support only a single FSA.";
  ContextPtr &c = fsa.Context();

  int32_t num_states = fsa.Dim0();
  if (num_states < 2) {
    K2_CHECK_EQ(num_states, 0)
        << "An empty fsa should contain no states at all";
    if (arc_map != nullptr) *arc_map = Array1<int32_t>(c, 0);
    return fsa;  // return itself if the input fsa is empty
  }

  const int32_t *fsa_row_splits_data = fsa.RowSplits(1).Data();
  const int32_t *fsa_row_ids_data = fsa.RowIds(1).Data();
  const Arc *fsa_arcs_data = fsa.values.Data();
  int32_t fsa_final_state = num_states - 1;

  int32_t num_out_states = num_states;

  // An arc from the start state to the final state with label == 0 is added.
  int32_t num_out_arcs = fsa.values.Dim() + 1;

  Array1<int32_t> out_row_ids(c, num_out_arcs);
  int32_t *out_row_ids_data = out_row_ids.Data();

  Array1<Arc> out_arcs(c, num_out_arcs);
  Arc *out_arcs_data = out_arcs.Data();

  Array1<int32_t> tmp_arc_map(c, num_out_arcs);
  int32_t *tmp_arc_map_data = tmp_arc_map.Data();

  K2_EVAL(
      c, fsa.values.Dim(), lambda_set_arcs, (int32_t fsa_arc_idx01) {
        int32_t fsa_state_idx0 = fsa_row_ids_data[fsa_arc_idx01];
        int32_t fsa_arc_idx0x = fsa_row_splits_data[fsa_state_idx0];
        int32_t fsa_arc_idx1 = fsa_arc_idx01 - fsa_arc_idx0x;
        int32_t this_state_num_arcs =
            fsa_row_splits_data[fsa_state_idx0 + 1] - fsa_arc_idx0x;

        Arc arc = fsa_arcs_data[fsa_arc_idx01];
        if (arc.dest_state == fsa_final_state) {
          // modify arcs entering the final state such that:
          //   - dest_state == 0
          //   - label == 0
          arc.dest_state = 0;
          K2_DCHECK_EQ(arc.label, -1);
          arc.label = 0;
        }

        int out_arc_idx01;
        if (arc.src_state > 0) {
          // this arc is not originated from the start state, so its index is
          // incremented
          out_arc_idx01 = fsa_arc_idx01 + 1;
        } else {
          out_arc_idx01 = fsa_arc_idx01;
          if (fsa_arc_idx1 == this_state_num_arcs - 1) {
            // This is the last arc of the original start state,
            // so we add a new arc just after it.
            Arc new_arc(0, fsa_final_state, -1, 0.0f);
            out_arcs_data[out_arc_idx01 + 1] = new_arc;
            out_row_ids_data[out_arc_idx01 + 1] = 0;
            tmp_arc_map_data[out_arc_idx01 + 1] = -1;
          }
        }

        // it may happen that the start state has no leaving arcs
        if (fsa_row_splits_data[1] == 0) {
          Arc new_arc(0, fsa_final_state, -1, 0.0f);
          out_arcs_data[0] = new_arc;
          out_row_ids_data[0] = 0;
          tmp_arc_map_data[0] = -1;
        }

        tmp_arc_map_data[out_arc_idx01] = fsa_arc_idx01;

        out_arcs_data[out_arc_idx01] = arc;
        out_row_ids_data[out_arc_idx01] = arc.src_state;
      });

  if (arc_map != nullptr) *arc_map = std::move(tmp_arc_map);

  Array1<int32_t> out_row_splits(c, num_out_states + 1);
  int32_t *out_row_splits_data = out_row_splits.Data();

  K2_EVAL(
      c, out_row_splits.Dim(), lambda_set_row_splits, (int32_t i) {
        if (i == 0)
          out_row_splits_data[i] = 0;
        else
          out_row_splits_data[i] = fsa_row_splits_data[i] + 1;
      });

  RaggedShape shape = RaggedShape2(&out_row_splits, &out_row_ids, num_out_arcs);
  Fsa ans = Ragged<Arc>(shape, out_arcs);
  return ans;
}

FsaOrVec ExpandArcs(FsaOrVec &fsas, RaggedShape &labels_shape,
                    Array1<int32_t> *fsas_arc_map /*=nullptr*/,
                    Array1<int32_t> *labels_arc_map /*=nullptr*/) {
  if (fsas.NumAxes() == 2) {
    FsaVec fsas_temp = FsaToFsaVec(fsas);
    return ExpandArcs(fsas_temp, labels_shape, fsas_arc_map, labels_arc_map)
        .RemoveAxis(0);
  }
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(labels_shape.NumAxes(), 2);
  K2_CHECK_EQ(fsas.NumElements(), labels_shape.Dim0());
  ContextPtr c = fsas.Context();
  K2_CHECK(c->IsCompatible(*labels_shape.Context()));

  RaggedShape state_to_arcs = GetLayer(fsas.shape, 1);

  // `state_to_foo` is a RaggedShape that, for each state in `fsas`, has a list
  // of length `tot_arcs + 1`.  Interpret this as: one element for the state
  // itself, then one for each arc leaving it.  This `foo` is an index that
  // corresponds to num-arcs plus one, but because it is really a placeholder
  // and we want to keep it distinct from other things, we call it `foo`.
  RaggedShape state_to_foo = ChangeSublistSize(state_to_arcs, 1);

  int32_t foo_size = state_to_foo.NumElements();

  // For each element of `state_to_foo`, `num_ostates_for` says how many states
  // there will be for this (state,foo) in the returned (output) FSA.  Here, the
  // idx0 is the state, the idx1 is foo.  If idx1 == 0 (interpret this as "the
  // state itself"), then `num_ostates_for[idx01] = 1`, meaning "keep the
  // original state".  Otherwise, idx1 - 1 represents an arc_idx2 [into `fsas`],
  // and we set `num_ostates_for[idx01] = max(0, seq_len-1)`, where seq_len is
  // the length of the sequence in `labels_shape` corresponding to this
  // arc-index.
  Array1<int32_t> num_ostates_for(c, foo_size + 1);
  int32_t *num_ostates_for_data = num_ostates_for.Data();

  const int32_t *labels_row_splits1_data = labels_shape.RowSplits(1).Data(),
                *fsas_row_splits2_data = fsas.RowSplits(2).Data(),
                *state_to_foo_row_splits1_data =
                    state_to_foo.RowSplits(1).Data(),
                *state_to_foo_row_ids1_data = state_to_foo.RowIds(1).Data();

  K2_EVAL(
      c, foo_size, lambda_set_num_ostates, (int32_t idx01)->void {
        // note: the idx01, idx0, idx0x are into `state_to_foo`.
        // This idx0 is a state-index into `fsas` (an idx01 w.r.t. `fsas`).
        int32_t idx0 = state_to_foo_row_ids1_data[idx01],
                idx0x = state_to_foo_row_splits1_data[idx0],
                idx1 = idx01 - idx0x;  // idx1 is `foo`.
        int32_t num_ostates;
        if (idx1 == 0) {
          num_ostates = 1;  // this is a copy of the original state.
        } else {
          int32_t fsas_arc_idx2 = idx1 - 1, fsas_state_idx01 = idx0,
                  fsas_arc_idx01x = fsas_row_splits2_data[fsas_state_idx01],
                  fsas_arc_idx012 = fsas_arc_idx01x + fsas_arc_idx2,
                  labels_shape_idx0 = fsas_arc_idx012,
                  labels_shape_idx0x =
                      labels_row_splits1_data[labels_shape_idx0],
                  labels_shape_idx0x_next =
                      labels_row_splits1_data[labels_shape_idx0 + 1],
                  labels_shape_len1 =
                      labels_shape_idx0x_next - labels_shape_idx0x;
          // A sequence of n symbols will require n-1 extra states to represent
          // it.
          num_ostates = max(labels_shape_len1 - 1, (int32_t)0);
        }
        num_ostates_for_data[idx01] = num_ostates;
      });
  ExclusiveSum(num_ostates_for, &num_ostates_for);
  Array1<int32_t> &foo_to_ostates_row_splits = num_ostates_for;
  RaggedShape foo_to_ostates =
      RaggedShape2(&foo_to_ostates_row_splits, nullptr, -1);

  // to_ostates_shape has 4 axes: [fsa_id][orig_state][foo][ostate]
  // where foo is a general-purpose index that ranges over the (num_arcs + 1) of
  // the original state.
  RaggedShape to_ostates_shape = ComposeRaggedShapes3(GetLayer(fsas.shape, 0),
                                                      state_to_foo,
                                                      foo_to_ostates);

  // Below, `tos` means `to_ostates_shape`.
  const int32_t *tos_row_splits1_data = to_ostates_shape.RowSplits(1).Data(),
                *tos_row_ids1_data = to_ostates_shape.RowIds(1).Data(),
                *tos_row_splits2_data = to_ostates_shape.RowSplits(2).Data(),
                *tos_row_ids2_data = to_ostates_shape.RowIds(2).Data(),
                *tos_row_splits3_data = to_ostates_shape.RowSplits(3).Data(),
                *tos_row_ids3_data = to_ostates_shape.RowIds(3).Data();

  // `num_oarcs` gives the number of arcs in the returned (output) FSA for each
  // `ostate` (i.e. leaving each state in the returned FSA).
  int32_t tot_ostates = to_ostates_shape.NumElements();
  Array1<int32_t> num_oarcs(c, tot_ostates + 1);
  int32_t *num_oarcs_data = num_oarcs.Data();
  K2_EVAL(
      c, tot_ostates, lambda_set_num_oarcs, (int32_t idx0123)->void {
        // All these indexes are into `to_ostates_shape`, indexed
        // `[fsa][state][foo][ostate].`
        int32_t idx012 = tos_row_ids3_data[idx0123],
                idx012x = tos_row_splits3_data[idx012],
                idx01 = tos_row_ids2_data[idx012],
                idx01x = tos_row_splits2_data[idx01],
                idx01x_next = tos_row_splits2_data[idx01 + 1],
                len2 = idx01x_next - idx01x, idx2 = idx012 - idx01x,
                idx3 = idx0123 - idx012x;
        int32_t num_arcs;
        if (idx2 == 0) {
          K2_CHECK_EQ(idx3, 0);
          // This ostate corresponds to the original state; it is not one of the
          // extra states added to support chains of arcs.
          // The original state had `orig_num_arcs` leaving it, which is the
          // number of `foo` indexes minus one.
          int32_t orig_num_arcs = len2 - 1;
          num_arcs = orig_num_arcs;
        } else {
          // All newly-created states have exactly one arc leaving them.
          num_arcs = 1;
        }
        num_oarcs_data[idx0123] = num_arcs;
      });
  ExclusiveSum(num_oarcs, &num_oarcs);
  Array1<int32_t> &ostate_to_oarcs_row_splits = num_oarcs;
  RaggedShape ostate_to_oarcs =
      RaggedShape2(&ostate_to_oarcs_row_splits, nullptr, -1);

  // `full_shape` has 5 axes: [fsa][orig_state][foo][ostate][oarc]
  RaggedShape full_shape =
      ComposeRaggedShapes(to_ostates_shape, ostate_to_oarcs);
  // for the lower-order row-splits and row-ids, use tot_row_{splits,idx}n_data
  const int32_t *full_row_splits4_data = full_shape.RowSplits(4).Data(),
                *full_row_ids4_data = full_shape.RowIds(4).Data();
  int32_t tot_oarcs = full_shape.NumElements();
  K2_CHECK_GE(tot_oarcs, fsas.NumElements());

  int32_t *fsas_arc_map_data = nullptr, *labels_arc_map_data = nullptr;
  if (fsas_arc_map) {
    *fsas_arc_map = Array1<int32_t>(c, tot_oarcs);
    fsas_arc_map_data = fsas_arc_map->Data();
  }
  if (labels_arc_map) {
    *labels_arc_map = Array1<int32_t>(c, tot_oarcs);
    labels_arc_map_data = labels_arc_map->Data();
  }
  Array1<Arc> oarcs(c, tot_oarcs);
  Arc *oarcs_data = oarcs.Data();
  const Arc *arcs_data = fsas.values.Data();

  K2_EVAL(
      c, tot_oarcs, lambda_set_arcs, (int32_t idx01234)->void {
        // All these indexes are into `full_shape`, indexed
        // `[fsa][state][foo][ostate][oarc].`
        int32_t idx0123 = full_row_ids4_data[idx01234],
                idx0123x = full_row_splits4_data[idx0123],
                idx4 = idx01234 - idx0123x, idx012 = tos_row_ids3_data[idx0123],
                idx012x = tos_row_splits3_data[idx012],
                idx3 = idx0123 - idx012x, idx01 = tos_row_ids2_data[idx012],
                idx01x = tos_row_splits2_data[idx01], idx2 = idx012 - idx01x,
                idx0 = tos_row_ids1_data[idx01],
                idx0x = tos_row_splits1_data[idx0],
                idx0xxx = tos_row_splits3_data[tos_row_splits2_data[idx0x]];

        int32_t fsa_idx01x = fsas_row_splits2_data[idx01];

        int32_t fsa_idx2;  // the idx2 (arc-index) into `fsas` of the input arc
                           // that's most relevant to us..
        int32_t seq_pos;  // seq_pos is our index into the sequence of arcs that
                          // we produce for each original arc
        if (idx2 == 0) {
          K2_CHECK_EQ(idx3, 0);
          fsa_idx2 = idx4;  // corresponds to foo=0, so idx3 will be 0; the idx4
                            // enumerates the arcs leaving it..
          seq_pos = 0;
        } else {
          // this is one of the extra `foo` indexes, one per arc in the input
          // FSA that leaves this state; each of those `foo` indexes has
          // (seq_len - 1) states in it (idx3=0,1..seq_len-1); and each state
          // has one arc leaving it (idx4==0).
          K2_CHECK_EQ(idx4, 0);
          fsa_idx2 = idx2 - 1;
          seq_pos = idx3 + 1;
        }
        int32_t fsa_idx012 = fsa_idx01x + fsa_idx2;  // index of the arc in
                                                     // source FSA FSA that
                                                     // we're expanding..
        Arc iarc = arcs_data[fsa_idx012];

        int32_t labels_idx0x = labels_row_splits1_data[fsa_idx012],
                labels_next_idx0x = labels_row_splits1_data[fsa_idx012 + 1],
                labels_len1 = labels_next_idx0x - labels_idx0x;
        // labels_len1 is length of label sequence for this arc
        K2_CHECK_LT(seq_pos, max(int32_t(1), labels_len1));

        int32_t dest_idx01 = idx0x + iarc.dest_state,  // original destination
                                                       // state-index
            orig_dest_idx0123 =
                tos_row_splits3_data[tos_row_splits2_data[dest_idx01]];

        Arc oarc;
        oarc.src_state = idx0123 - idx0xxx;
        // If this is the last arc in the sequence, the dest-state is the
        // original dest-state of the arc.  Otherwise the dest-state is one of
        // the new states that we created. The idx123 will be an idx1 after
        // removing axes.
        int32_t dest_idx123;
        if (seq_pos + 1 >= labels_len1) {  // last arc in sequence..
          dest_idx123 = orig_dest_idx0123 - idx0xxx;
        } else {
          int32_t dest_state_idx2 = fsa_idx2 + 1,  // index `foo` equals
                                                   // orig_arc_idx+1
              dest_state_idx3 = seq_pos,           // ostate index..
              dest_idx012 = idx01x + dest_state_idx2,
                  dest_idx012x = tos_row_splits3_data[dest_idx012],
                  dest_idx0123 = dest_idx012x + dest_state_idx3;
          dest_idx123 = dest_idx0123 - idx0xxx;
        }
        oarc.dest_state = dest_idx123;  // indexes 1,2,3 will be combined; in
                                        // the output FSA it will be an idx1.

        if (fsas_arc_map_data)
          fsas_arc_map_data[idx01234] = (seq_pos == 0 ? fsa_idx012 : -1);
        if (labels_arc_map_data)
          labels_arc_map_data[idx01234] =
              (seq_pos < labels_len1 ? labels_idx0x + seq_pos : -1);
        if (iarc.label != -1) {
          // normal case.. label goes on 1st arc in sequence
          oarc.label = (seq_pos == 0 ? iarc.label : 0);
        } else {
          // If the arc was to the final-state, we need to keep the label on the
          // last arc of the sequence to keep the output valid.  The following
          // would be "seq_pos + 1 == labels_len1 ? -1 : 0", but we make it ">="
          // not "=" to account for the case seq_pos=0, labels_len1 = 0.
          oarc.label = (seq_pos + 1 >= labels_len1 ? -1 : 0);
        }
        oarc.score = (seq_pos == 0 ? iarc.score : 0.0);
        oarcs_data[idx01234] = oarc;
      });

  // remove current axes 1 and 2... [after removing axis 1, old axis 2 becomes
  // axis 1, so remove axis 1 twice].
  RaggedShape temp = RemoveAxis(full_shape, 1);
  return FsaVec(RemoveAxis(temp, 1), oarcs);
}

void Invert(FsaOrVec &src, Ragged<int32_t> &src_aux_labels, FsaOrVec *dest,
            Ragged<int32_t> *dest_aux_labels,
            Array1<int32_t> *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src_aux_labels.NumAxes(), 2);
  K2_CHECK_EQ(src_aux_labels.Dim0(), src.NumElements());
  K2_CHECK(dest != nullptr && dest_aux_labels != nullptr);
  ContextPtr c = GetContext(src, src_aux_labels);
  if (src.NumAxes() == 2) {
    Fsa *srcs = &src;
    FsaVec src_vec = CreateFsaVec(1, &srcs), dest_vec;
    Invert(src_vec, src_aux_labels, &dest_vec, dest_aux_labels, arc_map);
    *dest = GetFsaVecElement(dest_vec, 0);
    return;
  }
  Array1<int32_t> src_arc_map, labels_arc_map;
  *dest = ExpandArcs(src, src_aux_labels.shape, &src_arc_map, &labels_arc_map);
  // swap labels and aux_labels
  int32_t dest_num_arcs = dest->NumElements();
  Arc *dest_arcs_data = dest->values.Data();
  const int32_t *labels_arc_map_data = labels_arc_map.Data(),
                *src_aux_labels_data = src_aux_labels.values.Data();
  Array1<int32_t> dest_aux_labels_row_splits(c, dest_num_arcs + 1);
  int32_t *dest_aux_labels_row_splits_data = dest_aux_labels_row_splits.Data();
  K2_EVAL(
      c, dest_num_arcs, lambda_set_dest_aux_labels_num,
      (int32_t dest_idx012)->void {
        Arc &dest_arc = dest_arcs_data[dest_idx012];
        // we'll remove epsilons in dest_aux_labels
        dest_aux_labels_row_splits_data[dest_idx012] =
            dest_arc.label == 0 ? 0 : 1;
      });
  ExclusiveSum(dest_aux_labels_row_splits.Arange(0, dest_num_arcs),
               &dest_aux_labels_row_splits);
  RaggedShape dest_aux_labels_shape =
      RaggedShape2(&dest_aux_labels_row_splits, nullptr, -1);
  Array1<int32_t> dest_aux_labels_values(c,
                                         dest_aux_labels_shape.NumElements());
  int32_t *dest_aux_labels_values_data = dest_aux_labels_values.Data();
  K2_EVAL(
      c, dest_num_arcs, lambda_set_dest_labels_and_aux_labels,
      (int32_t dest_idx012)->void {
        Arc &dest_arc = dest_arcs_data[dest_idx012];
        // swap label and aux_label
        if (dest_arc.label != 0) {
          int32_t dest_aux_labels_idx0x =
              dest_aux_labels_row_splits_data[dest_idx012];
          // every arc in dest has at most one aux_label (as the aux_label is
          // the label of src on this arc)
          dest_aux_labels_values_data[dest_aux_labels_idx0x] = dest_arc.label;
        }
        int32_t src_aux_labels_idx01 = labels_arc_map_data[dest_idx012];
        dest_arc.label = src_aux_labels_idx01 == -1
                             ? 0
                             : src_aux_labels_data[src_aux_labels_idx01];
      });
  *dest_aux_labels =
      Ragged<int32_t>(dest_aux_labels_shape, dest_aux_labels_values);
  if (arc_map != nullptr) *arc_map = src_arc_map;
}

// Will be used in InvertHost to process FsaVec input recursively.
void RecursionWrapperAuxLabels(void (*f)(FsaOrVec &, Ragged<int32_t> &,
                                         FsaOrVec *, Ragged<int32_t> *),
                               FsaOrVec &src, Ragged<int32_t> &src_aux_labels,
                               FsaOrVec *dest,
                               Ragged<int32_t> *dest_aux_labels) {
  NVTX_RANGE(K2_FUNC);
  // src is actually an FsaVec.  Just recurse for now.
  K2_CHECK_EQ(src.NumAxes(), 3);
  int32_t num_fsas = src.shape.Dim0();
  std::vector<Fsa> srcs(num_fsas), dests(num_fsas);
  std::vector<Ragged<int32_t>> src_aux_labels_vec(num_fsas),
      dest_aux_labels_vec(num_fsas);
  int32_t tot_num_arcs = 0;
  Array1<int32_t> src_aux_labels_row_splits = src_aux_labels.RowSplits(1),
                  src_aux_labels_values = src_aux_labels.values;
  for (int32_t i = 0; i < num_fsas; ++i) {
    srcs[i] = src.Index(0, i);
    int32_t cur_num_arcs = srcs[i].NumElements();
    // below block get aux_labels for srcs[i]
    // TODO(haowen): replace with Range op for ragged
    {
      Array1<int32_t> row_splits = src_aux_labels_row_splits.Arange(
          tot_num_arcs, tot_num_arcs + cur_num_arcs + 1);
      Array1<int32_t> values =
          src_aux_labels_values.Arange(row_splits[0], row_splits.Back());
      row_splits = Minus(row_splits, row_splits[0]);
      RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
      src_aux_labels_vec[i] = Ragged<int32_t>(shape, values);
    }
    f(srcs[i], src_aux_labels_vec[i], &(dests[i]), &(dest_aux_labels_vec[i]));
    tot_num_arcs += cur_num_arcs;
  }
  *dest = Stack(0, num_fsas, dests.data());
  *dest_aux_labels = Append(0, num_fsas, dest_aux_labels_vec.data());
}

void InvertHost(FsaOrVec &src, Ragged<int32_t> &src_aux_labels, FsaOrVec *dest,
                Ragged<int32_t> *dest_aux_labels) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(src_aux_labels.NumAxes(), 2);
  K2_CHECK_EQ(src_aux_labels.Dim0(), src.NumElements());
  K2_CHECK(dest != nullptr && dest_aux_labels != nullptr);
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapperAuxLabels(InvertHost, src, src_aux_labels, dest,
                                     dest_aux_labels);
  }

  k2host::Fsa host_fsa = FsaToHostFsa(src);
  // k2host::AuxLabels is a k2host::Array2
  k2host::AuxLabels host_aux_labels(
      src_aux_labels.Dim0(), src_aux_labels.NumElements(),
      src_aux_labels.RowSplits(1).Data(), src_aux_labels.values.Data());
  k2host::FstInverter inverter(host_fsa, host_aux_labels);
  k2host::Array2Size<int32_t> fsa_size, aux_size;
  inverter.GetSizes(&fsa_size, &aux_size);
  FsaCreator fsa_creator(fsa_size);
  k2host::Fsa host_dest_fsa = fsa_creator.GetHostFsa();
  Ragged2Creator<int32_t> ragged_creator(aux_size);
  k2host::AuxLabels host_dest_aux_labels = ragged_creator.GetHostArray2();
  inverter.GetOutput(&host_dest_fsa, &host_dest_aux_labels);
  *dest = fsa_creator.GetFsa();
  *dest_aux_labels = ragged_creator.GetRagged2();
}

}  // namespace k2
