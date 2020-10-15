/**
 * @brief fsa_algo  Implementation of FSA algorithm wrappers from fsa_algo.h

 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/intersect.h"
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
  for (int32_t i = 0; i < num_fsas; i++) {
    srcs[i] = src.Index(0, i);
    // Recurse.
    if (!f(srcs[i], &(dests[i]), (arc_map ? &(arc_maps[i]) : nullptr)))
      return false;
  }
  *dest = Stack(0, num_fsas, &(dests[0]));
  if (arc_map) *arc_map = Append(num_fsas, &(arc_maps[0]));
  return true;
}

bool ConnectFsa(Fsa &src, Fsa *dest, Array1<int32_t> *arc_map) {
  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    return RecursionWrapper(ConnectFsa, src, dest, arc_map);
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

void Intersect(FsaOrVec &a_fsas, FsaOrVec &b_fsas,
               FsaVec *out,
               Array1<int32_t> *arc_map_a,
               Array1<int32_t> *arc_map_b) {
  K2_CHECK(a_fsas.NumAxes() >= 2 && a_fsas.NumAxes() <= 3);
  K2_CHECK(b_fsas.NumAxes() >= 2 && b_fsas.NumAxes() <= 3);
  ContextPtr c = a_fsas.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  if (a_fsas.NumAxes() == 2) {
    FsaVec a_fsas_vec = FsaToFsaVec(a_fsas);
    Intersect(a_fsas_vec, b_fsas, out, arc_map_a, arc_map_b);
    return;
  }
  if (b_fsas.NumAxes() == 2) {
    FsaVec b_fsas_vec = FsaToFsaVec(b_fsas);
    Intersect(a_fsas, b_fsas_vec, out, arc_map_a, arc_map_b);
    return;
  }

  int32_t num_fsas_a = a_fsas.Dim0(),
    num_fsas_b = b_fsas.Dim0();
  K2_CHECK_GT(num_fsas_a, 0);
  K2_CHECK_GT(num_fsas_b, 0);
  int32_t stride_a = 1, stride_b = 1;
  if (num_fsas_a != num_fsas_b) {
    if (num_fsas_a == 1) { stride_a = 0; }
    else if (num_fsas_b == 1) { stride_b = 0; }
    else { K2_CHECK_EQ(num_fsas_a, num_fsas_b); }
    // the check on the previous line will fail.
  }
  int32_t num_fsas = std::max(num_fsas_a, num_fsas_b);

  std::vector<std::unique_ptr<k2host::Intersection>> intersections(num_fsas);
  std::vector<k2host::Array2Size<int32_t>> sizes(num_fsas);
  for (int32_t i = 0; i < num_fsas; i++) {
    k2host::Fsa host_fsa_a = FsaVecToHostFsa(a_fsas, i * stride_a),
      host_fsa_b = FsaVecToHostFsa(b_fsas, i * stride_b);
    intersections[i] = std::make_unique<k2host::Intersection>(host_fsa_a,
                                                              host_fsa_b);
    intersections[i]->GetSizes(&(sizes[i]));
  }
  FsaVecCreator creator(sizes);
  int32_t num_arcs = creator.NumArcs();

  if (arc_map_a) *arc_map_a = Array1<int32_t>(c, num_arcs);
  if (arc_map_b) *arc_map_b = Array1<int32_t>(c, num_arcs);


  // the following few lines will allow us to add suitable offsets to the
  // `arc_map`.
  Array1<int32_t> a_fsas_row_splits12 = a_fsas.RowSplits(2)[a_fsas.RowSplits(1)],
    b_fsas_row_splits12 = b_fsas.RowSplits(2)[b_fsas.RowSplits(1)];
  const int32_t *a_fsas_row_splits12_data = a_fsas_row_splits12.Data(),
    *b_fsas_row_splits12_data = b_fsas_row_splits12.Data();

  for (int32_t i = 0; i < num_fsas; i++) {
    k2host::Fsa host_fsa_out = creator.GetHostFsa(i);
    int32_t arc_offset = creator.GetArcOffsetFor(i);
    int32_t *this_arc_map_a = (arc_map_a ? arc_map_a->Data() + arc_offset :
                               nullptr),
      *this_arc_map_b = (arc_map_b ? arc_map_b->Data() + arc_offset :
                         nullptr);
    bool ans = intersections[i]->GetOutput(&host_fsa_out,
                                           this_arc_map_a, this_arc_map_b);
    int32_t this_num_arcs = creator.GetArcOffsetFor(i+1) - arc_offset;
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
}

namespace {
struct ArcComparer {
  __host__ __device__ __forceinline__ bool operator()(const Arc &lhs,
                                                      const Arc &rhs) const {
    return lhs.symbol < rhs.symbol;
  }
};
}  // namespace

void ArcSort(Fsa *fsa) {
  if (fsa->NumAxes() < 2) return;  // it is empty
  SortSublists<Arc, ArcComparer>(fsa);
}

}  // namespace k2
