/**
 * @brief fsa_algo  Implementation of FSA algorithm wrappers from fsa_algo.h

 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/host/connect.h"
#include "k2/csrc/host_shim.h"

// this contains a subset of the algorithms in fsa_algo.h; currently it just
// contains one that are wrappings of the corresponding algorithms in
// host/.
namespace k2 {


inline bool RecursionWrapper(void (Fsa&,Fsa*,Array1<int32_t*>) *f,
                             Fsa &src, Fsa *dest, Array1<int32_t> *arc_map) {
  // src is actually an FsaVec.  Just recurse for now.
  int32_t num_fsas = src.Dim0();
  std::vector<Fsa> srcs(num_fsas),
      dests(num_fsas);
  std::vector<Array1<int32_t> > arc_maps(num_fsas);
  for (int32_t i = 0; i < num_fsas; i++) {
    srcs[i] = src.Index(0, i);
    // Recurse.
    if (! f(srcs[i], &(dests[i]),
            (arc_map ? &(arc_maps[i]) : nullptr)))
      return false;
  }
  *dest = Stack(0, num_srcs, &(dests[0]));
  if (arc_map)
    *arc_map = Append(num_srcs, &(arc_maps[0]));

  return true;
}

  int32_t num_axes = src.NumAxes();
  if (num_axes < 2 || num_axes > 3) {
    K2_LOG(FATAL) << "Input has bad num-axes " << num_axes;
  } else if (num_axes == 3) {
    RecursionWrapper
    // src is actually an FsaVec.  Just recurse for now.
    int32_t num_fsas = src.Dim0();
    std::vector<Fsa> srcs(num_fsas),
        dests(num_fsas);
    std::vector<Array1<int32_t> > arc_maps(num_fsas);
    for (int32_t i = 0; i < num_fsas; i++) {
      srcs[i] = src.Index(0, i);
      // Recurse.
      if (! ConnectFsa(srcs[i], &(dests[i]),
                       (arc_map ? &(arc_maps[i]) : nullptr)))
          return false;
    }
    *dest = Stack(0, num_srcs, &(dests[0]));
    if (arc_map)
      *arc_map = Append(num_srcs, &(arc_maps[0]));

    return true;
  }
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

}  // namespace k2
