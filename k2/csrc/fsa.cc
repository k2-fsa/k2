// k2/csrc/fsa.cc

// Copyright (c)  2020  Daniel Povey
//                      Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa.h"

namespace {

// 64-byte alignment should be enough for AVX512 and other computations.
constexpr size_t kAlignment = 64;
static_assert((kAlignment & 15) == 0,
              "kAlignment should be at least multiple of 16");
static_assert(kAlignment % alignof(k2::Arc) == 0, "");

inline size_t AlignTo(size_t b, size_t alignment) {
  // alignment should be power of 2
  return (b + alignment - 1) & (~(alignment - 1));
}

}  // namespace

namespace k2 {

Cfsa::Cfsa()
    : num_states(0),
      begin_arc(0),
      end_arc(0),
      arc_indexes(nullptr),
      arcs(nullptr) {}

Cfsa::Cfsa(const Fsa &fsa) {
  begin_arc = 0;
  num_states = fsa.NumStates();
  if (num_states != 0) {
    // this is not an empty fsa
    arc_indexes = fsa.arc_indexes.data();
    end_arc = fsa.arc_indexes.back();
    arcs = const_cast<Arc *>(fsa.arcs.data());
  } else {
    // this is an empty fsa
    arc_indexes = nullptr;
    end_arc = 0;
    arcs = nullptr;
  }
}

CfsaVec::CfsaVec(size_t size, void *data)
    : data_(reinterpret_cast<int32_t *>(data)), size_(size) {
  const auto header = reinterpret_cast<const CfsaVecHeader *>(data_);
  num_fsas_ = header->num_fsas;
}

Cfsa CfsaVec::operator[](int32_t f) const {
  DCHECK_GE(f, 0);
  DCHECK_LT(f, num_fsas_);

  Cfsa cfsa;

  const auto header = reinterpret_cast<const CfsaVecHeader *>(data_);
  const auto state_offsets_array = data_ + header->state_offsets_start;

  int32_t num_states = state_offsets_array[f + 1] - state_offsets_array[f];
  if (num_states == 0) return cfsa;

  // we have to decrease num_states by one since the last entry of arc_indexes
  // is repeated.
  --num_states;
  DCHECK_GE(num_states, 2);

  const auto arc_indexes_array = data_ + header->arc_indexes_start;
  const auto arcs_array = reinterpret_cast<Arc *>(data_) + header->arcs_start;

  cfsa.num_states = num_states;
  cfsa.begin_arc = arc_indexes_array[state_offsets_array[f]];
  cfsa.end_arc = arc_indexes_array[state_offsets_array[f + 1] - 1];
  cfsa.arc_indexes = arc_indexes_array;
  cfsa.arcs = arcs_array;

  return cfsa;
}

size_t GetCfsaVecSize(const Cfsa &cfsa) {
  size_t res_bytes = 0;

  size_t header_bytes = sizeof(CfsaVecHeader);
  res_bytes += header_bytes;

  // padding to the alignment boundary for state_offsets_array
  res_bytes = AlignTo(res_bytes, kAlignment);

  // size in bytes for `int32_t state_offsets_array[num_fsas + 1];`
  size_t state_offsets_array_bytes = sizeof(int32_t) * 2;
  res_bytes += state_offsets_array_bytes;

  // padding to the alignment boundary for arc_indexes_array
  res_bytes = AlignTo(res_bytes, kAlignment);

  // size in bytes for `int32_t arc_indexes_array[num_states + num_fsas];`
  size_t arc_indexes_array_bytes = sizeof(int32_t) * (cfsa.num_states + 1);
  res_bytes += arc_indexes_array_bytes;

  static_assert((alignof(Arc) & 3) == 0,
                "The alignment of Arc should be multiple of 4");

  // padding to the alignment of `arcs_array`
  res_bytes = AlignTo(res_bytes, alignof(Arc));

  // size in bytes for `Arc arcs[num_arcs];`
  DCHECK_GE(cfsa.end_arc, cfsa.begin_arc);
  size_t arcs_array_bytes = sizeof(Arc) * (cfsa.end_arc - cfsa.begin_arc);
  res_bytes += arcs_array_bytes;

  return res_bytes;
}

size_t GetCfsaVecSize(const std::vector<Cfsa> &cfsas) {
  size_t res_bytes = 0;

  size_t header_bytes = sizeof(CfsaVecHeader);
  res_bytes += header_bytes;

  // padding to the alignment boundary for state_offsets_array
  res_bytes = AlignTo(res_bytes, kAlignment);

  // size in bytes for `int32_t state_offsets_array[num_fsas + 1];`
  size_t state_offsets_array_bytes = sizeof(int32_t) * (cfsas.size() + 1);
  res_bytes += state_offsets_array_bytes;

  // padding to the alignment boundary for arc_indexes_array
  res_bytes = AlignTo(res_bytes, kAlignment);

  size_t num_states = 0;
  int32_t num_arcs = 0;
  for (const auto &cfsa : cfsas) {
    num_states += cfsa.num_states;
    DCHECK_GE(cfsa.end_arc, cfsa.begin_arc);
    num_arcs += cfsa.end_arc - cfsa.begin_arc;
  }

  // size in bytes for `int32_t arc_indexes_array[num_states + num_fsas];`
  size_t arc_indexes_array_bytes =
      sizeof(int32_t) * (num_states + cfsas.size());
  res_bytes += arc_indexes_array_bytes;

  static_assert((alignof(Arc) & 3) == 0,
                "The alignment of Arc should be multiple of 4");

  // padding to the alignment of `arcs_array`
  res_bytes = AlignTo(res_bytes, alignof(Arc));

  // size in bytes for `Arc arcs[num_arcs];`
  size_t arcs_array_bytes = sizeof(Arc) * num_arcs;
  res_bytes += arcs_array_bytes;

  return res_bytes;
}

}  // namespace k2
