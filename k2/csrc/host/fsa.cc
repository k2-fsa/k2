/**
 * @brief
 * fsa
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/fsa.h"

namespace {

// 64-byte alignment should be enough for AVX512 and other computations.
constexpr std::size_t kAlignment = 64;
static_assert((kAlignment & 15) == 0,
              "kAlignment should be at least multiple of 16");
static_assert(kAlignment % alignof(k2host::Arc) == 0, "");

inline std::size_t AlignTo(std::size_t b, std::size_t alignment) {
  // alignment should be power of 2
  return (b + alignment - 1) & (~(alignment - 1));
}

}  // namespace

namespace k2host {

std::ostream &operator<<(std::ostream &os, const Arc &arc) {
  os << arc.src_state << " " << arc.dest_state << " " << arc.label << " "
     << arc.weight;
  return os;
}

std::ostream &operator<<(std::ostream &os, const Fsa &fsa) {
  os << "num_states: " << fsa.NumStates() << "\n";
  os << "num_arcs: " << fsa.size2 << "\n";
  for (const auto &arc : fsa) {
    os << arc << "\n";
  }
  return os;
}

}  // namespace k2host
