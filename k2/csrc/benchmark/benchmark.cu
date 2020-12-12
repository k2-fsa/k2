/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#include <sstream>
#include <string>

#include "k2/csrc/benchmark/benchmark.h"

namespace k2 {

std::string BenchmarkRun::ToString() const {
  std::ostringstream os;
  os << name << "," << num_iter << "," << std::fixed << eplased_per_iter;
  return os.str();
}

}  // namespace k2
