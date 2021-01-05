// k2/csrc/cudpp/cudpp_test.cu
//
// Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
//
// See LICENSE for clarification regarding multiple authors
//
#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/cudpp/cudpp.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"

int main() {
  using namespace k2;

  ContextPtr context = GetCudaContext();

  std::vector<int32_t> col_indexes{4, 5, 0, 1, 5, 3, 0, 2, 4, 5, 1, 4};
  std::vector<int32_t> _row_splits{0, 2, 5, 6, 9, 10, 12};

  Array1<int32_t> row_splits(context, _row_splits);
  // RaggedShape shape = RaggedShape2(&row_splits, nullptr, col_indexes.size());
  RaggedShape shape = RaggedShape2(&row_splits, nullptr, -1);
  Array1<int32_t> values(context, col_indexes);

  Ragged<int32_t> ragged(shape, values);
  Array1<int32_t> order = GetTransposeReordering(ragged, 6);
  K2_LOG(INFO) << order;

  return 0;
}
