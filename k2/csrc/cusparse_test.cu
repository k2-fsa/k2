/**
 * @brief csr2csc with cuSparse.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cusparse.h>

#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/log.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

void TestNoDuplicates() {
  //       col0  col1  col2  col3  col4  col5
  // row0                           a0    b1
  // row1   c2    d3                      e4
  // row2                     f5
  // row3   g6          h7          i8
  // row4                                 j9
  // row5         k10               l11

  std::vector<int32_t> _csr_val = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int32_t> _csr_col{4, 5, 0, 1, 5, 3, 0, 2, 4, 5, 1, 4};
  std::vector<int32_t> _csr_row{0, 2, 5, 6, 9, 10, 12};

  auto c = GetCudaContext();
  Array1<int32_t> csr_val(c, _csr_val);
  Array1<int32_t> csr_col(c, _csr_col);
  Array1<int32_t> csr_row(c, _csr_row);

  Array1<int32_t> csc_val(c, csr_val.Dim());
  Array1<int32_t> csc_col(c, 7);
  Array1<int32_t> csc_row(c, csr_val.Dim());

  cusparseHandle_t handle = nullptr;
  cusparseStatus_t status = cusparseCreate(&handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);

  size_t buffer_size;
  status = cusparseCsr2cscEx2_bufferSize(
      handle,
      6,               // m, number of rows
      6,               // n, number of cols
      csr_val.Dim(),   // nnz
      csr_val.Data(),  // csrVal
      csr_row.Data(),  // csrRowPtr
      csr_col.Data(),  // csrColInd
      csc_val.Data(),  // cscVal
      csc_col.Data(),  // cscColPtr
      csc_row.Data(),  // cscRowInd
      // NOTE: CUDA_R_32I will cause "invalid value error"
      CUDA_R_32F,                //  valType
      CUSPARSE_ACTION_NUMERIC,   // copyValues
      CUSPARSE_INDEX_BASE_ZERO,  // idxBase
      CUSPARSE_CSR2CSC_ALG1,     // alg
      &buffer_size);             // bufferSize

  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS)
      << "\n"
      << "Error name: " << cusparseGetErrorName(status) << "\n"
      << "Error string: " << cusparseGetErrorString(status) << "\n";
  Array1<int8_t> buffer(c, buffer_size);

  status = cusparseCsr2cscEx2(handle,
                              6,                         // m, number of rows
                              6,                         // n, number of cols
                              csr_val.Dim(),             // nnz
                              csr_val.Data(),            // csrVal
                              csr_row.Data(),            // csrRowPtr
                              csr_col.Data(),            // csrColInd
                              csc_val.Data(),            // cscVal
                              csc_col.Data(),            // cscColPtr
                              csc_row.Data(),            // cscRowInd
                              CUDA_R_32F,                //  valType
                              CUSPARSE_ACTION_NUMERIC,   // copyValues
                              CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                              CUSPARSE_CSR2CSC_ALG1,     // alg
                              buffer.Data());            // buffer
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
  CheckArrayData(csc_val,
                 std::vector<int32_t>{2, 6, 3, 10, 7, 5, 0, 8, 11, 1, 4, 9});
  K2_LOG(INFO) << csc_val;

  status = cusparseDestroy(handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
}

void TestDuplicates() {
  //       col0   col1   col2    col3      col4      col5
  // row0         a0,a1         b2,b3,b4
  // row1  c5,c6          d7
  // row2         e8
  // row3   f9   g10,g11         h12
  // row4                                i13,i14,i15
  // row5                        j16                  k17
  std::vector<int32_t> _csr_val = {0, 1,  2,  3,  4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16, 17};
  std::vector<int32_t> _csr_col{1, 1, 3, 3, 3, 0, 0, 2, 1,
                                0, 1, 1, 3, 4, 4, 4, 3, 5};
  std::vector<int32_t> _csr_row{0, 5, 8, 9, 13, 16, 18};

  auto c = GetCudaContext();
  Array1<int32_t> csr_val(c, _csr_val);
  Array1<int32_t> csr_col(c, _csr_col);
  Array1<int32_t> csr_row(c, _csr_row);

  Array1<int32_t> csc_val(c, csr_val.Dim());
  Array1<int32_t> csc_col(c, 7);
  Array1<int32_t> csc_row(c, csr_val.Dim());

  cusparseHandle_t handle = nullptr;
  cusparseStatus_t status = cusparseCreate(&handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);

  size_t buffer_size;
  status = cusparseCsr2cscEx2_bufferSize(handle,
                                         6,               // m, number of rows
                                         6,               // n, number of cols
                                         csr_val.Dim(),   // nnz
                                         csr_val.Data(),  // csrVal
                                         csr_row.Data(),  // csrRowPtr
                                         csr_col.Data(),  // csrColInd
                                         csc_val.Data(),  // cscVal
                                         csc_col.Data(),  // cscColPtr
                                         csc_row.Data(),  // cscRowInd
                                         CUDA_R_32F,      //  valType
                                         CUSPARSE_ACTION_NUMERIC,  // copyValues
                                         CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                                         CUSPARSE_CSR2CSC_ALG1,     // alg
                                         &buffer_size);  // bufferSize
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS)
      << "\n"
      << "Error name: " << cusparseGetErrorName(status) << "\n"
      << "Error string: " << cusparseGetErrorString(status) << "\n";

  Array1<int8_t> buffer(c, buffer_size);

  status = cusparseCsr2cscEx2(handle,
                              6,                         // m, number of rows
                              6,                         // n, number of cols
                              csr_val.Dim(),             // nnz
                              csr_val.Data(),            // csrVal
                              csr_row.Data(),            // csrRowPtr
                              csr_col.Data(),            // csrColInd
                              csc_val.Data(),            // cscVal
                              csc_col.Data(),            // cscColPtr
                              csc_row.Data(),            // cscRowInd
                              CUDA_R_32F,                //  valType
                              CUSPARSE_ACTION_NUMERIC,   // copyValues
                              CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                              CUSPARSE_CSR2CSC_ALG1,     // alg
                              buffer.Data());            // buffer
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
  CheckArrayData(csc_val, std::vector<int32_t>{5, 6, 9, 0, 1, 8, 10, 11, 7, 2,
                                               3, 4, 12, 16, 13, 14, 15, 17});
  K2_LOG(INFO) << csc_val;

  status = cusparseDestroy(handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
}

Array1<int32_t> GetTransposeReorderingCuSparse(cusparseHandle_t handle,
                                               Ragged<int32_t> &src,
                                               int32_t num_cols) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  Array1<int32_t> csr_val = Range(c, src.values.Dim(), 0);
  Array1<int32_t> csr_col = src.values;

  int32_t num_axes = src.NumAxes();
  Array1<int32_t> csr_row = src.RowSplits(num_axes - 1);

  Array1<int32_t> csc_val(c, csr_val.Dim());
  Array1<int32_t> csc_col(c, num_cols + 1);
  Array1<int32_t> csc_row(c, csr_val.Dim());

  int32_t num_rows = csr_row.Dim() - 1;

  size_t buffer_size;

  cusparseStatus_t status =
      cusparseCsr2cscEx2_bufferSize(handle,
                                    num_rows,        // m, number of rows
                                    num_cols,        // n, number of cols
                                    csr_val.Dim(),   // nnz
                                    csr_val.Data(),  // csrVal
                                    csr_row.Data(),  // csrRowPtr
                                    csr_col.Data(),  // csrColInd
                                    csc_val.Data(),  // cscVal
                                    csc_col.Data(),  // cscColPtr
                                    csc_row.Data(),  // cscRowInd
                                    CUDA_R_32F,      // valType
                                    CUSPARSE_ACTION_NUMERIC,   // copyValues
                                    CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                                    CUSPARSE_CSR2CSC_ALG1,     // alg
                                    &buffer_size);             // bufferSize

  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS)
      << "\n"
      << "Error name: " << cusparseGetErrorName(status) << "\n"
      << "Error string: " << cusparseGetErrorString(status) << "\n";
  Array1<int8_t> buffer(c, buffer_size);

  status =
      cusparseCsr2cscEx2(handle,
                         num_rows,        // m, number of rows
                         num_cols,        // n, number of cols
                         csr_val.Dim(),   // nnz
                         csr_val.Data(),  // csrVal
                         csr_row.Data(),  // csrRowPtr
                         csr_col.Data(),  // csrColInd
                         csc_val.Data(),  // cscVal
                         csc_col.Data(),  // cscColPtr
                         csc_row.Data(),  // cscRowInd
                         // NOTE: CUDA_R_32I will cause "invalid value error"
                         CUDA_R_32F,                // valType
                         CUSPARSE_ACTION_NUMERIC,   // copyValues
                         CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                         CUSPARSE_CSR2CSC_ALG1,     // alg
                         buffer.Data());            // buffer
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);

  return csc_val;
}

void TestRandom() {
  cusparseHandle_t handle = nullptr;
  cusparseStatus_t status = cusparseCreate(&handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
  ContextPtr c = GetCudaContext();

  for (int32_t i = 0; i != 10; ++i) {
    FsaVec fsa_vec = RandomFsaVec(100,     // min_num_fsas
                                  300,     // max_num_fsas
                                  true,    // acyclic
                                  1000,    // max_symbol
                                  10000,   // min_num_arcs
                                  20000);  // max_num_arcs
    fsa_vec = fsa_vec.To(c);
    Array1<int32_t> dest_states = GetDestStates(fsa_vec, true);
    Ragged<int32_t> dest_states_tensor(fsa_vec.shape, dest_states);
    Ragged<int32_t> dest_states_tensor_cloned(fsa_vec.shape,
                                              dest_states.Clone());

    int32_t num_states = fsa_vec.TotSize(1);

    Array1<int32_t> incoming_arcs_order;
    Array1<int32_t> incoming_arcs_order_cusparse;
    if (i % 2 == 0) {
      // first we run moderngpu, and then cuSparse
      {
        NVTX_RANGE("moderngpu");

        incoming_arcs_order =
            GetTransposeReordering(dest_states_tensor, num_states);
      }

      {
        NVTX_RANGE("cuSparse");

        incoming_arcs_order_cusparse = GetTransposeReorderingCuSparse(
            handle, dest_states_tensor_cloned, num_states);
      }
    } else {
      // first we run cuSparse, and then moderngpu
      {
        NVTX_RANGE("cuSparse");

        incoming_arcs_order_cusparse = GetTransposeReorderingCuSparse(
            handle, dest_states_tensor_cloned, num_states);
      }

      {
        NVTX_RANGE("moderngpu");

        incoming_arcs_order =
            GetTransposeReordering(dest_states_tensor, num_states);
      }
    }

    CheckArrayData(incoming_arcs_order, incoming_arcs_order_cusparse);
  }

  status = cusparseDestroy(handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
}

}  // namespace k2

int main() {
  // k2::TestNoDuplicates();
  // k2::TestDuplicates();
  k2::TestRandom();
  return 0;
}
