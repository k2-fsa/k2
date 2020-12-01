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

  std::vector<float> _csr_val = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int32_t> _csr_col{4, 5, 0, 1, 5, 3, 0, 2, 4, 5, 1, 4};
  std::vector<int32_t> _csr_row{0, 2, 5, 6, 9, 10, 12};

  auto c = GetCudaContext();
  Array1<float> csr_val(c, _csr_val);
  Array1<int32_t> csr_col(c, _csr_col);
  Array1<int32_t> csr_row(c, _csr_row);

  Array1<float> csc_val(c, csr_val.Dim());
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
                                         &buffer_size  // bufferSize
  );
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
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
                              buffer.Data()              // buffer
  );
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
  CheckArrayData(csc_val,
                 std::vector<float>{2, 6, 3, 10, 7, 5, 0, 8, 11, 1, 4, 9});
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
  std::vector<float> _csr_val = {0, 1,  2,  3,  4,  5,  6,  7,  8,
                                 9, 10, 11, 12, 13, 14, 15, 16, 17};
  std::vector<int32_t> _csr_col{1, 1, 3, 3, 3, 0, 0, 2, 1,
                                0, 1, 1, 3, 4, 4, 4, 3, 5};
  std::vector<int32_t> _csr_row{0, 5, 8, 9, 13, 16, 18};

  auto c = GetCudaContext();
  Array1<float> csr_val(c, _csr_val);
  Array1<int32_t> csr_col(c, _csr_col);
  Array1<int32_t> csr_row(c, _csr_row);

  Array1<float> csc_val(c, csr_val.Dim());
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
                                         &buffer_size  // bufferSize
  );
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
                              buffer.Data()              // buffer
  );
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
  CheckArrayData(csc_val, std::vector<float>{5, 6, 9, 0, 1, 8, 10, 11, 7, 2, 3,
                                             4, 12, 16, 13, 14, 15, 17});
  K2_LOG(INFO) << csc_val;

  status = cusparseDestroy(handle);
  K2_CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS);
}

}  // namespace k2

int main() {
  k2::TestNoDuplicates();
  k2::TestDuplicates();
  return 0;
}
