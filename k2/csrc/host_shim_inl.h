/**
 * @brief
 * host_shim_inl
 *
 * @note
 * Don't include this file directly; it is included by host_shim.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_SHIM_INL_H_
#define K2_CSRC_HOST_SHIM_INL_H_

#ifndef IS_IN_K2_CSRC_HOST_SHIM_H_
#error "this file is supposed to be included only by host_shim.h"
#endif

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/host/weights.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/ragged.h"

namespace k2 {
namespace internal {
// Will be used in below functions `GetForwardScores` and `GetBackwardScores` to
// convert Array1<double> to Array1<float> (then return it) or return itself.
// May delete this finally as we just use it for current test purpose.
template <typename FloatType>
Array1<FloatType> ConvertArrayType(const Array1<double> &src);
template <>
inline Array1<double> ConvertArrayType<double>(const Array1<double> &src) {
  K2_CHECK_EQ(src.Context()->GetDeviceType(), kCpu);
  return src;
}
template <>
inline Array1<float> ConvertArrayType<float>(const Array1<double> &src) {
  ContextPtr &c = src.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  int32_t dim = src.Dim();
  Array1<float> dest(c, dim);
  float *dest_data = dest.Data();
  const double *src_data = src.Data();
  for (int32_t i = 0; i != dim; ++i)
    dest_data[i] = static_cast<float>(src_data[i]);
  return dest;
}
}  // namespace internal

template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, bool log_semiring) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  ContextPtr &c = fsas.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1);
  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();

  // GetForwardWeights in host/weights.h only accepts double array as the
  // returned state_scores
  Array1<double> state_scores(c, num_states);
  double *state_scores_data = state_scores.Data();
  for (int32_t i = 0; i != num_fsas; ++i) {
    k2host::Fsa host_fsa = FsaVecToHostFsa(fsas, i);
    double *this_fsa_state_scores_data = state_scores_data + fsa_row_splits1[i];
    if (log_semiring) {
      k2host::ComputeForwardLogSumWeights(host_fsa, this_fsa_state_scores_data);
    } else {
      k2host::ComputeForwardMaxWeights(host_fsa, this_fsa_state_scores_data);
    }
  }
  return internal::ConvertArrayType<FloatType>(state_scores);
}

template <typename FloatType>
Array1<FloatType> GetBackwardScores(
    FsaVec &fsas, const Array1<FloatType> *tot_scores /*= nullptr*/,
    bool log_semiring /*= true*/) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  ContextPtr &c = fsas.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1);
  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  const int32_t *fsa_row_ids1 = fsas.RowIds(1).Data();

  // GetBackwardWeights in host/weights.h only accepts double array as the
  // returned state_scores
  Array1<double> state_scores(c, num_states);
  double *state_scores_data = state_scores.Data();
  for (int32_t i = 0; i != num_fsas; ++i) {
    k2host::Fsa host_fsa = FsaVecToHostFsa(fsas, i);
    double *this_fsa_state_scores_data = state_scores_data + fsa_row_splits1[i];
    if (log_semiring) {
      k2host::ComputeBackwardLogSumWeights(host_fsa,
                                           this_fsa_state_scores_data);
    } else {
      k2host::ComputeBackwardMaxWeights(host_fsa, this_fsa_state_scores_data);
    }
  }

  // add negative of tot_scores[i] to each state score in fsa[i]
  if (tot_scores != nullptr) {
    K2_CHECK_EQ(tot_scores->Context()->GetDeviceType(), kCpu);
    K2_CHECK_EQ(tot_scores->Dim(), num_fsas);
    const FloatType *tot_scores_data = tot_scores->Data();
    auto lambda_add_tot_scores = [=] __host__ __device__(int32_t state_idx01) {
      int32_t fsa_idx0 = fsa_row_ids1[state_idx01];
      state_scores_data[state_idx01] -= tot_scores_data[fsa_idx0];
    };
    Eval(c, num_states, lambda_add_tot_scores);
  }

  return internal::ConvertArrayType<FloatType>(state_scores);
}

}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_INL_H_
