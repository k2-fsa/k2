/**
 * @brief
 * host_shim
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <limits>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/host/weights.h"
#include "k2/csrc/host_shim.h"
#include "k2/csrc/ragged.h"

namespace k2 {

k2host::Fsa FsaToHostFsa(Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsa.NumAxes(), 2);
  K2_CHECK_EQ(fsa.Context()->GetDeviceType(), kCpu);
  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).
  return k2host::Fsa(fsa.Dim0(), fsa.TotSize(1), fsa.RowSplits(1).Data(),
                     reinterpret_cast<k2host::Arc *>(fsa.values.Data()));
}

k2host::Fsa FsaVecToHostFsa(FsaVec &fsa_vec, int32_t index) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsa_vec.NumAxes(), 3);
  K2_CHECK_LT(static_cast<uint32_t>(index),
              static_cast<uint32_t>(fsa_vec.Dim0()));
  K2_CHECK_EQ(fsa_vec.Context()->GetDeviceType(), kCpu);

  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).

  int32_t *row_splits1_data = fsa_vec.RowSplits(1).Data(),
          *row_splits2_data = fsa_vec.RowSplits(2).Data();
  Arc *arcs_data = fsa_vec.values.Data();
  int32_t start_state_idx01 = row_splits1_data[index],
          end_state_idx01 = row_splits1_data[index + 1],
          size1 = end_state_idx01 - start_state_idx01,
          start_arc_idx012 = row_splits2_data[start_state_idx01],
          end_arc_idx012 = row_splits2_data[end_state_idx01],
          size2 = end_arc_idx012 - start_arc_idx012;

  return k2host::Fsa(size1, size2, row_splits2_data + start_state_idx01,
                     reinterpret_cast<k2host::Arc *>(arcs_data));
}

void FsaVecCreator::Init(
    const std::vector<k2host::Array2Size<int32_t>> &sizes) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_fsas = static_cast<int32_t>(sizes.size());
  K2_CHECK_GT(num_fsas, 0);
  ContextPtr c = GetCpuContext();

  row_splits1_ = Array1<int32_t>(c, num_fsas + 1);
  row_splits12_ = Array1<int32_t>(c, num_fsas + 1);
  int32_t *row_splits1_data = row_splits1_.Data(),
          *row_splits12_data = row_splits12_.Data();
  for (int32_t i = 0; i < num_fsas; i++) {
    row_splits1_data[i] = sizes[i].size1;   // num_states
    row_splits12_data[i] = sizes[i].size2;  // num_arcs
  }
  ExclusiveSum(row_splits1_, &row_splits1_);
  ExclusiveSum(row_splits12_, &row_splits12_);

  int32_t tot_states = row_splits1_[num_fsas],
          tot_arcs = row_splits12_[num_fsas];
  row_splits2_ = Array1<int32_t>(c, tot_states + 1);
  arcs_ = Array1<Arc>(c, tot_arcs);

  finalized_row_splits2_ = false;
  next_fsa_idx_ = 0;
}

void FsaVecCreator::FinalizeRowSplits2() {
  NVTX_RANGE(K2_FUNC);
  if (finalized_row_splits2_) return;
  finalized_row_splits2_ = true;
  int32_t num_fsas = row_splits1_.Dim() - 1;
  K2_CHECK_EQ(next_fsa_idx_, num_fsas);

  const int32_t *row_splits1_data = row_splits1_.Data(),
                *row_splits12_data = row_splits12_.Data();
  int32_t *row_splits2_data = row_splits2_.Data();

  for (int32_t i = 0; i < num_fsas; i++) {
    int32_t num_states = row_splits1_data[i + 1] - row_splits1_data[i],
            begin_state = row_splits1_data[i], begin_arc = row_splits12_data[i];
    K2_CHECK(row_splits2_data[begin_state] == 0 || num_states == 0);
    // For the last FSA we need to modify the final element of row_splits2.
    // Note: for all but the last FSA, they would have written an element
    // to row_splits2_data which would have been overwritten when the next
    // FSA was processed.
    if (i + 1 == num_fsas) num_states++;
    for (int32_t j = 0; j < num_states; j++)
      row_splits2_data[begin_state + j] += begin_arc;
  }
}

k2host::Fsa FsaVecCreator::GetHostFsa(int32_t i) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(i, next_fsa_idx_);  // make sure they are called in order.
  next_fsa_idx_++;

  const int32_t *row_splits1_data = row_splits1_.Data(),
                *row_splits12_data = row_splits12_.Data();
  int32_t *row_splits2_data = row_splits2_.Data();
  int32_t num_states = row_splits1_data[i + 1] - row_splits1_data[i],
          num_arcs = row_splits12_data[i + 1] - row_splits12_data[i];
  k2host::Arc *arcs_data = reinterpret_cast<k2host::Arc *>(arcs_.Data());
  return k2host::Fsa(num_states, num_arcs,
                     row_splits2_data + row_splits1_data[i],
                     arcs_data + row_splits12_data[i]);
}

FsaVec FsaVecCreator::GetFsaVec() {
  NVTX_RANGE(K2_FUNC);
  FinalizeRowSplits2();
  return Ragged<Arc>(
      RaggedShape3(&row_splits1_, nullptr, -1, &row_splits2_, nullptr, -1),
      arcs_);
}

/*
  Check properties of FsaOrVec (which must be on CPU) with property test
  function `f` which is one of property test functions for k2host::Fsa in
  host/properties.h, e.g. IsValid(const k2::host Fsa&), IsTopSorted(const
  k2host::Fsa&).

  If `fsas` is FsaVec, the function will return an array on CPU which has
    ans[i] = f(FsaVecToHostFsa(fsa_vec, i)) for 0 <= i < fsa_vec.Dim0();
  else
    returns a CPU array with size 1 and ans[0] = f(FsaToHostFsa(fsas))
*/
static Array1<bool> CheckProperties(FsaOrVec &fsas,
                                    bool (*f)(const k2host::Fsa &)) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = fsas.Context();
  K2_CHECK_EQ(c->GetDeviceType(), kCpu);
  if (fsas.NumAxes() == 2) {
    k2host::Fsa host_fsa = FsaToHostFsa(fsas);
    bool status = f(host_fsa);
    return Array1<bool>(c, 1, status);
  } else {
    K2_CHECK_EQ(fsas.NumAxes(), 3);
    int32_t num_fsas = fsas.Dim0();
    Array1<bool> ans(c, num_fsas);
    bool *ans_data = ans.Data();
    for (int32_t i = 0; i != num_fsas; ++i) {
      k2host::Fsa host_fsa = FsaVecToHostFsa(fsas, i);
      ans_data[i] = f(host_fsa);
    }
    return ans;
  }
}

Array1<bool> IsTopSorted(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, k2host::IsTopSorted);
}

Array1<bool> IsArcSorted(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, k2host::IsArcSorted);
}

Array1<bool> HasSelfLoops(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, k2host::HasSelfLoops);
}

// As k2host::IsAcyclic has two input arguments, we create a wrapper function
// here so we can pass it to CheckProperties
static bool IsAcyclicWapper(const k2host::Fsa &fsa) {
  return k2host::IsAcyclic(fsa, nullptr);
}
Array1<bool> IsAcyclic(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, IsAcyclicWapper);
}

Array1<bool> IsDeterministic(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, k2host::IsDeterministic);
}

Array1<bool> IsEpsilonFree(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, k2host::IsEpsilonFree);
}

Array1<bool> IsConnected(FsaOrVec &fsas) {
  NVTX_RANGE(K2_FUNC);
  return CheckProperties(fsas, k2host::IsConnected);
}

bool IsRandEquivalentUnweighted(FsaOrVec &a, FsaOrVec &b,
                                bool treat_epsilons_specially /*=true*/,
                                std::size_t npath /*= 100*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(a.NumAxes(), 2);
  K2_CHECK_EQ(b.NumAxes(), a.NumAxes());
  if (a.Context()->GetDeviceType() != kCpu ||
      b.Context()->GetDeviceType() != kCpu) {
    FsaOrVec a_cpu = a.To(GetCpuContext()), b_cpu = b.To(GetCpuContext());
    return IsRandEquivalentUnweighted(a_cpu, b_cpu, treat_epsilons_specially,
                                      npath);
  }
  if (a.NumAxes() > 2) {
    for (int32_t i = 0; i < a.Dim0(); i++) {
      Fsa a_part = a.Index(0, i), b_part = b.Index(0, i);
      if (!IsRandEquivalentUnweighted(a_part, b_part, treat_epsilons_specially,
                                      npath))
        return false;
    }
    return true;
  }
  k2host::Fsa host_fsa_a = FsaToHostFsa(a);
  k2host::Fsa host_fsa_b = FsaToHostFsa(b);
  return k2host::IsRandEquivalent(host_fsa_a, host_fsa_b,
                                  treat_epsilons_specially, npath);
}

bool IsRandEquivalent(Fsa &a, Fsa &b, bool log_semiring,
                      float beam /*=k2host::kFloatInfinity*/,
                      bool treat_epsilons_specially /*=true*/,
                      float delta /*=1e-6*/, std::size_t npath /*= 100*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(a.NumAxes(), 2);
  K2_CHECK_EQ(b.NumAxes(), a.NumAxes());
  if (a.Context()->GetDeviceType() != kCpu ||
      b.Context()->GetDeviceType() != kCpu) {
    FsaOrVec a_cpu = a.To(GetCpuContext()), b_cpu = b.To(GetCpuContext());
    return IsRandEquivalent(a_cpu, b_cpu, log_semiring, beam,
                            treat_epsilons_specially, delta, npath);
  }
  if (a.NumAxes() > 2) {
    for (int32_t i = 0; i < a.Dim0(); i++) {
      Fsa a_part = a.Index(0, i), b_part = b.Index(0, i);
      if (!IsRandEquivalent(a_part, b_part, log_semiring, beam,
                            treat_epsilons_specially, delta, npath))
        return false;
    }
    return true;
  }
  k2host::Fsa host_fsa_a = FsaToHostFsa(a);
  k2host::Fsa host_fsa_b = FsaToHostFsa(b);
  if (log_semiring) {
    return k2host::IsRandEquivalent<k2host::kLogSumWeight>(
        host_fsa_a, host_fsa_b, beam, treat_epsilons_specially, delta, true,
        npath);
  } else {
    return k2host::IsRandEquivalent<k2host::kMaxWeight>(
        host_fsa_a, host_fsa_b, beam, treat_epsilons_specially, delta, true,
        npath);
  }
}

template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, bool log_semiring) {
  NVTX_RANGE(K2_FUNC);
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
  return state_scores.AsType<FloatType>();
}

template <typename FloatType>
Array1<FloatType> GetBackwardScores(
    FsaVec &fsas, const Array1<FloatType> *tot_scores /*= nullptr*/,
    bool log_semiring /*= true*/) {
  NVTX_RANGE(K2_FUNC);
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
  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  if (tot_scores != nullptr) {
    K2_CHECK_EQ(tot_scores->Context()->GetDeviceType(), kCpu);
    K2_CHECK_EQ(tot_scores->Dim(), num_fsas);
    const FloatType *tot_scores_data = tot_scores->Data();
    K2_EVAL(
        c, num_states, lambda_add_tot_scores, (int32_t state_idx01) {
          int32_t fsa_idx0 = fsa_row_ids1[state_idx01];
          if (tot_scores_data[fsa_idx0] != negative_infinity) {
            state_scores_data[state_idx01] -= tot_scores_data[fsa_idx0];
          } else {
            state_scores_data[state_idx01] = negative_infinity;
          }
        });
  }

  return state_scores.AsType<FloatType>();
}

// explicit instantiation here
template Array1<float> GetForwardScores<float>(FsaVec &fsas, bool log_semiring);
template Array1<double> GetForwardScores<double>(FsaVec &fsas,
                                                 bool log_semiring);
template Array1<float> GetBackwardScores<float>(FsaVec &fsas,
                                                const Array1<float> *tot_scores,
                                                bool log_semiring);
template Array1<double> GetBackwardScores<double>(
    FsaVec &fsas, const Array1<double> *tot_scores, bool log_semiring);

}  // namespace k2
