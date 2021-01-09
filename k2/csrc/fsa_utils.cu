/**
 * @brief Utilities for creating FSAs.
 *
 * Note that serializations are done in Python.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"

namespace k2 {

// field separator within a line for a text form FSA
static constexpr const char *kDelim = " \t";

// Convert a string to an integer. Abort the program on failure.
static int32_t StringToInt(const std::string &s) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(!s.empty());

  bool ok = false;
  char *p = nullptr;
  // std::strtol requires a `long` type
  long n = std::strtol(s.c_str(), &p, 10);  // NOLINT
  if (*p == '\0') ok = true;

  auto res = static_cast<int32_t>(n);
  if (n != res) ok = false;  // out of range

  K2_CHECK(ok) << "Failed to convert " << s << " to an integer";

  return res;
}

// Convert a string to a float. Abort the program on failure.
// TODO(guoguo): We may run into locale problems, with comma vs. period for
//               decimals. We have to test if the C code will behave the same
//               w.r.t. locale as Python does.
static float StringToFloat(const std::string &s) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(!s.empty());
  char *p = nullptr;
  float f = std::strtof(s.c_str(), &p);
  if (*p != '\0') K2_LOG(FATAL) << "Failed to convert " << s << " to a float";
  return f;
}

// Trim leading and trailing spaces of a string.
static void TrimString(std::string *s) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(s, nullptr);
  auto not_space = [](int32_t c) -> bool { return std::isspace(c) == 0; };

  s->erase(s->begin(), std::find_if(s->begin(), s->end(), not_space));
  s->erase(std::find_if(s->rbegin(), s->rend(), not_space).base(), s->end());
}

/* Split a string to a vector of strings using a set of delimiters.

   Example usage:

   @code
    std::string in = "1 2 3";
    const char *delim = " \t";
    std::vector<std::string> out;
    SplitStringToVector(in, delim, &out);
   @endcode

   @param [in]  in    The input string to be split.
   @param [in]  delim A string of delimiters.
   @param [out] out   It saves the split result.
*/
static void SplitStringToVector(const std::string &in, const char *delim,
                                std::vector<std::string> *out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(delim, nullptr);
  K2_CHECK_NE(out, nullptr);
  out->clear();
  std::size_t start = 0;
  while (true) {
    auto pos = in.find_first_of(delim, start);
    if (pos == std::string::npos) break;

    auto sub = in.substr(start, pos - start);
    start = pos + 1;

    TrimString(&sub);
    if (!sub.empty()) out->emplace_back(std::move(sub));
  }

  if (start < in.size()) {
    auto sub = in.substr(start);
    TrimString(&sub);
    if (!sub.empty()) out->emplace_back(std::move(sub));
  }
}

/* Create an acceptor from a stream, assuming the acceptor is in the k2 format:

   src_state1 dest_state1 label1 score1
   src_state2 dest_state2 label2 score2
   ... ...
   final_state

   The source states will be in non-descending order, and the final state does
   not bear a cost/score -- we put the cost/score on the arc that connects to
   the final state and set its label to -1.

   @param [in]  is    The input stream that contains the acceptor.

   @return It returns an Fsa on CPU.
*/
static Fsa K2AcceptorFromStream(std::istringstream &is) {
  NVTX_RANGE(K2_FUNC);
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  bool finished = false;  // when the final state is read, set it to true.
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    K2_CHECK_EQ(finished, false);

    auto num_fields = splits.size();
    if (num_fields == 4u) {
      //   0            1          2      3
      // src_state  dest_state   label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      float score = StringToFloat(splits[3]);
      arcs.emplace_back(src_state, dest_state, symbol, score);
    } else if (num_fields == 1u) {
      //   0
      // final_state
      (void)StringToInt(splits[0]);  // this is a final state
      finished = true;               // set finish
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nk2 acceptor expects a line with 1 (final_state) or "
                       "4 (src_state dest_state label score) fields";
    }
  }

  K2_CHECK_EQ(finished, true) << "The last line should be the final state";

  bool error = true;
  Array1<Arc> array(GetCpuContext(), arcs);
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create a transducer from a stream, assuming the transducer is in the K2
   format:

   src_state1 dest_state1 label1 aux_label1 score1
   src_state2 dest_state2 label2 aux_label2 score2
   ... ...
   final_state

   The source states will be in non-descending order, and the final state does
   not bear a cost/score -- we put the cost/score on the arc that connects to
   the final state and set its label to -1.

   @param [in]  is    The input stream that contains the transducer.

   @return It returns an Fsa on CPU.
*/
static Fsa K2TransducerFromStream(std::istringstream &is,
                                  Array1<int32_t> *aux_labels) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(aux_labels != nullptr);

  std::vector<int32_t> aux_labels_internal;
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  bool finished = false;  // when the final state is read, set it to true.
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    K2_CHECK_EQ(finished, false);

    auto num_fields = splits.size();
    if (num_fields == 5u) {
      //   0           1         2         3        4
      // src_state  dest_state label   aux_label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      int32_t aux_label = StringToInt(splits[3]);
      float score = StringToFloat(splits[4]);
      arcs.emplace_back(src_state, dest_state, symbol, score);
      aux_labels_internal.push_back(aux_label);
    } else if (num_fields == 1u) {
      //   0
      // final_state
      (void)StringToInt(splits[0]);
      finished = true;  // set finish
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nk2 transducer expects a line with 1 (final_state) or "
                       "5 (src_state dest_state label aux_label score) fields";
    }
  }

  K2_CHECK_EQ(finished, true) << "The last line should be the final state";

  auto cpu_context = GetCpuContext();
  *aux_labels = Array1<int32_t>(cpu_context, aux_labels_internal);
  Array1<Arc> array(cpu_context, arcs);

  bool error = true;
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create an acceptor from a stream, assuming the acceptor is in the OpenFST
   format:

   src_state1 dest_state1 label1 score1
   src_state2 dest_state2 label2 score2
   ... ...
   final_state final_score

   We will negate the cost/score when we read them in. Also note, OpenFST may
   omit the cost/score if it is 0.0.

   We always create the super final state. If there are final state(s) in the
   original FSA, then we add arc(s) from the original final state(s) to the
   super final state, with the (negated) old final state cost/score as its
   cost/score, and -1 as its label.

   @param [in]  is    The input stream that contains the acceptor.

   @return It returns an Fsa on CPU.
*/
static Fsa OpenFstAcceptorFromStream(std::istringstream &is) {
  NVTX_RANGE(K2_FUNC);
  std::vector<Arc> arcs;
  std::vector<std::vector<Arc>> state_to_arcs;  // indexed by states
  std::vector<std::string> splits;
  std::string line;

  int32_t max_state = -1;
  int32_t num_arcs = 0;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();
    if (num_fields == 3u || num_fields == 4u) {
      //   0            1          2
      // src_state  dest_state   label
      //
      // or
      //
      //   0            1          2      3
      // src_state  dest_state   label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      float score = 0.0f;
      if (num_fields == 4u) score = -1.0f * StringToFloat(splits[3]);

      // Add the arc to "state_to_arcs".
      ++num_arcs;
      max_state = std::max(max_state, std::max(src_state, dest_state));
      if (static_cast<int32_t>(state_to_arcs.size()) <= src_state)
        state_to_arcs.resize(src_state + 1);
      state_to_arcs[src_state].emplace_back(src_state, dest_state, symbol,
                                            score);
    } else if (num_fields == 1u || num_fields == 2u) {
      //   0            1
      // final_state  score
      float score = 0.0f;
      if (num_fields == 2u) score = -1.0f * StringToFloat(splits[1]);
      original_final_states.push_back(StringToInt(splits[0]));
      original_final_weights.push_back(score);
      max_state = std::max(max_state, original_final_states.back());
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nOpenFST acceptor expects a line with 1 (final_state),"
                       " 2 (final_state score), 3 (src_state dest_state label) "
                       "or 4 (src_state dest_state label score) fields.";
    }
  }

  K2_CHECK(is.eof());

  // Post processing on final states. If there are final state(s) in the
  // original FSA, we add the super final state as well as arc(s) from original
  // final state(s) to the super final state. Otherwise, the super final state
  // will be added by FsaFromArray1 (since there's no arc with label
  // kFinalSymbol).
  if (original_final_states.size() > 0) {
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    int32_t super_final_state = max_state + 1;
    state_to_arcs.resize(super_final_state);
    for (std::size_t i = 0; i != original_final_states.size(); ++i) {
      state_to_arcs[original_final_states[i]].emplace_back(
          original_final_states[i], super_final_state,
          -1,  // kFinalSymbol
          original_final_weights[i]);
      ++num_arcs;
    }
  }

  // Move arcs from "state_to_arcs" to "arcs".
  int32_t arc_index = 0;
  arcs.resize(num_arcs);
  for (std::size_t s = 0; s < state_to_arcs.size(); ++s) {
    for (std::size_t a = 0; a < state_to_arcs[s].size(); ++a) {
      K2_CHECK_GT(num_arcs, arc_index);
      arcs[arc_index] = state_to_arcs[s][a];
      ++arc_index;
    }
  }
  K2_CHECK_EQ(num_arcs, arc_index);

  bool error = true;
  Array1<Arc> array(GetCpuContext(), arcs);
  // FsaFromArray1 will add a super final state if the original FSA doesn't have
  // a final state.
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create a transducer from a stream, assuming the transducer is in the OpenFST
   format:

   src_state1 dest_state1 label1 aux_label1 score1
   src_state2 dest_state2 label2 aux_label2 score2
   ... ...
   final_state final_score

   We will negate the cost/score when we read them in. Also note, OpenFST may
   omit the cost/score if it is 0.0.

   We always create the super final state. If there are final state(s) in the
   original FST, then we add arc(s) from the original final state(s) to the
   super final state, with the (negated) old final state cost/score as its
   cost/score, -1 as its label and -1 as its aux_label.

   @param [in]  is    The input stream that contains the transducer.

   @return It returns an Fsa on CPU.
*/
static Fsa OpenFstTransducerFromStream(std::istringstream &is,
                                       Array1<int32_t> *aux_labels) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(aux_labels != nullptr);

  std::vector<std::vector<int32_t>> state_to_aux_labels;  // indexed by states
  std::vector<std::vector<Arc>> state_to_arcs;            // indexed by states
  std::vector<int32_t> aux_labels_internal;
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  int32_t max_state = -1;
  int32_t num_arcs = 0;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();
    if (num_fields == 4u || num_fields == 5u) {
      //   0           1         2         3
      // src_state  dest_state label   aux_label
      //
      // or
      //
      //   0           1         2         3        4
      // src_state  dest_state label   aux_label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      int32_t aux_label = StringToInt(splits[3]);
      float score = 0.0f;
      if (num_fields == 5u) score = -1.0f * StringToFloat(splits[4]);

      // Add the arc to "state_to_arcs", and aux_label to "state_to_aux_labels"
      ++num_arcs;
      max_state = std::max(max_state, std::max(src_state, dest_state));
      if (static_cast<int32_t>(state_to_arcs.size()) <= src_state) {
        state_to_arcs.resize(src_state + 1);
        state_to_aux_labels.resize(src_state + 1);
      }
      state_to_arcs[src_state].emplace_back(src_state, dest_state, symbol,
                                            score);
      state_to_aux_labels[src_state].push_back(aux_label);
    } else if (num_fields == 1u || num_fields == 2u) {
      //   0
      // final_state
      //
      // or
      //
      //   0            1
      // final_state  score
      // There could be multiple final states, so we first have to collect all
      // the final states, and then work out the super final state.
      float score = 0.0f;
      if (num_fields == 2u) score = -1.0f * StringToFloat(splits[1]);
      original_final_states.push_back(StringToInt(splits[0]));
      original_final_weights.push_back(score);
      max_state = std::max(max_state, original_final_states.back());
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nOpenFST transducer expects a line with "
                       "1 (final_state), 2 (final_state score), "
                       "4 (src_state dest_state label aux_label) or "
                       "5 (src_state dest_state label aux_label score) fields.";
    }
  }

  K2_CHECK(is.eof());

  // Post processing on final states. If there are final state(s) in the
  // original FST, we add the super final state as well as arc(s) from original
  // final state(s) to the super final state. Otherwise, the super final state
  // will be added by FsaFromArray1 (since there's no arc with label
  // kFinalSymbol).
  if (original_final_states.size() > 0) {
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    int32_t super_final_state = max_state + 1;
    state_to_arcs.resize(super_final_state);
    state_to_aux_labels.resize(super_final_state);
    for (std::size_t i = 0; i != original_final_states.size(); ++i) {
      state_to_arcs[original_final_states[i]].emplace_back(
          original_final_states[i], super_final_state,
          -1,  // kFinalSymbol
          original_final_weights[i]);
      state_to_aux_labels[original_final_states[i]].push_back(
          -1);  // kFinalSymbol
      ++num_arcs;
    }
  }

  // Move arcs from "state_to_arcs" to "arcs", and aux_labels from
  // "state_to_aux_labels" to "aux_labels_internal"
  int32_t arc_index = 0;
  arcs.resize(num_arcs);
  aux_labels_internal.resize(num_arcs);
  K2_CHECK_EQ(state_to_arcs.size(), state_to_aux_labels.size());
  for (std::size_t s = 0; s < state_to_arcs.size(); ++s) {
    K2_CHECK_EQ(state_to_arcs[s].size(), state_to_aux_labels[s].size());
    for (std::size_t a = 0; a < state_to_arcs[s].size(); ++a) {
      K2_CHECK_GT(num_arcs, arc_index);
      arcs[arc_index] = state_to_arcs[s][a];
      aux_labels_internal[arc_index] = state_to_aux_labels[s][a];
      ++arc_index;
    }
  }
  K2_CHECK_EQ(num_arcs, arc_index);

  auto cpu_context = GetCpuContext();
  *aux_labels = Array1<int32_t>(cpu_context, aux_labels_internal);
  Array1<Arc> array(cpu_context, arcs);

  bool error = true;
  // FsaFromArray1 will add a super final state if the original FSA doesn't have
  // a final state.
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

Fsa FsaFromString(const std::string &s, bool openfst /*= false*/,
                  Array1<int32_t> *aux_labels /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  std::istringstream is(s);
  K2_CHECK(is);

  if (openfst == false && aux_labels == nullptr)
    return K2AcceptorFromStream(is);
  else if (openfst == false && aux_labels != nullptr)
    return K2TransducerFromStream(is, aux_labels);
  else if (openfst == true && aux_labels == nullptr)
    return OpenFstAcceptorFromStream(is);
  else if (openfst == true && aux_labels != nullptr)
    return OpenFstTransducerFromStream(is, aux_labels);

  return Fsa();  // unreachable code
}

std::string FsaToString(const Fsa &fsa, bool openfst /*= false*/,
                        const Array1<int32_t> *aux_labels /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsa.NumAxes(), 2);

  if (fsa.Context()->GetDeviceType() != kCpu) {
    Fsa _fsa = fsa.To(GetCpuContext());
    Array1<int32_t> _aux_labels;
    if (aux_labels) _aux_labels = aux_labels->To(_fsa.Context());
    return FsaToString(_fsa, openfst, aux_labels ? &_aux_labels : nullptr);
  }

  K2_CHECK_EQ(fsa.Context()->GetDeviceType(), kCpu);
  const Array1<int32_t> &row_splits = fsa.shape.RowSplits(1);
  const Array1<Arc> &arcs = fsa.values;

  const int32_t *p = nullptr;
  if (aux_labels != nullptr) {
    K2_CHECK(IsCompatible(fsa, *aux_labels));
    K2_CHECK_EQ(aux_labels->Dim(), arcs.Dim());
    p = aux_labels->Data();
  }
  float scale = 1;
  if (openfst) scale = -1;

  std::ostringstream os;

  int32_t n = arcs.Dim();
  char sep = ' ';
  char line_sep = '\n';
  for (int32_t i = 0; i != n; ++i) {
    const auto &arc = arcs[i];
    os << arc.src_state << sep << arc.dest_state << sep << arc.label << sep;
    if (p != nullptr) os << p[i] << sep;
    os << (scale * arc.score) << line_sep;
  }
  os << (fsa.shape.Dim0() - 1) << line_sep;
  return os.str();
}

Array1<int32_t> GetDestStates(FsaVec &fsas, bool as_idx01) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_arcs = fsas.NumElements();
  Array1<int32_t> ans(c, num_arcs);
  const Arc *arcs_data = fsas.values.Data();
  int32_t *ans_data = ans.Data();
  if (!as_idx01) {
    K2_EVAL(
        c, num_arcs, lambda_set_dest_states1, (int32_t arc_idx012) {
          ans_data[arc_idx012] = arcs_data[arc_idx012].dest_state;
        });
  } else {
    const int32_t *row_ids2 = fsas.RowIds(2).Data();
    K2_EVAL(
        c, num_arcs, lambda_set_dest_states01, (int32_t arc_idx012) {
          int32_t src_state = arcs_data[arc_idx012].src_state,
                  dest_state = arcs_data[arc_idx012].dest_state;
          // (row_ids2[arc_idx012] - src_state) is the same as
          // row_splits1[row_ids1[row_ids2[arc_idx012]]]; it's the idx01 of the
          // 1st state in this FSA.
          ans_data[arc_idx012] =
              dest_state + (row_ids2[arc_idx012] - src_state);
        });
  }
  return ans;
}

Ragged<int32_t> GetStateBatches(FsaVec &fsas, bool transpose) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  Array1<int32_t> arc_dest_states = GetDestStates(fsas, true);

  MonotonicLowerBound(arc_dest_states, &arc_dest_states);

  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);

  // We can tune `log_power` as a tradeoff between work done and clock time on
  // GPU.
  int32_t log_power = (c->GetDeviceType() == kCpu ? 0 : 4);

  int32_t max_num_states = fsas.shape.MaxSize(1);
  // the following avoids doing too much extra work accumulating powers
  // of 'dest_states' for very small problem sizes.
  while (log_power > 0 && (1 << (1 + log_power)) > max_num_states) log_power--;

  // Ignoring edge effects: `dest_states_powers[0]` is just an array indexed by
  // state_idx01, that gives us the dest_state_idx01 that would be the beginning
  // of the next batch if state_idx01 were the beginning of the current batch.
  // So if we follow this chain forward from the start of one of the FSAs until
  // it passes the end of this FSA, we get the beginnings of the batches
  // we want.  The natural algorithm to find the beginnings of the batches
  // is sequential.
  Array2<int32_t> dest_states_powers(c, log_power + 1, num_states);
  const int32_t *arc_dest_states_data = arc_dest_states.Data(),
                *fsas_row_splits2_data = fsas.RowSplits(2).Data();
  int32_t *dest_states_power_data =
      dest_states_powers.Data();  // only process Row[0] below
  const int32_t int_max = std::numeric_limits<int32_t>::max();
  K2_EVAL(
      c, num_states, lambda_set_dest_states, (int32_t state_idx01)->void {
        int32_t arc_idx01x = fsas_row_splits2_data[state_idx01];
        // If this state has arcs, let its `dest_state` be the smallest
        // `dest_state` of any of its arcs (which is the first element of those
        // arcs' dest states in `arc_dest_states_data`); otherwise, take the
        // `dest_state` from the 1st arc of the next state, which is the largest
        // value we can take (if the definition is: the highest-numbered state s
        // for which neither this state nor any later-numbered state has an arc
        // to a state lower than s).

        // if this state has arcs,
        //    arc_idx01x is the first arc index of this state, we get the
        //    smallest dest state of this state's arcs using
        //    arc_dest_states_data[arc_idx01x]
        // else
        //    arc_idx01x is the first arc index of the next state, then
        //    arc_dest_states_data[arc_idx01x] is the largest value we can take,
        //    which is also the smallest dest state in the next state.
        int32_t dest_state =
            (arc_idx01x < num_arcs ? arc_dest_states_data[arc_idx01x]
                                   : int_max);
        dest_states_power_data[state_idx01] = dest_state;
        // if the following fails, it's either a code error or the input FSA had
        // cycles.
        K2_CHECK_GT(dest_state, state_idx01);
      });

  // `num_batches_per_fsa` will be set to the number of batches of states that
  // we'll use for each FSA... it corresponds to the number of times we have
  // to follow links forward in the dest_states array till we pass the
  // end of the array for this fSA.
  Array1<int32_t> num_batches_per_fsa(c, num_fsas + 1, 0);

  // `batch_starts` will contain the locations of the first state_idx01 for each
  // batch, but in an 'un-consolidated' format.  Specifically, for FSA with
  // index i, the batch_starts for that FSA begin at element fsa.RowSplits(1)[i]
  // of `batch_starts`.  This is just a convenient layout because we know there
  // can't be more batches than there are states.  We'll later consolidate the
  // information into a single array.
  Array1<int32_t> batch_starts(c, num_states + 1);

  int32_t *num_batches_per_fsa_data = num_batches_per_fsa.Data(),
          *batch_starts_data = batch_starts.Data();
  const int32_t *fsas_row_splits1_data = fsas.RowSplits(1).Data();

#if 0
  // This is a simple version of the kernel that demonstrates what we're trying
  // to do with the more complex code.
  K2_EVAL(
      c, num_fsas, lambda_set_batch_info_simple, (int32_t fsa_idx) {
        int32_t begin_state_idx01 = fsas_row_splits1_data[fsa_idx],
                end_state_idx01 = fsas_row_splits1_data[fsa_idx + 1];
        int32_t i = 0, cur_state_idx01 = begin_state_idx01;
        while (cur_state_idx01 < end_state_idx01) {
          batch_starts_data[begin_state_idx01 + i] = cur_state_idx01;
          cur_state_idx01 = dest_states_power_data[cur_state_idx01];
          ++i;
        }
        num_batches_per_fsa_data[fsa_idx] = i;
      });
#else
  int32_t stride = dest_states_powers.ElemStride0();
  for (int32_t power = 1; power <= log_power; power++) {
    const int32_t *src_data = dest_states_powers.Data() + (power - 1) * stride;
    int32_t *dest_data = dest_states_powers.Data() + power * stride;
    K2_EVAL(
        c, num_states, lambda_square_array, (int32_t state_idx01)->void {
          int32_t dest_state = src_data[state_idx01],
                  dest_state_sq =
                      (dest_state < num_states ? src_data[dest_state]
                                               : int_max);
          dest_data[state_idx01] = dest_state_sq;
        });
  }
  // jobs_per_fsa tells us how many separate chains of states we'll follow for
  // each FSA.
  // jobs_multiple is a kind of trick to ensure any given warp doesn't
  // issue more memory requests than it can handle at a time (we drop
  // some threads).
  int32_t jobs_per_fsa = (1 << log_power),
          jobs_multiple = (c->GetDeviceType() == kCuda ? 8 : 1);
  while (jobs_multiple > 1 && jobs_per_fsa * jobs_multiple * num_fsas > 10000)
    jobs_multiple /= 2;  // Likely won't get here.  Just reduce multiple if
                         // num-jobs is ridiculous.

  auto dest_states_powers_acc = dest_states_powers.Accessor();
  K2_EVAL2(
      c, num_fsas, jobs_per_fsa * jobs_multiple, lambda_set_batch_info,
      (int32_t fsa_idx, int32_t j) {
        if (j % jobs_multiple != 0)
          return;  // a trick to avoid too much random
                   // memory access for any given warp
        int32_t task_idx =
            j / jobs_multiple;  // Now 0 <= task_idx < jobs_per_fsa.

        // The task indexed `task_idx` is responsible for batches numbered
        // task_idx, task_idx + jobs_per_fsa, task_index + 2 * job_per_fsa and
        // so on, for the FSA numbered `fsa_idx`. Comparing this code to
        // `lambda_set_batch_info_simple`, this task is responsible for the
        // assignment to batch_starts_data for all i such that i % jobs_per_fsas
        // == task_idx, together with the assignment to
        // num_batchess_per_fsa_data if
        //  i % jobs_per_fsas == task_idx (here referring to the i value finally
        // assigned to that location).

        int32_t begin_state_idx01 = fsas_row_splits1_data[fsa_idx],
                end_state_idx01 = fsas_row_splits1_data[fsa_idx + 1];
        int32_t num_states_this_fsa = end_state_idx01 - begin_state_idx01;
        int32_t i = 0, cur_state_idx01 = begin_state_idx01;

        if (task_idx >= num_states_this_fsa) return;

        // The next loop advances `cur_state_idx01` by
        // a number of steps equal to `task_idx`.
        for (int32_t m = 0; m < log_power; ++m) {
          int32_t n = 1 << m;
          if ((task_idx & n) != 0) {
            i += n;
            int32_t next = dest_states_powers_acc(m, cur_state_idx01);
            if (next >= end_state_idx01) return;
            cur_state_idx01 = next;
          }
        }
        K2_CHECK_EQ(i, task_idx);

        while (1) {
          if (i >= num_states_this_fsa) return;
          batch_starts_data[begin_state_idx01 + i] = cur_state_idx01;
          int32_t next_state_idx01 = dest_states_powers_acc(
              log_power,
              cur_state_idx01);  // advance jobs_per_fsa = (1 << log_power)
                                 // steps
          if (next_state_idx01 >= end_state_idx01) {
            // if exactly one step would also be enough to take us past the
            // boundary...
            if (dest_states_powers_acc(0, cur_state_idx01) >= end_state_idx01) {
              num_batches_per_fsa_data[fsa_idx] = i + 1;
            }
            return;
          } else {
            i += jobs_per_fsa;
            cur_state_idx01 = next_state_idx01;
          }
        }
      });
#endif
  ExclusiveSum(num_batches_per_fsa, &num_batches_per_fsa);
  Array1<int32_t> &ans_row_splits1 = num_batches_per_fsa;
  int32_t num_batches = num_batches_per_fsa[num_fsas];
  Array1<int32_t> ans_row_ids1(c, num_batches);
  RowSplitsToRowIds(ans_row_splits1, &ans_row_ids1);
  Array1<int32_t> ans_row_splits2(c, num_batches + 1);
  const int32_t *ans_row_splits1_data = ans_row_splits1.Data(),
                *ans_row_ids1_data = ans_row_ids1.Data();
  int32_t *ans_row_splits2_data = ans_row_splits2.Data();
  ans_row_splits2.Range(num_batches, 1) = num_states;  // The kernel below won't
                                                       // set this last element
  K2_EVAL(
      c, num_batches, lambda_set_ans_row_splits2, (int32_t idx01)->void {
        int32_t idx0 = ans_row_ids1_data[idx01],  // Fsa index
            idx0x = ans_row_splits1_data[idx0], idx1 = idx01 - idx0x,
                fsas_idx0x =
                    fsas_row_splits1_data[idx0];  // 1st state-idx (idx01)
                                                  // in fsas_, for this FSA

        int32_t fsas_idx01 =
            fsas_idx0x + idx1;  // the idx1 is actually the
                                // batch-index, this statement
                                // reflects the 'un-consolidated'
                                // format of `batch_starts`.

        int32_t this_batch_start = batch_starts_data[fsas_idx01];
        ans_row_splits2_data[idx01] = this_batch_start;
      });

  RaggedShape ans_shape =
      RaggedShape3(&ans_row_splits1, &ans_row_ids1, num_batches,
                   &ans_row_splits2, nullptr, num_states);
  Array1<int32_t> ans_value = Range(c, num_states, 0);
  if (transpose) {
    ans_shape = MakeTransposable(ans_shape);
    Ragged<int32_t> ans(ans_shape, ans_value);
    return Transpose(ans);
  } else {
    return Ragged<int32_t>(ans_shape, ans_value);
  }
}

Ragged<int32_t> GetIncomingArcs(FsaVec &fsas,
                                const Array1<int32_t> &dest_states) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK(IsCompatible(fsas, dest_states));
  ContextPtr &c = fsas.Context();
  Ragged<int32_t> dest_states_tensor(fsas.shape, dest_states);
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);

  Array1<int32_t> incoming_arcs_order =
                      GetTransposeReordering(dest_states_tensor, num_states),

                  ans_row_ids2 = dest_states[incoming_arcs_order];
  // Note: incoming_arcs_row_ids2 will be monotonically increasing

  Array1<int32_t> ans_row_splits2(c, num_states + 1);
  RowIdsToRowSplits(ans_row_ids2, &ans_row_splits2);

  // Axis 1 corresponds to FSA states, so the row-ids and row-splits for axis
  // 1 are the same as for `fsas`.
  Array1<int32_t> ans_row_ids1 = fsas.RowIds(1),
                  ans_row_splits1 = fsas.RowSplits(1);
  return Ragged<int32_t>(
      RaggedShape3(&ans_row_splits1, &ans_row_ids1, num_states,
                   &ans_row_splits2, &ans_row_ids2, num_arcs),
      incoming_arcs_order);
}

Ragged<int32_t> GetLeavingArcIndexBatches(FsaVec &fsas,
                                          Ragged<int32_t> &state_batches) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  K2_DCHECK(state_batches.TotSize(1) == num_fsas * num_batches);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);

  // get ans_shape
  Array1<int32_t> ans_row_splits3(c, num_states + 1);
  int32_t *ans_row_splits3_data = ans_row_splits3.Data();
  const int32_t *fsa_states_row_splits_data = fsas.RowSplits(2).Data();
  const int32_t *batch_states_data = state_batches.values.Data();
  K2_EVAL(
      c, num_states, lambda_set_ans_row_splits3, (int32_t idx) {
        int32_t state_idx = batch_states_data[idx];
        ans_row_splits3_data[idx] = fsa_states_row_splits_data[state_idx + 1] -
                                    fsa_states_row_splits_data[state_idx];
      });
  ExclusiveSum(ans_row_splits3, &ans_row_splits3);
  Array1<int32_t> ans_row_ids3(c, num_arcs);
  RowSplitsToRowIds(ans_row_splits3, &ans_row_ids3);
  RaggedShape ans_shape = ComposeRaggedShapes(
      state_batches.shape,
      RaggedShape2(&ans_row_splits3, &ans_row_ids3, num_arcs));

  // get ans_values
  Array1<int32_t> ans_values(c, num_arcs);
  int32_t *ans_values_data = ans_values.Data();
  const int32_t *ans_row_ids3_data = ans_row_ids3.Data();
  K2_EVAL(
      c, num_arcs, lambda_set_ans_values, (int32_t idx0123) {
        int32_t ans_idx012 = ans_row_ids3_data[idx0123];
        int32_t state_idx =
            batch_states_data[ans_idx012];  // state_idx is idx01 in fsas
        int32_t fsa_idx01x = fsa_states_row_splits_data[state_idx];
        // ans_idx3 is fsas_idx2, i.e. the arc idx in a state
        int32_t ans_idx3 = idx0123 - ans_row_splits3_data[ans_idx012];
        ans_values_data[idx0123] = fsa_idx01x + ans_idx3;
      });

  return Ragged<int32_t>(ans_shape, ans_values);
}

Ragged<int32_t> GetEnteringArcIndexBatches(FsaVec &fsas,
                                           Ragged<int32_t> &incoming_arcs,
                                           Ragged<int32_t> &state_batches) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK(IsCompatible(fsas, incoming_arcs));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(incoming_arcs.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  // just using DCHECK below to save time in production code
  K2_DCHECK(state_batches.TotSize(1) == num_fsas * num_batches);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(incoming_arcs.Dim0(), num_fsas);
  K2_DCHECK_EQ(incoming_arcs.TotSize(1), num_states);
  K2_DCHECK_EQ(incoming_arcs.NumElements(), num_arcs);

  // get ans_shape
  Array1<int32_t> ans_row_splits3(c, num_states + 1);
  int32_t *ans_row_splits3_data = ans_row_splits3.Data();
  const int32_t *incoming_arcs_row_splits_data =
      incoming_arcs.RowSplits(2).Data();
  const int32_t *batch_states_data = state_batches.values.Data();
  K2_EVAL(
      c, num_states, lambda_set_ans_row_splits3, (int32_t idx) {
        int32_t state_idx = batch_states_data[idx];
        ans_row_splits3_data[idx] =
            incoming_arcs_row_splits_data[state_idx + 1] -
            incoming_arcs_row_splits_data[state_idx];
      });
  ExclusiveSum(ans_row_splits3, &ans_row_splits3);
  Array1<int32_t> ans_row_ids3(c, num_arcs);
  RowSplitsToRowIds(ans_row_splits3, &ans_row_ids3);
  RaggedShape ans_shape = ComposeRaggedShapes(
      state_batches.shape,
      RaggedShape2(&ans_row_splits3, &ans_row_ids3, num_arcs));

  // get ans_values
  Array1<int32_t> ans_values(c, num_arcs);
  int32_t *ans_values_data = ans_values.Data();
  const int32_t *ans_row_ids3_data = ans_row_ids3.Data();
  const int32_t *incoming_arcs_data = incoming_arcs.values.Data();
  K2_EVAL(
      c, num_arcs, lambda_set_ans_values, (int32_t idx0123) {
        int32_t ans_idx012 = ans_row_ids3_data[idx0123];
        int32_t state_idx =
            batch_states_data[ans_idx012];  // state_idx is idx01 in
                                            // incoming_arcs
        int32_t incoming_arcs_idx01x = incoming_arcs_row_splits_data[state_idx];
        // ans_idx3 is incoming_arcs_idx2, i.e. the entering arc idx for a state
        int32_t ans_idx3 = idx0123 - ans_row_splits3_data[ans_idx012];
        int32_t incoming_arcs_idx012 = incoming_arcs_idx01x + ans_idx3;
        ans_values_data[idx0123] = incoming_arcs_data[incoming_arcs_idx012];
      });

  return Ragged<int32_t>(ans_shape, ans_values);
}

FsaVec ConvertDenseToFsaVec(DenseFsaVec &src) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.shape.Context();
  // caution: 'num_symbols' is the number of symbols excluding the final-symbol
  // -1.
  int32_t num_fsas = src.shape.Dim0(), num_symbols = src.scores.Dim1() - 1;
  // the "1" is the extra state per FSA we need in the FsaVec format,
  // for the final-state.
  RaggedShape fsa2state = ChangeSublistSize(src.shape, 1);
  // again, the "+num_fsas" below is the extra state per FSA we need in the
  // FsaVec format, for the final-state.
  int32_t num_states = src.shape.NumElements() + num_fsas;
  // The explanation num-arcs below is as follows:
  // Firstly, all rows of src.scores (==all elements of src.shape) correspond
  // to states with arcs leaving them.  Most of them have `num_symbols` arcs,
  // but the final one for each FSA has 1 arc (with symbol -1)
  int32_t num_arcs =
      src.shape.NumElements() * num_symbols - (num_symbols - 1) * num_fsas;
  Array1<int32_t> row_splits2(c, num_states + 1), row_ids2(c, num_arcs);
  const int32_t *row_ids1_data = fsa2state.RowIds(1).Data(),
                *src_row_ids1_data = src.shape.RowIds(1).Data(),
                *src_row_splits1_data = src.shape.RowSplits(1).Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();

  auto scores_acc = src.scores.Accessor();

  int32_t *row_splits2_data = row_splits2.Data(),
          *row_ids2_data = row_ids2.Data();

  // 0 <= s < num_symbols; note, `num_symbols` excludes the final-symbol (-1).
  // note: `src` means: w.r.t. the numbering in the original DenseFsaVec.
  K2_EVAL2(
      c, src.shape.NumElements(), num_symbols, lambda_set_arcs_etc,
      (int32_t src_state_idx01, int32_t s)->void {
        int32_t fsa_idx0 = src_row_ids1_data[src_state_idx01],
                src_state_idx0x = src_row_splits1_data[fsa_idx0],
                state_idx1 = src_state_idx01 - src_state_idx0x,
                src_next_state_idx0x = src_row_splits1_data[fsa_idx0 + 1],
                src_num_states1 = src_next_state_idx0x - src_state_idx0x,
                ans_state_idx01 = src_state_idx01 +
                                  fsa_idx0;  // we add one final-state per FSA..
                                             // "+ fsa_idx0" gives the
                                             // difference from old->new
                                             // numbering.

        // arc_idx0xx is the 1st arc-index of the FSA we are creating.. each
        // source state has `num_symbols` arcs leaving it except the last one of
        // each FSA, which has 1 arc leaving it (to the final-state).
        int32_t arc_idx0xx = (src_state_idx0x * num_symbols) -
                             fsa_idx0 * (num_symbols - 1),
                arc_idx01x = arc_idx0xx + (state_idx1 * num_symbols),
                arc_idx012 = arc_idx01x + s;
        int32_t symbol_offset;
        if (state_idx1 + 1 == src_num_states1) {
          symbol_offset = -1;
          if (s > 0) return;  // we just need the arc with -1.

          // if this is the state before the final state of this FSA. it has the
          // responsibility to write the row_splits2 value for the final state.
          // It's arc_idx012 + 1; the "+1" corresponds to the single arc with
          // the final-symbol on it.
          row_splits2_data[ans_state_idx01 + 1] = arc_idx012 + 1;
        } else {
          symbol_offset = 0;
        }
        // the "+ 1" is because index 0 in `scores` is for the final-symbol -1,
        // then 0, 1, etc.
        int32_t symbol_index_in_scores = s + symbol_offset + 1;
        arcs_data[arc_idx012] =
            Arc(state_idx1, state_idx1 + 1, s + symbol_offset,
                scores_acc(src_state_idx01, symbol_index_in_scores));
        row_ids2_data[arc_idx012] = ans_state_idx01;
        if (s == 0) {  // 1st arc for this state.
          row_splits2_data[ans_state_idx01] = arc_idx012;
          K2_CHECK(row_ids1_data[ans_state_idx01] == fsa_idx0);
          if (src_state_idx01 == 0) row_splits2_data[num_states] = num_arcs;
        }
      });

  RaggedShape state2arc = RaggedShape2(&row_splits2, &row_ids2, num_arcs);
  return Ragged<Arc>(ComposeRaggedShapes(fsa2state, state2arc), arcs);
}

template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, Ragged<int32_t> &state_batches,
                                   Ragged<int32_t> &entering_arc_batches,
                                   bool log_semiring,
                                   Array1<int32_t> *entering_arcs) {
  NVTX_RANGE(K2_FUNC);
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK(IsCompatible(fsas, entering_arc_batches));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  K2_CHECK_EQ(entering_arc_batches.NumAxes(), 4);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  // just using DCHECK below to save time in production code
  K2_DCHECK_EQ(state_batches.TotSize(1), num_fsas * num_batches);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(entering_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(entering_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(entering_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(entering_arc_batches.NumElements(), num_arcs);

  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> state_scores(c, num_states, negative_infinity);
  FloatType *state_scores_data = state_scores.Data();
  // set the score of start state in each fsa to be 0
  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  K2_EVAL(
      c, num_fsas, lambda_set_start_state_score, (int32_t fsa_idx) {
        int32_t start_state = fsa_row_splits1[fsa_idx],
                start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
        if (start_state_next_fsa - start_state > 0)
          state_scores_data[start_state] = 0;
      });

  // get the 1st entering arc index in each batch, +1 so we can get the number
  // of entering arcs in each batch by taking the difference of adjacent
  // elements
  Array1<int32_t> entering_arc_start_index(c, num_batches + 1);
  int32_t *entering_arc_start_index_data = entering_arc_start_index.Data();
  const int32_t *arc_batches_row_splits1 =
      entering_arc_batches.RowSplits(1).Data();
  const int32_t *arc_batches_row_splits2 =
      entering_arc_batches.RowSplits(2).Data();
  const int32_t *arc_batches_row_splits3 =
      entering_arc_batches.RowSplits(3).Data();
  K2_EVAL(
      c, num_batches, lambda_set_entering_arc_start_index, (int32_t batch_idx) {
        int32_t this_state_idx0xx =
            arc_batches_row_splits2[batch_idx * num_fsas];
        int32_t this_arc_idx0xxx = arc_batches_row_splits3[this_state_idx0xx];
        entering_arc_start_index_data[batch_idx] = this_arc_idx0xxx;
        if (batch_idx == num_batches - 1) {
          // process the last element
          int32_t next_state_idx0xx =
              arc_batches_row_splits2[num_batches * num_fsas];
          int32_t next_arc_idx0xxx = arc_batches_row_splits3[next_state_idx0xx];
          entering_arc_start_index_data[num_batches] = next_arc_idx0xxx;
        }
      });

  const int32_t *arc_batches_row_ids1 = entering_arc_batches.RowIds(1).Data();
  const int32_t *arc_batches_row_ids2 = entering_arc_batches.RowIds(2).Data();
  const int32_t *arc_batches_row_ids3 = entering_arc_batches.RowIds(3).Data();
  const int32_t *entering_arc_ids = entering_arc_batches.values.Data();
  const int32_t *states_data = state_batches.values.Data();
  const Arc *arcs = fsas.values.Data();
  Array1<FloatType> entering_arc_score_values(
      c, num_arcs);  // entering arc_scores in batches
  FloatType *arc_scores_data = entering_arc_score_values.Data();
  // copy entering_arc_start_index to cpu as we will access its elements in
  // below Eval function for `lambda_set_entering_arc_scores`
  Array1<int32_t> cpu_entering_arc_start_index =
      entering_arc_start_index.To(GetCpuContext());
  const int32_t *cpu_entering_arc_start = cpu_entering_arc_start_index.Data();
  // copy the index of start state in each fsa to CPU
  Array1<int32_t> &arc_batches_row_splits1_array =
      entering_arc_batches.RowSplits(1);
  Array1<int32_t> arc_batches_row_splits12_cpu =
      entering_arc_batches.RowSplits(2)[arc_batches_row_splits1_array].To(
          GetCpuContext());
  K2_CHECK_EQ(arc_batches_row_splits12_cpu.Dim(), num_batches + 1);
  const int32_t *arc_batches_row_splits12_cpu_data =
      arc_batches_row_splits12_cpu.Data();
  Array1<int32_t> arc_row_splits_mem(c, num_states + 1);
  Array1<FloatType> score_cache(c, num_states + 1);

  int32_t *entering_arcs_data = nullptr;
  if (entering_arcs) {
    K2_CHECK_EQ(log_semiring, false) << " entering_arcs supplied";
    *entering_arcs = Array1<int32_t>(c, num_states, -1);
    entering_arcs_data = entering_arcs->Data();
  }

  // process batch sequentially.
  for (int32_t i = 0; i < num_batches; ++i) {
    // get the range we would call Max/LogSum per sub list
    int32_t this_state_idx0xx = arc_batches_row_splits12_cpu_data[i],
            next_state_idx0xx = arc_batches_row_splits12_cpu_data[i + 1];
    K2_CHECK_LT(this_state_idx0xx, num_states);
    K2_CHECK_LE(next_state_idx0xx, num_states);
    int32_t num_states_this_batch = next_state_idx0xx - this_state_idx0xx;
    K2_CHECK_LT(num_states_this_batch, arc_row_splits_mem.Dim());
    // we always use the first `num_states_this_batch` elements in
    // arc_row_splits_mem.
    Array1<int32_t> arc_row_splits_part = arc_row_splits_mem.Range(
        0, num_states_this_batch + 1);  // +1 for the last element
    int32_t num_arcs_this_batch =
        cpu_entering_arc_start[i + 1] - cpu_entering_arc_start[i];
    {
      ParallelRunner pr(c);
      // get entering arc scores
      {
        With w(pr.NewStream());
        K2_EVAL(
            c, num_arcs_this_batch, lambda_set_entering_arc_score,
            (int32_t idx123) {
              // all idx** in below code are the indexes to entering_arc_batches
              int32_t idx0123 = entering_arc_start_index_data[i] + idx123;
              int32_t idx012 = arc_batches_row_ids3[idx0123];
              int32_t idx01 = arc_batches_row_ids2[idx012];
              K2_CHECK_EQ(idx01 / num_fsas, i);  // idx01/num_fsas is batch_id
              int32_t fsa_id = idx01 % num_fsas;

              int32_t entering_arc_id = entering_arc_ids[idx0123];
              float curr_arc_score = arcs[entering_arc_id].score;
              int32_t src_state_idx1 = arcs[entering_arc_id].src_state;
              int32_t src_state_idx01 =
                  fsa_row_splits1[fsa_id] + src_state_idx1;
              arc_scores_data[idx0123] =
                  state_scores_data[src_state_idx01] + curr_arc_score;
            });
      }
      {
        With w(pr.NewStream());
        // make entering arc row splits info in each batch starting from zero,
        // we will use it to call MaxPerSublist or LogSumPerSubList
        int32_t *sum_splits_data = arc_row_splits_part.Data();
        K2_EVAL(
            c, num_states_this_batch + 1, lambda_set_row_splits_for_sum,
            (int32_t idx) {
              sum_splits_data[idx] =
                  arc_batches_row_splits3[idx + this_state_idx0xx] -
                  arc_batches_row_splits3[this_state_idx0xx];
            });
      }
    }
    int32_t this_arc_idx0xxx = cpu_entering_arc_start[i];
    Array1<FloatType> sub_scores_values =
        entering_arc_score_values.Range(this_arc_idx0xxx, num_arcs_this_batch);
    RaggedShape sub_scores_shape =
        RaggedShape2(&arc_row_splits_part, nullptr, sub_scores_values.Dim());
    Ragged<FloatType> sub_scores(sub_scores_shape, sub_scores_values);
    // we always use the first num_rows elements in score_cache.
    Array1<FloatType> sub_state_scores =
        score_cache.Range(0, num_states_this_batch);
    // get scores per state in this batch
    if (log_semiring) {
      LogSumPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    } else {
      MaxPerSublist(sub_scores, negative_infinity, &sub_state_scores);
      if (entering_arcs_data != nullptr) {
        FloatType *sub_state_scores_data = sub_state_scores.Data(),
                  *sub_scores_data = sub_scores.values.Data();
        int32_t *sub_scores_row_ids_data = sub_scores.RowIds(1).Data();
        const int32_t *sub_state_ids_data = states_data + this_state_idx0xx,
                      *sub_entering_arc_ids_data =
                          entering_arc_ids + this_arc_idx0xxx;
        // arc_idx01 below is an index into sub_scores, it is also an arc_idx123
        // into entering_arc_batches.
        K2_EVAL(
            c, sub_scores.NumElements(), lambda_set_entering_arcs,
            (int32_t arc_idx01) {
              // state_idx0 below is idx0 into `sub_scores`, also an index into
              // `sub_scores`.
              int32_t state_idx0 = sub_scores_row_ids_data[arc_idx01];
              if (sub_scores_data[arc_idx01] ==
                  sub_state_scores_data[state_idx0]) {
                int32_t fsas_state_idx01 = sub_state_ids_data[state_idx0],
                        fsas_entering_arc_idx012 =
                            sub_entering_arc_ids_data[arc_idx01];
                // The following statement has a race condition if there is a
                // tie on scores, but this is OK and by design.  It makes the
                // choice of traceback non-deterministic in these cases.
                entering_arcs_data[fsas_state_idx01] = fsas_entering_arc_idx012;
              }
            });
      }
    }
    const FloatType *sub_state_scores_data = sub_state_scores.Data();
    // Copy those scores to corresponding state in state_scores.
    // `state_idx12` is an idx12 w.r.t. state_batches and entering_arc_batches,
    // but an idx1 w.r.t. sub_scores and an index into the array
    // sub_state_scores.
    K2_EVAL(
        c, num_states_this_batch, lambda_copy_state_scores,
        (int32_t state_idx12) {
          int32_t batches_idx012 = this_state_idx0xx + state_idx12;
          int32_t fsas_state_idx01 = states_data[batches_idx012];
          int32_t batches_idx01 = arc_batches_row_ids2[batches_idx012];
          int32_t fsa_idx0 = batches_idx01 % num_fsas;
          int32_t start_state_idx01 = fsa_row_splits1[fsa_idx0];
          // don't override score 0 in the start state in each fsa.
          if (fsas_state_idx01 != start_state_idx01)
            state_scores_data[fsas_state_idx01] =
                sub_state_scores_data[state_idx12];
        });
  }

  return state_scores;
}

template <typename FloatType>
Array1<FloatType> GetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches,
    const Array1<FloatType> *tot_scores /*= nullptr*/,
    bool log_semiring /*= true*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK(IsCompatible(fsas, leaving_arc_batches));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  K2_CHECK_EQ(leaving_arc_batches.NumAxes(), 4);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  K2_DCHECK(state_batches.TotSize(1) == num_fsas * num_batches);
  // just using DCHECK below to save time in production code
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.NumElements(), num_arcs);

  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> state_scores(c, num_states, negative_infinity);
  FloatType *state_scores_data = state_scores.Data();
  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  if (tot_scores != nullptr) {
    K2_CHECK(IsCompatible(fsas, *tot_scores));
    K2_CHECK_EQ(tot_scores->Dim(), num_fsas);
    const FloatType *tot_scores_data = tot_scores->Data();
    // set the score of final state in fsa i to be negative of tot_scores[i]
    K2_EVAL(
        c, num_fsas, lambda_set_final_state_score, (int32_t fsa_idx) {
          int32_t start_state = fsa_row_splits1[fsa_idx],
                  start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
          if (start_state_next_fsa - start_state > 0) {
            // We never set the score of a state to positive_infinity, otherwise
            // we may get NaN when add it with negative_infinity. But this
            // usually would not happen for a connected FSA.
            if (tot_scores_data[fsa_idx] != negative_infinity) {
              state_scores_data[start_state_next_fsa - 1] =
                  -tot_scores_data[fsa_idx];
            } else {
              state_scores_data[start_state_next_fsa - 1] = negative_infinity;
            }
          }
        });
  } else {
    // set the score of final state in each fsa to be 0
    K2_EVAL(
        c, num_fsas, lambda_set_final_state_score, (int32_t fsa_idx) {
          int32_t start_state = fsa_row_splits1[fsa_idx],
                  start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
          if (start_state_next_fsa - start_state > 0)
            state_scores_data[start_state_next_fsa - 1] = 0;
        });
  }

  // get the 1st leaving arc index in each batch, +1 so we can get the number of
  // leaving arcs in each batch by taking the difference of adjacent elements
  Array1<int32_t> leaving_arc_start_index(c, num_batches + 1);
  int32_t *leaving_arc_start_index_data = leaving_arc_start_index.Data();
  const int32_t *arc_batches_row_splits1 =
      leaving_arc_batches.RowSplits(1).Data();
  const int32_t *arc_batches_row_splits2 =
      leaving_arc_batches.RowSplits(2).Data();
  const int32_t *arc_batches_row_splits3 =
      leaving_arc_batches.RowSplits(3).Data();
  K2_EVAL(
      c, num_batches, lambda_set_leaving_arc_start_index, (int32_t batch_idx) {
        int32_t this_state_idx0xx =
            arc_batches_row_splits2[batch_idx * num_fsas];
        int32_t this_arc_idx0xxx = arc_batches_row_splits3[this_state_idx0xx];
        leaving_arc_start_index_data[batch_idx] = this_arc_idx0xxx;
        if (batch_idx == num_batches - 1) {
          // process the last element
          int32_t next_state_idx0xx =
              arc_batches_row_splits2[num_batches * num_fsas];
          int32_t next_arc_idx0xxx = arc_batches_row_splits3[next_state_idx0xx];
          leaving_arc_start_index_data[num_batches] = next_arc_idx0xxx;
        }
      });

  const int32_t *arc_batches_row_ids1 = leaving_arc_batches.RowIds(1).Data();
  const int32_t *arc_batches_row_ids2 = leaving_arc_batches.RowIds(2).Data();
  const int32_t *arc_batches_row_ids3 = leaving_arc_batches.RowIds(3).Data();
  const int32_t *leaving_arc_ids = leaving_arc_batches.values.Data();
  const int32_t *states_data = state_batches.values.Data();
  const Arc *arcs = fsas.values.Data();
  Array1<FloatType> leaving_arc_score_values(
      c, num_arcs);  // leaving arc_scores in batches
  FloatType *arc_scores_data = leaving_arc_score_values.Data();
  // copy leaving_arc_start_index to cpu as we will access its elements in below
  // Eval function for `lambda_set_leaving_arc_scores`
  Array1<int32_t> cpu_leaving_arc_start_index =
      leaving_arc_start_index.To(GetCpuContext());
  const int32_t *cpu_leaving_arc_start = cpu_leaving_arc_start_index.Data();
  // copy the index of start state in each fsa to CPU
  Array1<int32_t> arc_batches_row_splits1_array =
      leaving_arc_batches.RowSplits(1);
  Array1<int32_t> arc_batches_row_splits12_cpu =
      leaving_arc_batches.RowSplits(2)[arc_batches_row_splits1_array].To(
          GetCpuContext());
  K2_CHECK_EQ(arc_batches_row_splits12_cpu.Dim(), num_batches + 1);
  const int32_t *arc_batches_row_splits12_cpu_data =
      arc_batches_row_splits12_cpu.Data();
  Array1<int32_t> arc_row_splits_mem(c, num_states + 1);
  Array1<FloatType> score_cache(c, num_states + 1);
  // process batch sequentially.
  for (int32_t i = num_batches - 1; i >= 0; --i) {
    // get the range we would call Max/LogSum per sub list
    int32_t this_state_idx0xx = arc_batches_row_splits12_cpu_data[i];
    int32_t next_state_idx0xx =
        arc_batches_row_splits12_cpu_data[i + 1];  // the 1st state idx in the
                                                   // next batch
    K2_CHECK_LT(this_state_idx0xx, num_states);
    K2_CHECK_LE(next_state_idx0xx, num_states);
    int32_t num_states_this_batch = next_state_idx0xx - this_state_idx0xx;
    K2_CHECK_LT(num_states_this_batch, arc_row_splits_mem.Dim());
    // we always use the first `num_states_this_batch` elements in
    // arc_row_splits_mem.
    Array1<int32_t> arc_row_splits_part = arc_row_splits_mem.Range(
        0, num_states_this_batch + 1);  // +1 for the last element
    int32_t num_arcs_this_batch =
        cpu_leaving_arc_start[i + 1] - cpu_leaving_arc_start[i];
    {
      ParallelRunner pr(c);
      // get leaving arc scores
      {
        With w(pr.NewStream());
        K2_EVAL(
            c, num_arcs_this_batch, lambda_set_leaving_arc_score,
            (int32_t idx123) {
              // all idx** in below code are the indexes to leaving_arc_batches
              int32_t idx0123 = leaving_arc_start_index_data[i] + idx123;
              int32_t idx012 = arc_batches_row_ids3[idx0123];
              int32_t idx01 = arc_batches_row_ids2[idx012];
              K2_CHECK_EQ(idx01 / num_fsas, i);  // idx01/num_fsas is batch_id
              int32_t fsa_id = idx01 % num_fsas;

              int32_t leaving_arc_id = leaving_arc_ids[idx0123];
              float curr_arc_score = arcs[leaving_arc_id].score;
              int32_t dest_state_idx1 = arcs[leaving_arc_id].dest_state;
              int32_t dest_state_idx01 =
                  fsa_row_splits1[fsa_id] + dest_state_idx1;
              arc_scores_data[idx0123] =
                  state_scores_data[dest_state_idx01] + curr_arc_score;
            });
      }
      {
        With w(pr.NewStream());
        // make leaving arc row splits info in each batch starting from zero,
        // we will use it to call MaxPerSublist or LogSumPerSubList
        int32_t *sum_splits_data = arc_row_splits_part.Data();
        K2_EVAL(
            c, num_states_this_batch + 1, lambda_set_row_splits_for_sum,
            (int32_t idx) {
              sum_splits_data[idx] =
                  arc_batches_row_splits3[idx + this_state_idx0xx] -
                  arc_batches_row_splits3[this_state_idx0xx];
            });
      }
    }
    int32_t this_arc_idx0xxx = cpu_leaving_arc_start[i];
    Array1<FloatType> sub_scores_values =
        leaving_arc_score_values.Range(this_arc_idx0xxx, num_arcs_this_batch);
    RaggedShape sub_scores_shape =
        RaggedShape2(&arc_row_splits_part, nullptr, sub_scores_values.Dim());
    Ragged<FloatType> sub_scores(sub_scores_shape, sub_scores_values);
    // we always use the first num_rows elements in score_cache.
    Array1<FloatType> sub_state_scores =
        score_cache.Range(0, num_states_this_batch);
    // get scores per state in this batch
    if (log_semiring)
      LogSumPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    else
      MaxPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    const FloatType *sub_state_scores_data = sub_state_scores.Data();
    // copy those scores to corresponding state in state_scores
    K2_EVAL(
        c, num_states_this_batch, lambda_copy_state_scores, (int32_t idx2) {
          int32_t idx012 = this_state_idx0xx + idx2;
          int32_t state_idx012 = states_data[idx012];
          int32_t idx01 = arc_batches_row_ids2[idx012];
          int32_t fsa_id = idx01 % num_fsas;
          int32_t start_state = fsa_row_splits1[fsa_id],
                  start_state_next_fsa = fsa_row_splits1[fsa_id + 1];
          if (start_state_next_fsa - start_state > 0) {  // non-empty fsa
            int32_t final_state_idx = start_state_next_fsa - 1;
            // don't override score in the final state in each fsa.
            if (state_idx012 != final_state_idx)
              state_scores_data[state_idx012] = sub_state_scores_data[idx2];
          }
        });
  }

  return state_scores;
}

template <typename FloatType>
Array1<FloatType> GetTotScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(fsas, forward_scores));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1);
  K2_CHECK_EQ(num_states, forward_scores.Dim());

  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> tot_scores(c, num_fsas, negative_infinity);
  FloatType *tot_scores_data = tot_scores.Data();

  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  K2_EVAL(
      c, num_fsas, lambda_copy_tot_scores, (int32_t fsa_idx) {
        int32_t start_state = fsa_row_splits1[fsa_idx],
                start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
        if (start_state_next_fsa > start_state) {  // non-empty fsa
          int32_t final_state_idx = start_state_next_fsa - 1;
          tot_scores_data[fsa_idx] = forward_scores_data[final_state_idx];
        }
      });

  return tot_scores;
}

template <typename FloatType>
Array1<FloatType> GetArcScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores,
                               const Array1<FloatType> &backward_scores) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(fsas, forward_scores));
  K2_CHECK(IsCompatible(fsas, backward_scores));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  K2_CHECK_EQ(num_states, forward_scores.Dim());
  K2_CHECK_EQ(num_states, backward_scores.Dim());

  Array1<FloatType> arc_scores(c, num_arcs);
  FloatType *arc_scores_data = arc_scores.Data();

  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  const int32_t *fsa_row_ids1 = fsas.RowIds(1).Data();
  const int32_t *fsa_row_ids2 = fsas.RowIds(2).Data();
  const Arc *arcs = fsas.values.Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  const FloatType *backward_scores_data = backward_scores.Data();
  K2_EVAL(
      c, num_arcs, lambda_get_arc_scores, (int32_t arc_idx012) {
        int32_t src_state_idx1 = arcs[arc_idx012].src_state;
        int32_t dest_state_idx1 = arcs[arc_idx012].dest_state;
        float arc_score = arcs[arc_idx012].score;

        int32_t idx01 = fsa_row_ids2[arc_idx012];
        int32_t idx0 = fsa_row_ids1[idx01];
        int32_t idx0x = fsa_row_splits1[idx0];
        int32_t src_state_idx01 = idx0x + src_state_idx1;
        int32_t dest_state_idx01 = idx0x + dest_state_idx1;
        arc_scores_data[arc_idx012] = arc_score +
                                      forward_scores_data[src_state_idx01] +
                                      backward_scores_data[dest_state_idx01];
      });

  return arc_scores;
}

// explicit instantiation for those score computation functions above
template Array1<float> GetForwardScores(FsaVec &fsas,
                                        Ragged<int32_t> &state_batches,
                                        Ragged<int32_t> &entering_arc_batches,
                                        bool log_semiring,
                                        Array1<int32_t> *entering_arcs);
template Array1<double> GetForwardScores(FsaVec &fsas,
                                         Ragged<int32_t> &state_batches,
                                         Ragged<int32_t> &entering_arc_batches,
                                         bool log_semiring,
                                         Array1<int32_t> *entering_arcs);

template Array1<float> GetBackwardScores(FsaVec &fsas,
                                         Ragged<int32_t> &state_batches,
                                         Ragged<int32_t> &leaving_arc_batches,
                                         const Array1<float> *tot_scores,
                                         bool log_semiring);
template Array1<double> GetBackwardScores(FsaVec &fsas,
                                          Ragged<int32_t> &state_batches,
                                          Ragged<int32_t> &leaving_arc_batches,
                                          const Array1<double> *tot_scores,
                                          bool log_semiring);

template Array1<float> GetArcScores(FsaVec &fsas,
                                    const Array1<float> &forward_scores,
                                    const Array1<float> &backward_scores);
template Array1<double> GetArcScores(FsaVec &fsas,
                                     const Array1<double> &forward_scores,
                                     const Array1<double> &backward_scores);

template Array1<float> GetTotScores(FsaVec &fsas,
                                    const Array1<float> &forward_scores);
template Array1<double> GetTotScores(FsaVec &fsas,
                                     const Array1<double> &forward_scores);

Fsa RandomFsa(bool acyclic /*=true*/, int32_t max_symbol /*=50*/,
              int32_t min_num_arcs /*=0*/, int32_t max_num_arcs /*=1000*/) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetCpuContext();
  K2_CHECK_GE(min_num_arcs, 0);
  K2_CHECK_GE(max_num_arcs, min_num_arcs);
  K2_CHECK_GE(max_symbol, 0);
  RaggedShape shape =
      RandomRaggedShape(false, 2, 2, min_num_arcs, max_num_arcs);
  int32_t dim0 = shape.Dim0();
  // empty Fsa
  if (dim0 == 0) return Fsa(shape, Array1<Arc>(c, std::vector<Arc>{}));
  // as there should be no arcs leaving the final_state, we always push back an
  // empty row here.
  Array1<int32_t> ans_row_splits1(c, dim0 + 2);
  Array1<int32_t> sub_range = ans_row_splits1.Range(0, dim0 + 1);
  sub_range.CopyFrom(shape.RowSplits(1));
  int32_t *ans_row_splits1_data = ans_row_splits1.Data();
  ans_row_splits1_data[dim0 + 1] = ans_row_splits1_data[dim0];
  // create returned shape
  RaggedShapeLayer ans_shape_dim;
  ans_shape_dim.row_splits = ans_row_splits1;
  ans_shape_dim.cached_tot_size = shape.TotSize(1);
  RaggedShape ans_shape(std::vector<RaggedShapeLayer>{ans_shape_dim}, true);
  ans_shape.Populate();

  // will be used to generate scores on arcs.
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO(haowen): let the users set the range of scores? it's fine to use it
  // for now as we just use it to test.
  std::uniform_real_distribution<float> dis_score(0, 10);

  // create arcs
  int32_t *row_ids1 = ans_shape.RowIds(1).Data();
  int32_t num_states = ans_shape.Dim0(), num_arcs = ans_shape.TotSize(1);
  int32_t start_state = 0, final_state = num_states - 1;
  std::vector<Arc> arcs(num_arcs);
  for (int32_t i = 0; i != num_arcs; ++i) {
    int32_t curr_state = row_ids1[i];
    int32_t dest_state = acyclic ? RandInt(curr_state + 1, final_state)
                                 : RandInt(start_state, final_state);
    int32_t symbol = dest_state == final_state ? -1 : RandInt(0, max_symbol);
    float score = dis_score(gen);
    arcs[i] = Arc(curr_state, dest_state, symbol, score);
  }
  return Fsa(ans_shape, Array1<Arc>(c, arcs));
}

FsaVec RandomFsaVec(int32_t min_num_fsas /*=1*/, int32_t max_num_fsas /*=1000*/,
                    bool acyclic /*=true*/, int32_t max_symbol /*=50*/,
                    int32_t min_num_arcs /*=0*/,
                    int32_t max_num_arcs /*=1000*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_GE(min_num_fsas, 0);
  K2_CHECK_GE(max_num_fsas, min_num_fsas);
  int32_t num_fsas = RandInt(min_num_fsas, max_num_fsas);
  std::vector<Fsa> fsas(num_fsas);
  for (int32_t i = 0; i != num_fsas; ++i) {
    fsas[i] = RandomFsa(acyclic, max_symbol, min_num_arcs, max_num_arcs);
  }
  return Stack(0, num_fsas, fsas.data());
}

DenseFsaVec RandomDenseFsaVec(int32_t min_num_fsas, int32_t max_num_fsas,
                              int32_t min_frames, int32_t max_frames,
                              int32_t min_symbols, int32_t max_symbols,
                              float scores_scale) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetCpuContext();
  int32_t num_fsas = RandInt(min_num_fsas, max_num_fsas);

  // num_symbols includes epsilon but not final-symbol -1.
  int32_t num_symbols = RandInt(min_symbols, max_symbols);

  // `num_frames` includes the extra 1 frame for the final-symbol.
  std::vector<int32_t> num_frames(num_fsas + 1);
  int32_t tot_frames = 0;
  for (int32_t i = 0; i < num_fsas; i++) {
    num_frames[i] = RandInt(min_frames, max_frames) + 1;
    tot_frames += num_frames[i];
  }

  Array2<float> scores(c, tot_frames, num_symbols + 1);
  auto scores_acc = scores.Accessor();

  std::vector<int32_t> row_splits_vec(num_fsas + 1);
  row_splits_vec[0] = 0;
  int32_t cur_start_frame = 0;
  RandIntGenerator gen;
  for (int32_t i = 0; i < num_fsas; i++) {
    int32_t this_num_frames = num_frames[i],
            end_frame = cur_start_frame + this_num_frames;
    for (int32_t f = cur_start_frame; f + 1 < end_frame; f++) {
      scores_acc(f, 0) = -std::numeric_limits<float>::infinity();
      for (int32_t j = 0; j < num_symbols; j++)
        scores_acc(f, j + 1) = scores_scale * gen(-50, 50) * 0.01;
    }
    // on the last frame the placement of infinity vs. finite is reversed:
    // -1 gets finite value, others get infinity.
    int32_t f = end_frame - 1;
    scores_acc(f, 0) = scores_scale * gen(-50, 50) * 0.01;
    for (int32_t j = 0; j < num_symbols; j++)
      scores_acc(f, j + 1) = -std::numeric_limits<float>::infinity();
    row_splits_vec[i + 1] = cur_start_frame = end_frame;
  }
  Array1<int32_t> row_splits(c, row_splits_vec);
  return DenseFsaVec(RaggedShape2(&row_splits, nullptr, tot_frames), scores);
}

Ragged<int32_t> GetStartStates(FsaVec &src) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr &c = src.Context();
  K2_CHECK_EQ(src.NumAxes(), 3);
  int32_t num_fsas = src.Dim0();
  const int32_t *src_row_splits1_data = src.RowSplits(1).Data();

  Array1<int32_t> ans_row_splits(c, num_fsas + 1);
  // will first set the elements of ans_row_splits to the number of states kept
  // from this FSA (either 0 or 1).
  int32_t *num_states_data = ans_row_splits.Data();
  K2_EVAL(
      c, num_fsas, lambda_set_num_states, (int32_t fsa_idx0)->void {
        // 1 if the FSA is not empty, 0 if empty.
        num_states_data[fsa_idx0] = (src_row_splits1_data[fsa_idx0 + 1] >
                                     src_row_splits1_data[fsa_idx0]);
      });
  ExclusiveSum(ans_row_splits, &ans_row_splits);
  int32_t ans_dim = ans_row_splits.Back();
  Ragged<int32_t> ans(RaggedShape2(&ans_row_splits, nullptr, ans_dim),
                      Array1<int32_t>(c, ans_dim));
  const int32_t *ans_row_ids1_data = ans.shape.RowIds(1).Data();
  int32_t *ans_values_data = ans.values.Data();
  K2_EVAL(
      c, ans_dim, lambda_set_ans_values, (int32_t ans_idx01)->void {
        int32_t idx0 = ans_row_ids1_data[ans_idx01];
        int32_t src_start_state_idx01 = src_row_splits1_data[idx0];
        K2_DCHECK_GT(src_row_splits1_data[idx0 + 1], src_row_splits1_data[idx0]);
        ans_values_data[ans_idx01] = src_start_state_idx01;
      });
  return ans;
}

FsaVec FsaVecFromArcIndexes(FsaVec &fsas, Ragged<int32_t> &best_arc_indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(best_arc_indexes.NumAxes(), 2);
  K2_CHECK(IsCompatible(fsas, best_arc_indexes));
  K2_CHECK_EQ(fsas.Dim0(), best_arc_indexes.Dim0());

  // if there are n arcs (for n > 0), there are n + 1 states; if there are 0
  // arcs, there are 0 states (that FSA will have no arcs or states).
  RaggedShape states_shape = ChangeSublistSizePinned(best_arc_indexes.shape, 1);
  const int32_t *states_shape_row_splits1_data =
      states_shape.RowSplits(1).Data();

  int32_t num_fsas = fsas.Dim0();
  int32_t num_states = states_shape.NumElements();
  int32_t num_arcs = best_arc_indexes.shape.NumElements();
  ContextPtr &context = fsas.Context();

  if (num_arcs == 0) {
    RaggedShape shape_a = RegularRaggedShape(context, num_fsas, 0),
                shape_b = RegularRaggedShape(context, 0, 0);
    return FsaVec(ComposeRaggedShapes(shape_a, shape_b),
                  Array1<Arc>(context, 0));
  }

  Array1<int32_t> row_splits2(context, num_states + 1);
  Array1<int32_t> row_ids2(context, num_arcs);
  int32_t *row_splits2_data = row_splits2.Data();
  int32_t *row_ids2_data = row_ids2.Data();

  Array1<Arc> arcs(context, num_arcs);
  Arc *arcs_data = arcs.Data();

  const int32_t *best_arc_indexes_row_splits1_data =
      best_arc_indexes.RowSplits(1).Data();

  const int32_t *best_arc_indexes_row_ids1_data =
      best_arc_indexes.RowIds(1).Data();

  const int32_t *best_arc_indexes_data = best_arc_indexes.values.Data();
  const Arc *fsas_values_data = fsas.values.Data();

  K2_EVAL(
      context, num_arcs, lambda_set_arcs, (int32_t best_arc_idx01) {
        int32_t fsas_idx0 = best_arc_indexes_row_ids1_data[best_arc_idx01];
        int32_t best_arc_idx0x = best_arc_indexes_row_splits1_data[fsas_idx0];
        int32_t best_arc_idx0x_next =
            best_arc_indexes_row_splits1_data[fsas_idx0 + 1];
        int32_t num_best_arcs = best_arc_idx0x_next - best_arc_idx0x;
        int32_t best_arc_idx1 = best_arc_idx01 - best_arc_idx0x;

        int32_t state_offset = states_shape_row_splits1_data[fsas_idx0];

        const Arc &arc =
            fsas_values_data[best_arc_indexes_data[best_arc_idx01]];
        int32_t src_state = best_arc_idx1;
        int32_t dest_state = src_state + 1;
        int32_t label = arc.label;
        float score = arc.score;
        arcs_data[best_arc_idx01] = Arc(src_state, dest_state, label, score);

        int32_t state_idx01 = state_offset + src_state;
        row_ids2_data[best_arc_idx01] = state_idx01;
        row_splits2_data[state_idx01 + 1] = best_arc_idx01 + 1;
        if (best_arc_idx01 == 0) row_splits2_data[0] = 0;

        if (best_arc_idx1 + 1 == num_best_arcs)
          row_splits2_data[state_idx01 + 2] = best_arc_idx01 + 1;
      });
  RaggedShape shape =
      RaggedShape3(&states_shape.RowSplits(1), &states_shape.RowIds(1),
                   num_states, &row_splits2, &row_ids2, num_arcs);
  Ragged<Arc> ans(shape, arcs);
  return ans;
}

FsaVec GetIncomingFsaVec(FsaVec &fsas) {
  Array1<int32_t> dest_states = GetDestStates(fsas, true);
  Ragged<int32_t> arc_indexes = GetIncomingArcs(fsas, dest_states);
  return FsaVec(arc_indexes.shape, fsas.values[arc_indexes.values]);
}

Ragged<int32_t> ComposeArcMaps(Ragged<int32_t> &step1_arc_map,
                               Ragged<int32_t> &step2_arc_map) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(step1_arc_map.NumAxes(), 2);
  K2_CHECK_EQ(step2_arc_map.NumAxes(), 2);
  return Index(step1_arc_map, step2_arc_map, true);
}

void FixNumStates(FsaVec *fsas) {
  K2_CHECK_EQ(fsas->NumAxes(), 3);
  ContextPtr c = fsas->Context();
  int32_t num_fsas = fsas->Dim0(),
        num_states = fsas->TotSize(1);

  Array1<int32_t> changed(c, 1, 0);
  Renumbering renumber_states(c, num_states);
  renumber_states.Keep() = static_cast<char>(1);  // by default keep all states.

  int32_t *changed_data = changed.Data();
  char *keep_data = renumber_states.Keep().Data();
  const int32_t *row_splits1_data = fsas->RowSplits(1).Data();
  K2_EVAL(c, num_fsas, lambda_set_must_remove, (int32_t i) -> void {
      int32_t num_states = (row_splits1_data[i+1] -
                            row_splits1_data[i]);
      if (num_states == 1)
        keep_data[row_splits1_data[i]] = 0;
      changed_data[0] = 1;
    });
  if (changed[0] == 0)
    return;  // an optimization..
  fsas->shape = RemoveSomeEmptyLists(fsas->shape, 1,
                                     renumber_states);
}


}  // namespace k2
