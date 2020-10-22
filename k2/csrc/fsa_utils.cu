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

#include "k2/csrc/context.h"
#include "k2/csrc/fsa_utils.h"

namespace k2 {

// field separator within a line for a text form FSA
static constexpr const char *kDelim = " \t";

// Convert a string to an integer. Abort the program on failure.
static int32_t StringToInt(const std::string &s) {
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
  K2_CHECK(!s.empty());
  char *p = nullptr;
  float f = std::strtof(s.c_str(), &p);
  if (*p != '\0') K2_LOG(FATAL) << "Failed to convert " << s << " to a float";
  return f;
}

// Trim leading and trailing spaces of a string.
static void TrimString(std::string *s) {
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
   cost/score, -1 as its label and 0 as its aux_label.

   @param [in]  is    The input stream that contains the transducer.

   @return It returns an Fsa on CPU.
*/
static Fsa OpenFstTransducerFromStream(std::istringstream &is,
                                       Array1<int32_t> *aux_labels) {
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
      // TODO(guoguo) We are not sure yet what to put as the auxiliary label for
      //              arcs entering the super final state. The only real choices
      //              are kEpsilon or kFinalSymbol. We are using kEpsilon for
      //              now.
      state_to_aux_labels[original_final_states[i]].push_back(0);  // kEpsilon
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
    os << arc.src_state << sep << arc.dest_state << sep << arc.symbol << sep;
    if (p != nullptr) os << p[i] << sep;
    os << (scale * arc.score) << line_sep;
  }
  os << (fsa.shape.Dim0() - 1) << line_sep;
  return os.str();
}

Array1<int32_t> GetDestStates(FsaVec &fsas, bool as_idx01) {
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_arcs = fsas.NumElements();
  Array1<int32_t> ans(c, num_arcs);
  const Arc *arcs_data = fsas.values.Data();
  int32_t *ans_data = ans.Data();
  if (!as_idx01) {
    auto lambda_set_dest_states1 = [=] __host__ __device__(int32_t arc_idx012) {
      ans_data[arc_idx012] = arcs_data[arc_idx012].dest_state;
    };
    Eval(c, num_arcs, lambda_set_dest_states1);
  } else {
    const int32_t *row_ids2 = fsas.RowIds(2).Data();
    auto lambda_set_dest_states01 = [=] __host__ __device__(
                                        int32_t arc_idx012) {
      int32_t src_state = arcs_data[arc_idx012].src_state,
              dest_state = arcs_data[arc_idx012].dest_state;
      // (row_ids2[arc_idx012] - src_state) is the same as
      // row_splits1[row_ids1[row_ids2[arc_idx012]]]; it's the idx01 of the 1st
      // state in this FSA.
      ans_data[arc_idx012] = dest_state + (row_ids2[arc_idx012] - src_state);
    };
    Eval(c, num_arcs, lambda_set_dest_states01);
  }
  return ans;
}

Ragged<int32_t> GetStateBatches(FsaVec &fsas, bool transpose) {
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
  auto lambda_set_dest_states =
      [=] __host__ __device__(int32_t state_idx01) -> void {
    int32_t arc_idx01x = fsas_row_splits2_data[state_idx01];
    // If this state has arcs, let its `dest_state` be the smallest `dest_state`
    // of any of its arcs (which is the first element of those arcs' dest states
    // in `arc_dest_states_data`); otherwise, take the `dest_state` from the 1st
    // arc of the next state, which is the largest value we can take (if the
    // definition is: the highest-numbered state s for which neither this state
    // nor any later-numbered state has an arc to a state lower than s).

    // if this state has arcs,
    //    arc_idx01x is the first arc index of this state, we get the
    //    smallest dest state of this state's arcs using
    //    arc_dest_states_data[arc_idx01x]
    // else
    //    arc_idx01x is the first arc index of the next state, then
    //    arc_dest_states_data[arc_idx01x] is the largest value we can take,
    //    which is also the smallest dest state in the next state.
    int32_t dest_state =
        (arc_idx01x < num_arcs ? arc_dest_states_data[arc_idx01x] : int_max);
    dest_states_power_data[state_idx01] = dest_state;
    // if the following fails, it's either a code error or the input FSA had
    // cycles.
    K2_CHECK_GT(dest_state, state_idx01);
  };
  Eval(c, num_states, lambda_set_dest_states);

  // `num_batches_per_fsa` will be set to the number of batches of states that
  // we'll use for each FSA... it corresponds to the number of times we have
  // to follow links forward in the dest_states array till we pass the
  // end of the array for this fSA.
  Array1<int32_t> num_batches_per_fsa(c, num_fsas + 1);

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
  auto lambda_set_batch_info_simple = [=] __host__ __device__(int32_t fsa_idx) {
    int32_t begin_state_idx01 = fsas_row_splits1_data[fsa_idx],
            end_state_idx01 = fsas_row_splits1_data[fsa_idx + 1];
    int32_t i = 0, cur_state_idx01 = begin_state_idx01;
    while (cur_state_idx01 < end_state_idx01) {
      batch_starts_data[begin_state_idx01 + i] = cur_state_idx01;
      cur_state_idx01 = dest_states_power_data[cur_state_idx01];
      ++i;
    }
    num_batches_per_fsa_data[fsa_idx] = i;
  };
  Eval(c, num_fsas, lambda_set_batch_info_simple);
#else
  int32_t stride = dest_states_powers.ElemStride0();
  for (int32_t power = 1; power <= log_power; power++) {
    const int32_t *src_data = dest_states_powers.Data() + (power - 1) * stride;
    int32_t *dest_data = dest_states_powers.Data() + power * stride;
    auto lambda_square_array =
        [=] __host__ __device__(int32_t state_idx01) -> void {
      int32_t dest_state = src_data[state_idx01],
              dest_state_sq =
                  (dest_state < num_states ? src_data[dest_state] : int_max);
      dest_data[state_idx01] = dest_state_sq;
    };
    Eval(c, num_states, lambda_square_array);
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
  auto lambda_set_batch_info = [=] __host__ __device__(int32_t fsa_idx,
                                                       int32_t j) {
    if (j % jobs_multiple != 0)
      return;                              // a trick to avoid too much random
                                           // memory access for any given warp
    int32_t task_idx = j / jobs_multiple;  // Now 0 <= task_idx < jobs_per_fsa.

    // The task indexed `task_idx` is responsible for batches numbered
    // task_idx, task_idx + jobs_per_fsa, task_index + 2 * job_per_fsa and so
    // on, for the FSA numbered `fsa_idx`. Comparing this code to
    // `lambda_set_batch_info_simple`, this task is responsible for the
    // assignment to batch_starts_data for all i such that i % jobs_per_fsas ==
    // task_idx, together with the assignment to num_batchess_per_fsa_data if
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
          cur_state_idx01);  // advance jobs_per_fsa = (1 << log_power) steps
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
  };
  Eval2(c, num_fsas, jobs_per_fsa * jobs_multiple, lambda_set_batch_info);
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
  auto lambda_set_ans_row_splits2 =
      [=] __host__ __device__(int32_t idx01) -> void {
    int32_t idx0 = ans_row_ids1_data[idx01],  // Fsa index
        idx0x = ans_row_splits1_data[idx0], idx1 = idx01 - idx0x,
            fsas_idx0x = fsas_row_splits1_data[idx0],  // 1st state-idx (idx01)
                                                       // in fsas_, for this FSA
        fsas_idx01 = fsas_idx0x + idx1,  // the idx1 is actually the
                                         // batch-index, this statement reflects
                                         // the 'un-consolidated' format of
                                         // `batch_starts`.
        this_batch_start = batch_starts_data[fsas_idx01];
    ans_row_splits2_data[idx01] = this_batch_start;
  };
  Eval(c, num_batches, lambda_set_ans_row_splits2);

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
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  K2_DCHECK_EQ((state_batches.TotSize(1) / num_batches), num_fsas);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);

  // get ans_shape
  Array1<int32_t> ans_row_splits3(c, num_states + 1);
  int32_t *ans_row_splits3_data = ans_row_splits3.Data();
  const int32_t *fsa_states_row_splits_data = fsas.RowSplits(2).Data();
  const int32_t *batch_states_data = state_batches.values.Data();
  auto lambda_set_ans_row_splits3 = [=] __host__ __device__(int32_t idx) {
    int32_t state_idx = batch_states_data[idx];
    ans_row_splits3_data[idx] = fsa_states_row_splits_data[state_idx + 1] -
                                fsa_states_row_splits_data[state_idx];
  };
  Eval(c, num_states, lambda_set_ans_row_splits3);
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
  auto lambda_set_ans_values = [=] __host__ __device__(int32_t idx0123) {
    int32_t ans_idx012 = ans_row_ids3_data[idx0123];
    int32_t state_idx =
        batch_states_data[ans_idx012];  // state_idx is idx01 in fsas
    int32_t fsa_idx01x = fsa_states_row_splits_data[state_idx];
    // ans_idx3 is fsas_idx2, i.e. the arc idx in a state
    int32_t ans_idx3 = idx0123 - ans_row_splits3_data[ans_idx012];
    ans_values_data[idx0123] = fsa_idx01x + ans_idx3;
  };
  Eval(c, num_arcs, lambda_set_ans_values);

  return Ragged<int32_t>(ans_shape, ans_values);
}

Ragged<int32_t> GetEnteringArcIndexBatches(FsaVec &fsas,
                                           Ragged<int32_t> &incoming_arcs,
                                           Ragged<int32_t> &state_batches) {
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
  K2_DCHECK_EQ((state_batches.TotSize(1) / num_batches), num_fsas);
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
  auto lambda_set_ans_row_splits3 = [=] __host__ __device__(int32_t idx) {
    int32_t state_idx = batch_states_data[idx];
    ans_row_splits3_data[idx] = incoming_arcs_row_splits_data[state_idx + 1] -
                                incoming_arcs_row_splits_data[state_idx];
  };
  Eval(c, num_states, lambda_set_ans_row_splits3);
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
  auto lambda_set_ans_values = [=] __host__ __device__(int32_t idx0123) {
    int32_t ans_idx012 = ans_row_ids3_data[idx0123];
    int32_t state_idx =
        batch_states_data[ans_idx012];  // state_idx is idx01 in incoming_arcs
    int32_t incoming_arcs_idx01x = incoming_arcs_row_splits_data[state_idx];
    // ans_idx3 is incoming_arcs_idx2, i.e. the entering arc idx for a state
    int32_t ans_idx3 = idx0123 - ans_row_splits3_data[ans_idx012];
    int32_t incoming_arcs_idx012 = incoming_arcs_idx01x + ans_idx3;
    ans_values_data[idx0123] = incoming_arcs_data[incoming_arcs_idx012];
  };
  Eval(c, num_arcs, lambda_set_ans_values);

  return Ragged<int32_t>(ans_shape, ans_values);
}

FsaVec ConvertDenseToFsaVec(DenseFsaVec &src) {
  ContextPtr &c = src.shape.Context();
  // caution: 'num_symbols' is the number of symbols excluding the final-symbol
  // -1.
  int32_t num_fsas = src.shape.Dim0(), num_symbols = src.scores.Dim1() - 1;
  // the "1" is the extra state per FSA we need in the FsaVec format,
  // for the final-state.
  RaggedShape fsa2state = ChangeSublistSize(src.shape, 1);

  int32_t num_states = src.shape.NumElements() + num_fsas,
          num_arcs = src.shape.NumElements() * num_symbols -
                     (num_symbols - 1) * num_fsas;
  Array1<int32_t> row_splits2(c, num_states), row_ids2(c, num_arcs);
  const int32_t *row_ids1_data = fsa2state.RowIds(1).Data(),
                *src_row_ids1_data = src.shape.RowIds(1).Data(),
                *src_row_splits1_data = src.shape.RowSplits(1).Data();
  Array1<Arc> arcs(c, num_arcs);
  Arc *arcs_data = arcs.Data();

  auto scores_acc = src.scores.Accessor();

  // each FSA we return has one extra state (we add the final state).
  int32_t *row_splits2_data = row_splits2.Data(),
          *row_ids2_data = row_ids2.Data();

  // 0 <= s < num_symbols; note, `num_symbols` excludes the final-symbol (-1).
  auto lambda_set_arcs_etc = [=] __host__ __device__(int32_t src_state_idx01,
                                                     int32_t s) -> void {
    int32_t fsa_idx0 = src_row_ids1_data[src_state_idx01],
            src_state_idx0x = src_row_splits1_data[fsa_idx0],
            state_idx1 = src_state_idx01 - src_state_idx0x,
            src_next_state_idx0x = src_row_splits1_data[fsa_idx0 + 1],
            src_num_states1 = src_next_state_idx0x - src_state_idx0x,
            ans_state_idx01 =
                src_state_idx01 + fsa_idx0;  // add final-state per FSA..

    // arc_idx0x is the 1st arc-index of the FSA we are creating.. each source
    // state has `num_symbols` arcs leaving it except the last one of each FSA,
    // which has 1 arc leaving it (to the final-state).
    int32_t arc_idx0xx =
                (src_state_idx0x * num_symbols) - fsa_idx0 * (num_symbols - 1),
            arc_idx01x = arc_idx0xx + (state_idx1 * num_symbols),
            arc_idx012 = arc_idx01x + s;
    int32_t symbol_offset;
    if (state_idx1 + 1 < src_num_states1) {
      symbol_offset = -1;
      if (s > 0) return;  // we just need the arc with -1.
    } else {
      symbol_offset = 0;
    }
    // the "+ 1" is because index 0 in `scores` is for the final-symbol -1,
    // then 0, 1, etc.
    int32_t symbol_index_in_scores = s + symbol_offset + 1;
    arcs_data[arc_idx012] =
        Arc(state_idx1, state_idx1 + 1, s + symbol_offset,
            FloatAsInt(scores_acc(src_state_idx01, symbol_index_in_scores)));
    row_ids2_data[arc_idx012] = ans_state_idx01;
    if (s == 0) {  // 1st arc for this state.
      row_splits2_data[ans_state_idx01] = arc_idx012;
      K2_CHECK(row_ids1_data[ans_state_idx01] == fsa_idx0);
      if (src_state_idx01 == 0) row_splits2_data[num_states] = num_arcs;
    }
  };
  Eval2(c, src.shape.NumElements(), num_symbols, lambda_set_arcs_etc);

  RaggedShape state2arc = RaggedShape2(&row_splits2, &row_ids2, num_arcs);
  return Ragged<Arc>(ComposeRaggedShapes(fsa2state, state2arc), arcs);
}

}  // namespace k2
