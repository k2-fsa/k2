/**
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include <cooperative_groups.h>

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

/* Create an Fsa from a stream, assuming the Fsa is in k2 format:

   src_state dest_state label [aux_label1 aux_label2 ... ] [score]
   ... ...
   final_state

   The source states will be in non-descending order, and the final state does
   not bear a cost/score -- we put the cost/score on the arc that connects to
   the final state and set its label to -1.  The score defaults to 0.0.

   @param [in]  is    The input stream that contains the Fsa.
   @param  [in]  num_aux_labels  The number of auxiliary labels to expect
                     per arc; 0 == acceptor, 1 == transducer, but may be more.
   @param [out] aux_labels_out  If num_aux_labels > 0, this will be
                     assigned to,  with a new array on CPU of shape
                     (num_aux_labels, num_arcs).
   @return It returns an Fsa on CPU.
*/
static Fsa K2FsaFromStream(std::istringstream &is,
                           int32_t num_aux_labels,
                           Array2<int32_t> *aux_labels_out) {
  K2_CHECK(num_aux_labels == 0 || aux_labels_out != nullptr);
  NVTX_RANGE(K2_FUNC);
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::vector<int32_t> aux_labels;
  std::string line;
  int32_t max_state = -1;
  int32_t final_state = -1;

  bool finished = false;  // when the final state is read, set it to true.
  while (std::getline(is, line)) {
    // `splits` is cleared inside the function, so no need to clear it here.
    SplitStringToVector(line, kDelim, &splits);
    if (splits.empty()) continue;  // this is an empty line

    K2_CHECK_EQ(finished, false);

    int32_t num_fields = static_cast<int32_t>(splits.size());
    // The score field of each arc is optional.
    // When num_aux_labels is 0
    //   - num_fields is 3, this means the score field is absent
    //   - num_fields is 4, this means the score field is present
    // When num_aux_labels is > 0
    //   - num_fields is 3 + num_aux_labels, then the score field is absent
    //   - num_fields is 4 + num_aux_labels, then the score field is present
    if (num_fields == 3 + num_aux_labels || num_fields == 4 + num_aux_labels) {
      //   0            1          2      3            3+num_aux_labels
      // src_state  dest_state   label   aux_label1... score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      for (int32_t i = 0; i < num_aux_labels; ++i)
        aux_labels.push_back(StringToInt(splits[3 + i]));
      float score = (num_fields == 4 + num_aux_labels ?
                     StringToFloat(splits[3 + num_aux_labels]) : 0.0f);
      K2_CHECK_GE(src_state, 0);
      K2_CHECK_GE(dest_state, 0);
      arcs.emplace_back(src_state, dest_state, symbol, score);
      max_state = std::max(max_state, std::max(src_state, dest_state));
    } else if (num_fields == 1) {
      if (final_state != -1) {
        K2_LOG(FATAL) << "Invalid line: " << line
                      << ", final state has already been read, value="
                      << final_state;
      }
      //   0
      // final_state
      final_state = StringToInt(splits[0]);
      max_state = std::max(max_state, final_state);
      if (final_state > 0) {
        finished = true;  // set finish
      }
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nk2 FSA with num_aux_labels=" << num_aux_labels
                    << " expects a line with 1 (final_state) or "
                    << (num_aux_labels + 3) << " or " << (num_aux_labels + 4)
                    << " fields.";
    }
  }


  K2_CHECK_EQ(finished || arcs.empty(), true)
      << "If there are arcs, there should be a final state";

  K2_CHECK_EQ(max_state, final_state) << "The final_state id isn't "
                                         "the max of all states";

  auto c = GetCpuContext();

  if (num_aux_labels > 0) {
    *aux_labels_out =
        Array2<int32_t>(c, num_aux_labels, static_cast<int32_t>(arcs.size()));
    K2_CHECK_EQ(aux_labels.size(), arcs.size() * num_aux_labels);
    auto aux_labels_acc = aux_labels_out->Accessor();
    int32_t arcs_size = static_cast<int32_t>(arcs.size());
    for (int32_t i = 0; i < arcs_size; ++i)
      for (int32_t j = 0; j < num_aux_labels; ++j)
        aux_labels_acc(j, i) = aux_labels[i * num_aux_labels + j];
  }

  if (arcs.size() == 0u) {
    return Fsa(EmptyRaggedShape(c, 2));
  }
  bool error = true;
  Array1<Arc> array(c, arcs);
  auto fsa = FsaFromArray1(array, &error, final_state);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create an Fsa from a stream in OpenFst format.  Supports acceptors
   (num_aux_labels=0) and transducers (num_aux_labels=1)
   and also more than one aux_label which is not valid OpenFst format
   but our own extension.

   src_state dest_state label [aux_label1 aux_label2..] [cost]
   ... ...
   final_state [cost]

   We will negate the cost to produce a score when we read it in.  The cost
   defaults to 0.0.

   We always create the super final state. If there are final state(s) in the
   original FST, then we add arc(s) from the original final state(s) to the
   super final state, with the (negated) old final state cost/score as its
   cost/score, -1 as its label and -1 as its aux_label.

   @param [in]  is    The input stream that contains the Fsa.
   @param [in]  num_aux_labels  The number of auxiliary labels to expect
                     per arc; 0 == acceptor, 1 == transducer, but may be more.
   @param [out] aux_labels_out  If num_aux_labels > 0, this will be
                     a new array on CPU of shape (num_aux_labels, num_arcs)
                     will be assigned to this location.


   @return It returns an Fsa on CPU.
*/
static Fsa OpenFstFromStream(std::istringstream &is,
                             int32_t num_aux_labels,
                             Array2<int32_t> *aux_labels_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(num_aux_labels == 0 || aux_labels_out != nullptr);

  std::vector<std::vector<int32_t>> state_to_aux_labels;  // indexed by states
  std::vector<std::vector<Arc>> state_to_arcs;            // indexed by states
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  // We assume the source state of the first line
  // is the start state in OpenFST.
  int32_t start_state = -1;
  int32_t max_state = -1;
  int32_t num_arcs = 0;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  while (std::getline(is, line)) {
    // `splits` is cleared inside the function, so no need to clear it here.
    SplitStringToVector(line, kDelim, &splits);
    if (splits.empty()) continue;  // this is an empty line

    int32_t num_fields = static_cast<int32_t>(splits.size());
    if (num_fields == 3 + num_aux_labels || num_fields == 4 + num_aux_labels) {
      //   0            1          2      3            3+num_aux_labels
      // src_state  dest_state   label   aux_label1... [cost]
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      float cost = (num_fields == 4 + num_aux_labels ?
                    StringToFloat(splits[3 + num_aux_labels]) : 0.0f);
      K2_CHECK_GE(src_state, 0);
      K2_CHECK_GE(dest_state, 0);
      if (start_state == -1) start_state = src_state;
      // Add the arc to "state_to_arcs", and aux_label[s] to
      // "state_to_aux_labels"
      ++num_arcs;
      max_state = std::max(max_state, std::max(src_state, dest_state));
      if (static_cast<int32_t>(state_to_arcs.size()) <= src_state) {
        state_to_arcs.resize(src_state + 1);
        if (num_aux_labels > 0)
          state_to_aux_labels.resize(src_state + 1);
      }
      state_to_arcs[src_state].emplace_back(src_state, dest_state, symbol,
                                            -cost);
      for (int32_t i = 0; i < num_aux_labels; ++i) {
        int32_t aux_label = StringToInt(splits[3 + i]);
        state_to_aux_labels[src_state].push_back(aux_label);
      }
    } else if (num_fields == 1 || num_fields == 2) {
      //   0           1
      // final_state  [cost]
      // There could be multiple final states, so we first have to collect all
      // the final states, and then work out the super final state.
      int32_t original_final_state = StringToInt(splits[0]);
      if (start_state == -1)
        start_state = original_final_state;
      float cost = (num_fields == 2 ?
                    StringToFloat(splits[1]) : 0.0f);
      K2_CHECK_GE(original_final_state, 0);
      max_state = std::max(max_state, original_final_state);
      if (cost != std::numeric_limits<float>::infinity()) {
        original_final_states.push_back(original_final_state);
        original_final_weights.push_back(-cost);
      }
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\n... num-fields=" << num_fields
                    << ", expected 1, 2, " << (3+num_aux_labels)
                    << " or " << (4+num_aux_labels)
                    << " [given that num_aux_labels="
                    << num_aux_labels << "]";
    }
  }

  K2_CHECK(is.eof());

  auto c = GetCpuContext();
  if (start_state == -1) {
    if (num_aux_labels > 0)
      *aux_labels_out = Array2<int32_t>(c, num_aux_labels, 0);
    return Fsa(EmptyRaggedShape(c, 2));
  }

  K2_CHECK_GE(max_state, 0);

  // Post processing on final states. If there are final state(s) in the
  // original FST, we add the super final state as well as arc(s) from original
  // final state(s) to the super final state. Otherwise, the super final state
  // will be added by FsaFromArray1 (since there's no arc with label
  // kFinalSymbol).
  int32_t super_final_state = max_state + 1;
  {
    // Deal with final-states.
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    state_to_arcs.resize(super_final_state);
    if (num_aux_labels > 0)
      state_to_aux_labels.resize(super_final_state);
    for (std::size_t i = 0; i != original_final_states.size(); ++i) {
      state_to_arcs[original_final_states[i]].emplace_back(
          original_final_states[i], super_final_state,
          -1,  // kFinalSymbol
          original_final_weights[i]);
      for (int32_t j = 0; j < num_aux_labels; ++j)
        state_to_aux_labels[original_final_states[i]].push_back(
            -1);  // kFinalSymbol
      ++num_arcs;
    }
  }

  if (start_state != 0) {
    // swap start_state and 0
    std::swap(state_to_arcs[0], state_to_arcs[start_state]);
    if (num_aux_labels > 0)
      std::swap(state_to_aux_labels[0], state_to_aux_labels[start_state]);

    // fix source state
    for (auto &a : state_to_arcs[0]) a.src_state = 0;
    for (auto &a : state_to_arcs[start_state]) a.src_state = start_state;

    // fix dest state
    for (auto &state_arcs : state_to_arcs) {
      for (auto &a : state_arcs) {
        if (a.dest_state == 0) {
          a.dest_state = start_state;
        } else if (a.dest_state == start_state) {
          a.dest_state = 0;
        }
      }
    }
  }

  // Move arcs from "state_to_arcs" to "arcs", and aux_labels from
  // "state_to_aux_labels" to "aux_labels"
  int32_t arc_index = 0;
  arcs.resize(num_arcs);
  Array2<int32_t> aux_labels(c, num_aux_labels, num_arcs);
  auto aux_labels_acc = aux_labels.Accessor();

  if (num_aux_labels > 0)
    K2_CHECK_EQ(state_to_arcs.size(), state_to_aux_labels.size());
  for (std::size_t s = 0; s < state_to_arcs.size(); ++s) {
    if (num_aux_labels > 0) {
      K2_CHECK_EQ(state_to_arcs[s].size() * num_aux_labels,
                  state_to_aux_labels[s].size());
    }
    for (std::size_t a = 0; a < state_to_arcs[s].size(); ++a) {
      K2_CHECK_GT(num_arcs, arc_index);
      arcs[arc_index] = state_to_arcs[s][a];
      for (int32_t i = 0; i < num_aux_labels; ++i)
        aux_labels_acc(i, arc_index) =
            state_to_aux_labels[s][a * num_aux_labels + i];
      ++arc_index;
    }
  }
  K2_CHECK_EQ(arc_index, num_arcs);

  if (num_aux_labels != 0)
    *aux_labels_out = aux_labels;
  Array1<Arc> array(c, arcs);
  bool error = true;
  // FsaFromArray1 will add a super final state if the original FSA
  // doesn't have a final state.
  auto fsa = FsaFromArray1(array, &error, super_final_state);
  K2_CHECK_EQ(error, false);
  return fsa;
}

Fsa FsaFromString(const std::string &s, bool openfst /* = false*/,
                  int32_t num_aux_labels /* = 0*/,
                  Array2<int32_t> *aux_labels /* = nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  std::istringstream is(s);
  K2_CHECK(is);

  if (openfst)
    return OpenFstFromStream(is, num_aux_labels, aux_labels);
  else
    return K2FsaFromStream(is, num_aux_labels, aux_labels);
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

  if (n > 0) {
    os << (fsa.shape.Dim0() - 1) << line_sep;
  } else {
    // Output nothing, the empty string is considered a valid representation
    // for the FSA with no arcs or states.
  }
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
    const int32_t *row_ids2_data = fsas.RowIds(2).Data();
    K2_EVAL(
        c, num_arcs, lambda_set_dest_states01, (int32_t arc_idx012) {
          int32_t src_state = arcs_data[arc_idx012].src_state,
                  dest_state = arcs_data[arc_idx012].dest_state;
          // (row_ids2[arc_idx012] - src_state) is the same as
          // row_splits1[row_ids1[row_ids2[arc_idx012]]]; it's the idx01 of the
          // 1st state in this FSA.
          ans_data[arc_idx012] =
              dest_state + (row_ids2_data[arc_idx012] - src_state);
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

  const int32_t *fsas_row_ids1_data = fsas.RowIds(1).Data(),
                *fsas_row_splits1_data = fsas.RowSplits(1).Data(),
                *fsas_row_ids2_data = fsas.RowIds(2).Data();

  const FloatType negative_infinity =
      -std::numeric_limits<FloatType>::infinity();
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

  const Arc *arcs = fsas.values.Data();

  int32_t *entering_arcs_data = nullptr;
  if (entering_arcs) {
    K2_CHECK_EQ(log_semiring, false) << " entering_arcs supplied";
    *entering_arcs = Array1<int32_t>(c, num_states, -1);
    entering_arcs_data = entering_arcs->Data();
  }

  RaggedAxis0Splitter<int32_t> arc_batches_splitter(entering_arc_batches);

  // process batch sequentially.
  for (int32_t i = 0; i < num_batches; ++i) {
    // entering_arc_batch is indexed [fsa][state_list][arc_list]
    int32_t arc_begin;
    Ragged<int32_t> entering_arc_batch =
        arc_batches_splitter.GetElement(i, &arc_begin);
    const int32_t *entering_arc_batch_data = entering_arc_batch.values.Data();
    int32_t state_begin = arc_batches_splitter.GetOffset(i, 2),
            state_end = arc_batches_splitter.GetOffset(i + 1, 2),
            num_states_this_batch = state_end - state_begin,
            num_arcs_this_batch = entering_arc_batch.NumElements();
    Array1<int32_t> states_batch =
        state_batches.values.Arange(state_begin, state_end);
    const int32_t *states_batch_data = states_batch.Data();

    Ragged<FloatType> entering_arc_batch_scores(entering_arc_batch.shape);
    FloatType *entering_arc_batch_scores_data =
        entering_arc_batch_scores.values.Data();

    // get entering arc scores
    K2_EVAL(
        c, num_arcs_this_batch, lambda_set_entering_arc_score,
        (int32_t idx012)->void {
          // `idx012` is into the batch.
          int32_t fsas_arc_idx012 = entering_arc_batch_data[idx012];
          float curr_arc_score = arcs[fsas_arc_idx012].score;
          int32_t src_state_idx01 = fsas_row_ids2_data[fsas_arc_idx012];
          entering_arc_batch_scores_data[idx012] =
              state_scores_data[src_state_idx01] + curr_arc_score;
        });

    Array1<FloatType> state_batch_scores(c, num_states_this_batch);
    FloatType *state_batch_scores_data = state_batch_scores.Data();

    // get scores per state in this batch
    if (log_semiring) {
      LogSumPerSublist(entering_arc_batch_scores, negative_infinity,
                       &state_batch_scores);
    } else {
      if (entering_arcs_data == nullptr) {
        MaxPerSublist(entering_arc_batch_scores, negative_infinity,
                      &state_batch_scores);
      } else {
        // entering_arc_idxs will contain indexes into
        // `entering_arc_batch_scores`, equiv. to indexes into
        // `entering_arc_batch`.
        Array1<int32_t> entering_arc_idxs(c, num_states_this_batch);
        ArgMaxPerSublist(entering_arc_batch_scores, negative_infinity,
                         &entering_arc_idxs);

        const int32_t *entering_arc_idxs_data = entering_arc_idxs.Data(),
                      *entering_arc_batch_data =
                          entering_arc_batch.values.Data();

        // arc_idx01 below is an index into sub_scores, it is also an arc_idx123
        // into entering_arc_batches.
        K2_EVAL(
            c, num_states_this_batch, lambda_set_entering_arcs_etc,
            (int32_t state_idx) {  // state_idx is into state_batch_scores_data
                                   // and entering_arc_idxs.
              // arc_idx is into entering_arc_batch_data.
              int32_t arc_idx = entering_arc_idxs_data[state_idx];
              FloatType score;
              int32_t fsas_arc_idx012;
              if (arc_idx == -1) {
                score = negative_infinity;
                fsas_arc_idx012 = -1;
              } else {
                fsas_arc_idx012 = entering_arc_batch_data[arc_idx];
                score = entering_arc_batch_scores_data[arc_idx];
              }
              // we'll later ignore this score if it was the start state.
              state_batch_scores_data[state_idx] = score;
              int32_t fsas_state_idx01 = states_batch_data[state_idx];
              entering_arcs_data[fsas_state_idx01] = fsas_arc_idx012;
            });
      }
    }

    // Copy those scores to the corresponding state in state_scores.
    // `state_idx` is an index into `states_batch_data.values`.
    K2_EVAL(
        c, num_states_this_batch, lambda_copy_state_scores,
        (int32_t state_idx) {
          int32_t fsas_state_idx01 = states_batch_data[state_idx];
          FloatType score = state_batch_scores_data[state_idx];
          // The if-statement below is to prevent it overriding the zero score
          // for the start-states.  We only bother checking whether it's a start
          // state if the score is -infinity, to save memory bandwidth.  (It
          // would always be -infinity for start states because they have no
          // entering arcs; these FSAs are acyclic.
          if (score != negative_infinity ||
              fsas_state_idx01 !=
                  fsas_row_splits1_data[fsas_row_ids1_data[fsas_state_idx01]]) {
            state_scores_data[fsas_state_idx01] = score;
          }
        });
  }

  return state_scores;
}

template <typename FloatType>
void BackpropGetArcPost(FsaVec &fsas, Ragged<int32_t> &incoming_arcs,
                        const Array1<FloatType> &arc_post_deriv,
                        Array1<FloatType> *forward_scores_deriv,
                        Array1<FloatType> *backward_scores_deriv) {
  NVTX_RANGE(K2_FUNC);
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  K2_CHECK(forward_scores_deriv != nullptr && backward_scores_deriv != nullptr);
  ContextPtr c = GetContext(fsas, incoming_arcs, arc_post_deriv);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(incoming_arcs.NumAxes(), 3);
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  K2_CHECK_EQ(arc_post_deriv.Dim(), num_arcs);
  K2_DCHECK_EQ(incoming_arcs.Dim0(), num_fsas);
  K2_DCHECK_EQ(incoming_arcs.TotSize(1), num_states);
  K2_DCHECK_EQ(incoming_arcs.TotSize(2), num_arcs);

  *forward_scores_deriv = Array1<FloatType>(c, num_states);
  *backward_scores_deriv = Array1<FloatType>(c, num_states);
  // compute forward_scores_deriv
  Ragged<FloatType> ragged_forward_scores_deriv(fsas.shape, arc_post_deriv);
  SumPerSublist<FloatType>(ragged_forward_scores_deriv, FloatType(0),
                           forward_scores_deriv);
  // compute backward_scores_deriv
  Array1<FloatType> incoming_arc_post_deriv =
      arc_post_deriv[incoming_arcs.values];
  Ragged<FloatType> ragged_backward_scores_deriv(incoming_arcs.shape,
                                                 incoming_arc_post_deriv);
  SumPerSublist<FloatType>(ragged_backward_scores_deriv, FloatType(0),
                           backward_scores_deriv);
  // set the forward_scores_deriv for the final state and backward_scores_deriv
  // for the start state.
  Ragged<FloatType> arc_post_deriv_per_fsa =
      ragged_forward_scores_deriv.RemoveAxis(1);
  Array1<FloatType> tot_arc_post_deriv(c, num_fsas);
  SumPerSublist<FloatType>(arc_post_deriv_per_fsa, FloatType(0),
                           &tot_arc_post_deriv);
  FloatType *tot_arc_post_deriv_data = tot_arc_post_deriv.Data(),
            *forward_scores_deriv_data = forward_scores_deriv->Data(),
            *backward_scores_deriv_data = backward_scores_deriv->Data();
  const int32_t *fsa_row_splits1_data = fsas.RowSplits(1).Data();
  K2_EVAL(
      c, num_fsas, lambda_set_deriv_for_start_and_final_state,
      (int32_t fsa_idx) {
        int32_t start_state = fsa_row_splits1_data[fsa_idx],
                start_state_next_fsa = fsa_row_splits1_data[fsa_idx + 1];
        if (start_state_next_fsa - start_state > 0) {
          FloatType deriv = FloatType(-0.5) * tot_arc_post_deriv_data[fsa_idx];
          forward_scores_deriv_data[start_state_next_fsa - 1] = deriv;
          backward_scores_deriv_data[start_state] = deriv;
        }
      });
}

template void BackpropGetArcPost(FsaVec &fsas, Ragged<int32_t> &incoming_arcs,
                                 const Array1<float> &arc_post_deriv,
                                 Array1<float> *forward_scores_deriv,
                                 Array1<float> *backward_scores_deriv);
template void BackpropGetArcPost(FsaVec &fsas, Ragged<int32_t> &incoming_arcs,
                                 const Array1<double> &arc_post_deriv,
                                 Array1<double> *forward_scores_deriv,
                                 Array1<double> *backward_scores_deriv);

template <typename FloatType>
Array1<FloatType> GetBackwardScores(FsaVec &fsas,
                                    Ragged<int32_t> &state_batches,
                                    Ragged<int32_t> &leaving_arc_batches,
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
  // just using DCHECK below to save time in production code
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.NumElements(), num_arcs);

  const FloatType negative_infinity =
      -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> state_scores(c, num_states, negative_infinity);
  FloatType *state_scores_data = state_scores.Data();
  const int32_t *fsas_row_splits1_data = fsas.RowSplits(1).Data(),
                *fsas_row_ids1_data = fsas.RowIds(1).Data(),
                *fsas_row_ids2_data = fsas.RowIds(2).Data();

  // set the score of final state in each fsa to be 0
  K2_EVAL(
      c, num_fsas, lambda_set_final_state_score, (int32_t fsa_idx) {
        int32_t start_state = fsas_row_splits1_data[fsa_idx],
                start_state_next_fsa = fsas_row_splits1_data[fsa_idx + 1];
        if (start_state_next_fsa - start_state > 0)
          state_scores_data[start_state_next_fsa - 1] = 0;
      });

  RaggedAxis0Splitter<int32_t> arc_batches_splitter(leaving_arc_batches);

  const Arc *arcs = fsas.values.Data();

  // process batch sequentially.
  for (int32_t i = num_batches - 1; i >= 0; --i) {
    int32_t arc_begin;
    Ragged<int32_t> this_arc_batch =
        arc_batches_splitter.GetElement(i, &arc_begin);
    int32_t state_begin = arc_batches_splitter.GetOffset(i, 2),
            state_end = arc_batches_splitter.GetOffset(i + 1, 2),
            num_states_this_batch = state_end - state_begin,
            num_arcs_this_batch = this_arc_batch.NumElements();

    Ragged<FloatType> this_arc_batch_scores(this_arc_batch.shape);

    const int32_t *this_arc_batch_data = this_arc_batch.values.Data();
    FloatType *this_arc_batch_scores_data = this_arc_batch_scores.values.Data();

    // Get arc backward scores at the beginning of arcs in this batch
    K2_EVAL(
        c, num_arcs_this_batch, lambda_set_leaving_arc_score,
        (int32_t arc_idx) {
          int32_t fsa_arc_idx012 = this_arc_batch_data[arc_idx];
          float curr_arc_score = arcs[fsa_arc_idx012].score;
          int32_t dest_state_idx1 = arcs[fsa_arc_idx012].dest_state,
                  src_state_idx1 = arcs[fsa_arc_idx012].src_state,
                  src_state_idx01 = fsas_row_ids2_data[fsa_arc_idx012],
                  idx0x = src_state_idx01 - src_state_idx1,
                  dest_state_idx01 = idx0x + dest_state_idx1;
          this_arc_batch_scores_data[arc_idx] =
              state_scores_data[dest_state_idx01] + curr_arc_score;
        });

    Array1<FloatType> this_batch_state_scores(c, num_states_this_batch);

    // get scores per state in this batch
    if (log_semiring) {
      LogSumPerSublist(this_arc_batch_scores, negative_infinity,
                       &this_batch_state_scores);
    } else {
      MaxPerSublist(this_arc_batch_scores, negative_infinity,
                    &this_batch_state_scores);
    }

    Array1<int32_t> this_batch_state_ids =
        state_batches.values.Arange(state_begin, state_end);
    const int32_t *this_batch_state_ids_data = this_batch_state_ids.Data();

    const FloatType *this_batch_state_scores_data =
        this_batch_state_scores.Data();
    // copy those scores to the corresponding states in state_scores (they are
    // in a different order).
    K2_EVAL(
        c, num_states_this_batch, lambda_copy_state_scores,
        (int32_t state_idx) {
          int32_t fsas_state_idx01 = this_batch_state_ids_data[state_idx];
          FloatType score = this_batch_state_scores_data[state_idx];
          if (score != negative_infinity ||
              fsas_state_idx01 + 1 !=
                  fsas_row_splits1_data[fsas_row_ids1_data[fsas_state_idx01] +
                                        1]) {
            // The if-block is to ensure we don't overwrite the final-states'
            // backward-probs (0) with -infinity.  We check the score first to
            // avoid unnecessary memory traffic.
            state_scores_data[fsas_state_idx01] = score;
          }
        });
  }
  return state_scores;
}

template <typename FloatType>
Array1<FloatType> BackpropGetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &entering_arc_batches, bool log_semiring,
    const Array1<FloatType> &backward_scores,
    const Array1<FloatType> &backward_scores_deriv_in) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetContext(fsas, state_batches, entering_arc_batches,
                            backward_scores, backward_scores_deriv_in);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  K2_CHECK_EQ(entering_arc_batches.NumAxes(), 4);
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = entering_arc_batches.Dim0();
  K2_DCHECK_EQ(state_batches.TotSize(1), num_fsas * num_batches);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(entering_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(entering_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(entering_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(entering_arc_batches.NumElements(), num_arcs);
  K2_DCHECK_EQ(backward_scores.Dim(), num_states);
  K2_DCHECK_EQ(backward_scores_deriv_in.Dim(), num_states);

  // We will be adding to the elements of `backward_scores_deriv`.
  // `backward_scores_deriv_in` was just the derivative w.r.t. the output of
  // GetBackwardScores(), but because GetBackwardScores() is recursive,
  // the derivatives for earlier states contribute to those of later ones.
  Array1<FloatType> backward_scores_deriv(backward_scores_deriv_in.Clone());
  FloatType *backward_scores_deriv_data = backward_scores_deriv.Data();
  const int32_t *fsas_row_splits1_data = fsas.RowSplits(1).Data(),
                *fsas_row_ids1_data = fsas.RowIds(1).Data(),
                *fsas_row_ids2_data = fsas.RowIds(2).Data();
  const FloatType *backward_scores_data = backward_scores.Data();
  const Arc *arcs = fsas.values.Data();

  Array1<FloatType> arc_scores_deriv(c, num_arcs);  // will return this.
  FloatType *arc_scores_deriv_data = arc_scores_deriv.Data();
  RaggedAxis0Splitter<int32_t> arc_batches_splitter(entering_arc_batches);
  const FloatType negative_infinity =
      -std::numeric_limits<FloatType>::infinity();

  if (log_semiring) {
    // For each batch of states, from start to end (opposite direction to
    // GetBackwardScores())...
    for (int32_t b = 0; b < num_batches; ++b) {
      int32_t arc_begin;
      Ragged<int32_t> entering_arc_batch =
          arc_batches_splitter.GetElement(b, &arc_begin);
      const int32_t *entering_arc_batch_data = entering_arc_batch.values.Data();
      Ragged<FloatType> entering_arc_deriv(entering_arc_batch.shape);
      FloatType *entering_arc_deriv_data = entering_arc_deriv.values.Data();

      K2_EVAL(
          c, entering_arc_batch.NumElements(), lambda_set_arc_deriv_etc,
          (int32_t arc_idx) {
            int32_t fsas_arc_idx012 = entering_arc_batch_data[arc_idx];
            const Arc &arc = arcs[fsas_arc_idx012];
            int32_t dest_state_idx1 = arc.dest_state,
                    src_state_idx1 = arc.src_state,
                    src_state_idx01 = fsas_row_ids2_data[fsas_arc_idx012],
                    state_idx0x = src_state_idx01 - src_state_idx1,
                    dest_state_idx01 = state_idx0x + dest_state_idx1;
            FloatType dest_score = backward_scores_data[dest_state_idx01],
                      arc_begin_score = dest_score + arc.score,
                      src_score = backward_scores_data[src_state_idx01];
            // so that arc_begin_score - src_score will never be nan
            if (src_score == negative_infinity) src_score = -negative_infinity;
            // alpha = d(src_score) / d(arc_begin_score)
            FloatType alpha = exp(arc_begin_score - src_score),
                      arc_deriv =
                          alpha * backward_scores_deriv_data[src_state_idx01];
            K2_CHECK_LT(alpha, 1.1);
            arc_scores_deriv_data[fsas_arc_idx012] = arc_deriv;
            entering_arc_deriv_data[arc_idx] = arc_deriv;
          });

      int32_t state_begin = arc_batches_splitter.GetOffset(b, 2),
              state_end = arc_batches_splitter.GetOffset(b + 1, 2),
              this_num_states = state_end - state_begin;

      // `state_score_derivs` is the extra part contributed to
      // `backward_scores_deriv` by the recursion, for the batch of states we're
      // currently processing.
      Array1<FloatType> state_score_derivs(c, this_num_states);
      SumPerSublist<FloatType>(entering_arc_deriv, 0, &state_score_derivs);
      const FloatType *state_score_derivs_data = state_score_derivs.Data();
      const int32_t *state_ids_batch_data =
          state_batches.values.Data() + state_begin;
      K2_EVAL(
          c, this_num_states, lambda_modify_state_score_derivs,
          (int32_t state_idx) {
            int32_t fsas_state_idx01 = state_ids_batch_data[state_idx];
            FloatType state_score_extra_deriv =
                state_score_derivs_data[state_idx];
            backward_scores_deriv_data[fsas_state_idx01] +=
                state_score_extra_deriv;
          });
    }
  } else {
    // in a single kernel, figure out the contribution of each arc to its
    // source-state's backward prob by seeing which outgoing arc contributes the
    // max loglike; this uses the shape of the fsas.  Note, it's arbitrary in
    // case of ties, we pick one.
    Ragged<FloatType> arc_begin_scores(fsas.shape);
    FloatType *arc_begin_scores_data = arc_begin_scores.values.Data();
    K2_EVAL(
        c, num_arcs, lambda_set_arc_begin_scores, (int32_t arc_idx012) {
          const Arc &arc = arcs[arc_idx012];
          int32_t dest_state_idx1 = arc.dest_state,
                  src_state_idx1 = arc.src_state,
                  src_state_idx01 = fsas_row_ids2_data[arc_idx012],
                  state_idx0x = src_state_idx01 - src_state_idx1,
                  dest_state_idx01 = state_idx0x + dest_state_idx1;
          FloatType dest_score = backward_scores_data[dest_state_idx01],
                    arc_begin_score = dest_score + arc.score;
          arc_begin_scores_data[arc_idx012] = arc_begin_score;
        });
    Array1<int32_t> best_leaving_arc_idx(c, num_states);
    ArgMaxPerSublist(arc_begin_scores, negative_infinity,
                     &best_leaving_arc_idx);
    const int32_t *best_leaving_arc_idx_data = best_leaving_arc_idx.Data();

    for (int32_t b = 0; b < num_batches; ++b) {
      int32_t arc_begin;
      Ragged<int32_t> entering_arc_batch =
          arc_batches_splitter.GetElement(b, &arc_begin);
      const int32_t *entering_arc_batch_data = entering_arc_batch.values.Data();
      Ragged<FloatType> entering_arc_deriv(entering_arc_batch.shape);
      FloatType *entering_arc_deriv_data = entering_arc_deriv.values.Data();

      K2_EVAL(
          c, entering_arc_batch.NumElements(), lambda_set_arc_deriv_etc,
          (int32_t arc_idx)->void {
            int32_t fsas_arc_idx012 = entering_arc_batch_data[arc_idx];
            int32_t src_state_idx01 = fsas_row_ids2_data[fsas_arc_idx012];
            FloatType arc_deriv = FloatType(0);
            if (best_leaving_arc_idx_data[src_state_idx01] == fsas_arc_idx012) {
              arc_deriv = backward_scores_deriv_data[src_state_idx01];
            }  // otherwise arc_deriv is 0.0, the arc's score has no effect
            arc_scores_deriv_data[fsas_arc_idx012] = arc_deriv;
            entering_arc_deriv_data[arc_idx] = arc_deriv;
          });

      int32_t state_begin = arc_batches_splitter.GetOffset(b, 2),
              state_end = arc_batches_splitter.GetOffset(b + 1, 2),
              this_num_states = state_end - state_begin;

      // `state_score_derivs` is the extra part contributed to
      // `backward_scores_deriv` by the recursion, for the batch of states we're
      // currently processing.
      Array1<FloatType> state_score_derivs(c, this_num_states);
      SumPerSublist<FloatType>(entering_arc_deriv, 0, &state_score_derivs);
      const FloatType *state_score_derivs_data = state_score_derivs.Data();
      const int32_t *state_ids_batch_data =
          state_batches.values.Data() + state_begin;
      K2_EVAL(
          c, this_num_states, lambda_modify_state_score_derivs,
          (int32_t state_idx)->void {
            int32_t fsas_state_idx01 = state_ids_batch_data[state_idx];
            FloatType state_score_extra_deriv =
                state_score_derivs_data[state_idx];
            backward_scores_deriv_data[fsas_state_idx01] +=
                state_score_extra_deriv;
          });
    }
  }
  return arc_scores_deriv;
}

template Array1<float> BackpropGetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &entering_arc_batches, bool log_semiring,
    const Array1<float> &backward_scores,
    const Array1<float> &backward_scores_deriv_in);
template Array1<double> BackpropGetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &entering_arc_batches, bool log_semiring,
    const Array1<double> &backward_scores,
    const Array1<double> &backward_scores_deriv_in);

template <typename FloatType>
Array1<FloatType> BackpropGetForwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches, bool log_semiring,
    const Array1<int32_t> *entering_arcs,
    const Array1<FloatType> &forward_scores,
    const Array1<FloatType> &forward_scores_deriv_in) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = GetContext(fsas, state_batches, leaving_arc_batches,
                            forward_scores, forward_scores_deriv_in);
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  K2_CHECK_EQ(leaving_arc_batches.NumAxes(), 4);
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = leaving_arc_batches.Dim0();
  K2_DCHECK_EQ(state_batches.TotSize(1), num_fsas * num_batches);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.NumElements(), num_arcs);
  K2_DCHECK_EQ(forward_scores.Dim(), num_states);
  K2_DCHECK_EQ(forward_scores_deriv_in.Dim(), num_states);

  // We will be adding to the elements of `forward_scores_deriv`.
  // `forward_scores_deriv_in` was just the derivative w.r.t. the output of
  // GetForwardScores(), but because GetForwardScores() is recursive,
  // the derivatives for later states contribute to those of earlier ones.
  Array1<FloatType> forward_scores_deriv(forward_scores_deriv_in.Clone());
  FloatType *forward_scores_deriv_data = forward_scores_deriv.Data();
  const int32_t *fsas_row_splits1_data = fsas.RowSplits(1).Data(),
                *fsas_row_ids1_data = fsas.RowIds(1).Data(),
                *fsas_row_ids2_data = fsas.RowIds(2).Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  const Arc *arcs = fsas.values.Data();

  Array1<FloatType> arc_scores_deriv(c, num_arcs);  // will return this.
  FloatType *arc_scores_deriv_data = arc_scores_deriv.Data();
  RaggedAxis0Splitter<int32_t> arc_batches_splitter(leaving_arc_batches);
  const FloatType negative_infinity =
      -std::numeric_limits<FloatType>::infinity();

  if (log_semiring) {
    // For each batch of states, from end to start (opposite direction to
    // GetForwardScores())...

    for (int32_t b = num_batches - 1; b >= 0; --b) {
      int32_t arc_begin;
      Ragged<int32_t> leaving_arc_batch =
          arc_batches_splitter.GetElement(b, &arc_begin);
      int32_t *leaving_arc_batch_data = leaving_arc_batch.values.Data();
      Ragged<FloatType> leaving_arc_deriv(leaving_arc_batch.shape);
      FloatType *leaving_arc_deriv_data = leaving_arc_deriv.values.Data();

      K2_EVAL(
          c, leaving_arc_batch.NumElements(), lambda_set_arc_deriv_etc,
          (int32_t arc_idx) {
            int32_t fsas_arc_idx012 = leaving_arc_batch_data[arc_idx];
            const Arc &arc = arcs[fsas_arc_idx012];
            int32_t dest_state_idx1 = arc.dest_state,
                    src_state_idx1 = arc.src_state,
                    src_state_idx01 = fsas_row_ids2_data[fsas_arc_idx012],
                    state_idx0x = src_state_idx01 - src_state_idx1,
                    dest_state_idx01 = state_idx0x + dest_state_idx1;
            FloatType src_score = forward_scores_data[src_state_idx01],
                      arc_end_score = src_score + arc.score,
                      dest_score = forward_scores_data[dest_state_idx01];
            // so that arc_end_score - dest_score will never be nan
            if (dest_score == negative_infinity)
              dest_score = -negative_infinity;
            // alpha = d(dest_score) / d(arc_end_score)
            FloatType alpha = exp(arc_end_score - dest_score),
                      arc_deriv =
                          alpha * forward_scores_deriv_data[dest_state_idx01];
            K2_CHECK_LT(alpha, 1.1);
            arc_scores_deriv_data[fsas_arc_idx012] = arc_deriv;
            leaving_arc_deriv_data[arc_idx] = arc_deriv;
          });

      int32_t state_begin = arc_batches_splitter.GetOffset(b, 2),
              state_end = arc_batches_splitter.GetOffset(b + 1, 2),
              this_num_states = state_end - state_begin;

      // `state_score_derivs` is the extra part contributed to
      // `forward_scores_deriv` by the recursion, for the batch of states we're
      // currently processing.
      Array1<FloatType> state_score_derivs(c, this_num_states);
      SumPerSublist<FloatType>(leaving_arc_deriv, 0, &state_score_derivs);
      const FloatType *state_score_derivs_data = state_score_derivs.Data();
      const int32_t *state_ids_batch_data =
          state_batches.values.Data() + state_begin;
      K2_EVAL(
          c, this_num_states, lambda_modify_state_score_derivs,
          (int32_t state_idx) {
            int32_t fsas_state_idx01 = state_ids_batch_data[state_idx];
            FloatType state_score_extra_deriv =
                state_score_derivs_data[state_idx];
            forward_scores_deriv_data[fsas_state_idx01] +=
                state_score_extra_deriv;
          });
    }
  } else {
    K2_CHECK_NE(entering_arcs, nullptr);
    K2_CHECK_EQ(entering_arcs->Dim(), num_states);
    K2_CHECK(entering_arcs->Context()->IsCompatible(*c));
    const int32_t *entering_arcs_data = entering_arcs->Data();

    for (int32_t b = num_batches - 1; b >= 0; --b) {
      int32_t arc_begin;
      Ragged<int32_t> leaving_arc_batch =
          arc_batches_splitter.GetElement(b, &arc_begin);
      const int32_t *leaving_arc_batch_data = leaving_arc_batch.values.Data();
      Ragged<FloatType> leaving_arc_deriv(leaving_arc_batch.shape);
      FloatType *leaving_arc_deriv_data = leaving_arc_deriv.values.Data();

      K2_EVAL(
          c, leaving_arc_batch.NumElements(), lambda_set_arc_deriv_etc,
          (int32_t arc_idx)->void {
            int32_t fsas_arc_idx012 = leaving_arc_batch_data[arc_idx];
            const Arc &arc = arcs[fsas_arc_idx012];
            int32_t dest_state_idx1 = arc.dest_state,
                    src_state_idx1 = arc.src_state,
                    src_state_idx01 = fsas_row_ids2_data[fsas_arc_idx012],
                    state_idx0x = src_state_idx01 - src_state_idx1,
                    dest_state_idx01 = state_idx0x + dest_state_idx1;
            FloatType arc_deriv = FloatType(0);
            if (entering_arcs_data[dest_state_idx01] == fsas_arc_idx012) {
              arc_deriv = forward_scores_deriv_data[dest_state_idx01];
            }  // otherwise arc_deriv is 0.0, the arc's score has no effect
            arc_scores_deriv_data[fsas_arc_idx012] = arc_deriv;
            leaving_arc_deriv_data[arc_idx] = arc_deriv;
          });

      int32_t state_begin = arc_batches_splitter.GetOffset(b, 2),
              state_end = arc_batches_splitter.GetOffset(b + 1, 2),
              this_num_states = state_end - state_begin;

      // `state_score_derivs` is the extra part contributed to
      // `forward_scores_deriv` by the recursion, for the batch of states we're
      // currently processing.
      Array1<FloatType> state_score_derivs(c, this_num_states);
      SumPerSublist<FloatType>(leaving_arc_deriv, 0, &state_score_derivs);
      const FloatType *state_score_derivs_data = state_score_derivs.Data();
      const int32_t *state_ids_batch_data =
          state_batches.values.Data() + state_begin;
      K2_EVAL(
          c, this_num_states, lambda_modify_state_score_derivs,
          (int32_t state_idx)->void {
            int32_t fsas_state_idx01 = state_ids_batch_data[state_idx];
            FloatType state_score_extra_deriv =
                state_score_derivs_data[state_idx];
            forward_scores_deriv_data[fsas_state_idx01] +=
                state_score_extra_deriv;
          });
    }
  }
  return arc_scores_deriv;
}

template Array1<float> BackpropGetForwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches, bool log_semiring,
    const Array1<int32_t> *entering_arcs, const Array1<float> &forward_scores,
    const Array1<float> &forward_scores_deriv_in);

template Array1<double> BackpropGetForwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches, bool log_semiring,
    const Array1<int32_t> *entering_arcs, const Array1<double> &forward_scores,
    const Array1<double> &forward_scores_deriv_in);

template <typename FloatType>
Array1<FloatType> GetTotScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK(IsCompatible(fsas, forward_scores));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1);
  K2_CHECK_EQ(num_states, forward_scores.Dim());

  const FloatType negative_infinity =
      -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> tot_scores(c, num_fsas, negative_infinity);
  FloatType *tot_scores_data = tot_scores.Data();

  const int32_t *fsa_row_splits1_data = fsas.RowSplits(1).Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  K2_EVAL(
      c, num_fsas, lambda_copy_tot_scores, (int32_t fsa_idx) {
        int32_t start_state = fsa_row_splits1_data[fsa_idx],
                start_state_next_fsa = fsa_row_splits1_data[fsa_idx + 1];
        if (start_state_next_fsa > start_state) {  // non-empty fsa
          int32_t final_state_idx = start_state_next_fsa - 1;
          tot_scores_data[fsa_idx] = forward_scores_data[final_state_idx];
        }
      });

  return tot_scores;
}

template <typename FloatType>
Array1<FloatType> GetArcPost(FsaVec &fsas,
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

  Array1<FloatType> arc_scores(c, num_arcs),
      fsa_neg_tot_scores(c, num_fsas);  // minus the tot scores per FSA.
  FloatType *arc_scores_data = arc_scores.Data(),
            *fsa_neg_tot_scores_data = fsa_neg_tot_scores.Data();

  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  const int32_t *fsa_row_ids1 = fsas.RowIds(1).Data();
  const int32_t *fsa_row_ids2 = fsas.RowIds(2).Data();
  const Arc *arcs = fsas.values.Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  const FloatType *backward_scores_data = backward_scores.Data();
  const FloatType negative_infinity =
      -std::numeric_limits<FloatType>::infinity();

  K2_EVAL(
      c, num_fsas, lambda_set_fsa_scores, (int32_t fsa_idx0)->void {
        int32_t begin = fsa_row_splits1[fsa_idx0],
                end = fsa_row_splits1[fsa_idx0 + 1];
        FloatType tot_score = FloatType(0);
        if (begin != end) {
          tot_score = FloatType(0.5) * (forward_scores_data[end - 1] +
                                        backward_scores_data[begin]);
        }
        // We never set the score of a state to positive_infinity, otherwise
        // we may get NaN when add it with negative_infinity below. But this
        // usually would not happen for a connected FSA.
        fsa_neg_tot_scores_data[fsa_idx0] =
            tot_score != negative_infinity ? -tot_score : negative_infinity;
      });

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
                                      backward_scores_data[dest_state_idx01] +
                                      fsa_neg_tot_scores_data[idx0];
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
                                         bool log_semiring);
template Array1<double> GetBackwardScores(FsaVec &fsas,
                                          Ragged<int32_t> &state_batches,
                                          Ragged<int32_t> &leaving_arc_batches,
                                          bool log_semiring);

template Array1<float> GetArcPost(FsaVec &fsas,
                                  const Array1<float> &forward_scores,
                                  const Array1<float> &backward_scores);
template Array1<double> GetArcPost(FsaVec &fsas,
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
  for (int32_t i = 0; i < num_fsas; ++i) {
    num_frames[i] = RandInt(min_frames, max_frames) + 1;
    tot_frames += num_frames[i];
  }

  Array2<float> scores(c, tot_frames, num_symbols + 1);
  auto scores_acc = scores.Accessor();

  std::vector<int32_t> row_splits_vec(num_fsas + 1);
  row_splits_vec[0] = 0;
  int32_t cur_start_frame = 0;
  RandIntGenerator gen;
  for (int32_t i = 0; i < num_fsas; ++i) {
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
        K2_DCHECK_GT(src_row_splits1_data[idx0 + 1],
                     src_row_splits1_data[idx0]);
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
  int32_t num_fsas = fsas->Dim0(), num_states = fsas->TotSize(1);

  Array1<int32_t> changed(c, 1, 0);
  Renumbering renumber_states(c, num_states);
  renumber_states.Keep() = static_cast<char>(1);  // by default keep all states.

  int32_t *changed_data = changed.Data();
  char *keep_data = renumber_states.Keep().Data();
  const int32_t *row_splits1_data = fsas->RowSplits(1).Data();
  K2_EVAL(
      c, num_fsas, lambda_set_must_remove, (int32_t i)->void {
        int32_t num_states = (row_splits1_data[i + 1] - row_splits1_data[i]);
        if (num_states == 1) keep_data[row_splits1_data[i]] = 0;
        changed_data[0] = 1;
      });
  if (changed[0] == 0) return;  // an optimization..
  fsas->shape = RemoveSomeEmptyLists(fsas->shape, 1, renumber_states);
}

template <typename FloatType>
Array1<FloatType> GetArcCdf(FsaOrVec &fsas,
                            Array1<FloatType> &arc_post) {
  K2_CHECK_GE(fsas.NumAxes(), 2);
  K2_CHECK_LE(fsas.NumAxes(), 3);
  int32_t state_axis = (fsas.NumAxes() == 3 ? 1 : 0);
  ContextPtr c = GetContext(fsas, arc_post);
  int32_t num_states = fsas.TotSize(state_axis),
      num_arcs = fsas.NumElements();
  Array1<FloatType> state_post(c, num_states);
  // Use lowest() instead of -infinity() to initialize the sum, to avoid
  // -infinity - (-infinity) = NaN; but actually this shouldn't really be an
  // issue as we shouldn't be taking paths with scores equal to -infinity.
  Ragged<FloatType> arc_post_ragged(fsas.shape, arc_post);
  LogSumPerSublist(arc_post_ragged,
                   std::numeric_limits<FloatType>::lowest(),
                   &state_post);

  Array1<FloatType> arc_pdf(c, num_arcs);
  const Arc *arcs_data = fsas.values.Data();
  const FloatType *arc_post_data = arc_post.Data(),
      *state_post_data = state_post.Data();
  FloatType *arc_pdf_data = arc_pdf.Data();
  // it's row_ids2 if it's an FsaVec (3 axes), else row_ids1.
  const int32_t *fsas_row_ids2_data = fsas.RowIds(1 + state_axis).Data(),
      *fsas_row_splits2_data = fsas.RowSplits(1 + state_axis).Data();
  K2_EVAL(c, num_arcs, lambda_set_arc_probs, (int32_t i) {
      FloatType arc_post = arc_post_data[i];
      int32_t state_idx = fsas_row_ids2_data[i];
      arc_post -= state_post_data[state_idx];
      FloatType arc_pdf_val = exp(arc_post);
      arc_pdf_data[i] = arc_pdf_val;
      K2_DCHECK_GE(arc_pdf_val, 0);
    });


  Ragged<FloatType> arc_pdf_ragged(fsas.shape, arc_pdf);

  Array1<FloatType> arc_cdf(c, num_arcs);

  SegmentedExclusiveSum(arc_pdf_ragged, &arc_cdf);
  FloatType *arc_cdf_data = arc_cdf.Data();

  /*
    The remaining code would not be necessary if we didn't have to deal with
    roundoff effects.  The point of the remaining code is to ensure that
    the "implicit last element" is exactly 1.0 and not, say, 1.00001 or 0.9999999.

    Specifically, we ensure that if exp(arc_post[arc_idx012]) == 0, then the
    cdf value for the next arc (taking it to be 1.0 if this is the last arc
    leaving this state) will be exactly equal to the cdf value for this arc.

    This makes easier the job of deciding, for some  0 <= p <= 1.0, which
    arc leaving a specific state it "belongs to", without any danger of
    picking an arc that goes to a state that cannot reach the final state.
  */

  // `arc_inv_tots` will contain the inverses of the totals of the arc_post
  // values leaving each state, if those totals were nonzero and finite; and
  // otherwise, 1.0.
  Array1<FloatType> arc_inv_tots(c, num_states);
  FloatType *arc_inv_tots_data = arc_inv_tots.Data();
  K2_EVAL(c, num_states, lambda_set_inv_tots, (int32_t i) {
      int32_t begin_arc = fsas_row_splits2_data[i],
          end_arc = fsas_row_splits2_data[i + 1];
      FloatType inv_tot = FloatType(1.0);
      if (end_arc > begin_arc) {
        FloatType this_tot = arc_cdf_data[end_arc - 1] +
            arc_pdf_data[end_arc - 1];
        if (this_tot > 0)
          inv_tot = FloatType(1.0) / this_tot;
        // we'll leave inv_tot at 1.0 for states that had zero or NaN
        // `this_tot`, which could happen if those states were not accessible or
        // not coaccessible.
      }
      arc_inv_tots_data[i] = inv_tot;
    });

  // The next kernel divides by the total, to account for where the sum is not
  // exactly 1.0.
  K2_EVAL(c, num_arcs, lambda_modify_cdf, (int32_t arc_idx012) {
      int32_t state_idx01 = fsas_row_ids2_data[arc_idx012];
      FloatType cdf_value = arc_cdf_data[arc_idx012],
          cdf_value_modified = cdf_value * arc_inv_tots_data[state_idx01];
      arc_cdf_data[arc_idx012] = cdf_value_modified;
    });

  // The next kernel ensures that no value (for the arcs leaving a state) is
  // greater than any subsequent value, by setting it to the smallest of any of
  // the later values (including an implicit final 1.0).  Typically this will be
  // a no-op; in certain cases when there are many arcs with small
  // probabilities, this could have a small effect due to roundoff.
  K2_EVAL(c, num_arcs, lambda_fix_cdf, (int32_t arc_idx012) {
      int32_t state_idx01 = fsas_row_ids2_data[arc_idx012],
          arc_idx01x_next = fsas_row_splits2_data[state_idx01 + 1];
      FloatType cur_value = arc_cdf_data[arc_idx012],
          cutoff = cur_value + FloatType(0.0001);
      FloatType smallest_later_value = cur_value;

      for (int32_t next_arc = arc_idx012 + 1; next_arc < arc_idx01x_next;
           ++next_arc) {
        FloatType next_val = arc_cdf_data[next_arc];
        if (next_val < smallest_later_value)
          smallest_later_value = next_val;
        // any of these errors will only be between values that are very close
        // together... so we can stop this search after there is a decent gap in
        // value.
        if (next_val > cutoff)
          break;
      }
      if (1.0 < smallest_later_value)
        smallest_later_value = 1.0;
      if (smallest_later_value != cur_value)
        arc_cdf_data[arc_idx012] = smallest_later_value;
    });
  return arc_cdf;
}

template
Array1<float> GetArcCdf(FsaOrVec &fsas, Array1<float> &arc_post);
template
Array1<double> GetArcCdf(FsaOrVec &fsas, Array1<double> &arc_post);



namespace random_paths_internal {


// shared state (the changing part of it) for the algorithm that gets the paths.
template <typename FloatType>
struct PathState {
  int32_t begin_arc_idx01x;  // first arc_idx012 leaving this state..
  int32_t num_arcs;          // num arcs leaving this state.

  // `p` is a number in the interval [0, 1], which you can think of as being
  // random (although it's initialized deterministically at fixed intervals).
  // As we advance `cur_state_idx01` from the start state to later states in the
  // path, we zoom in more and more; and `p` is always the position within the
  // probability interval spanned by arcs leaving `cur_state_idx01`.
  FloatType p;
};

}  // namespace random_paths_internal

template <typename FloatType>
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<FloatType> &arc_cdf,
                            const Array1<int32_t> &num_paths,
                            Ragged<int32_t> &state_batches) {
  using namespace random_paths_internal;  // NOLINT
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(fsas.NumElements(), arc_cdf.Dim());
  K2_CHECK_EQ(fsas.Dim0(), num_paths.Dim());

  ContextPtr c = GetContext(fsas, arc_cdf, num_paths, state_batches);
  int32_t num_fsas = fsas.Dim0();

  // For each FSA, work out the number of batches of states that it has,
  // which will tell us how much memory we need to allocate.
  Array1<int32_t> num_state_batches(c, num_fsas);
  int32_t max_batches = state_batches.Dim0();
  if (max_batches > 0)
    num_state_batches = -1;  // so we can more easily detect errors..
  else
    num_state_batches = 0;  // kernel below won't set them in this case.

  const int32_t *state_batches_row_splits1 = state_batches.RowSplits(1).Data(),
      *state_batches_row_splits2 = state_batches.RowSplits(2).Data();
  int32_t *num_state_batches_data = num_state_batches.Data();
  K2_EVAL2(
      c, num_fsas, max_batches, lambda_set_num_state_batches,
      (int32_t i, int32_t b) {
        int32_t this_batch_start = state_batches_row_splits1[b],
                num_states =
                    state_batches_row_splits2[this_batch_start + i + 1] -
                    state_batches_row_splits2[this_batch_start + i];
        if (num_states == 0) {
          if (b == 0) {
            num_state_batches_data[i] = 0;
          } else {
            int32_t prev_batch_start = state_batches_row_splits1[b - 1],
                    prev_num_states =
                        state_batches_row_splits2[prev_batch_start + i + 1] -
                        state_batches_row_splits2[prev_batch_start + i];
            if (prev_num_states != 0) num_state_batches_data[i] = b;
          }
        } else if (b + 1 == max_batches) {
          num_state_batches_data[i] = max_batches;
        }
      });

  // For each FSA, the amount of space we'll need is its num_batches times
  // the num_paths requested (no path can be longer than num_batches).
  Array1<int32_t> storage_row_splits(c, num_fsas + 1);
  Array1<int32_t> num_paths_sum(c, num_fsas + 1);

  const int32_t *num_paths_data = num_paths.Data();

  int32_t *storage_row_splits_data = storage_row_splits.Data(),
      *num_paths_sum_data = num_paths_sum.Data();

  K2_EVAL(c, num_fsas, lambda_set_storage_sizes, (int32_t i) {
          int32_t num_paths = num_paths_data[i],
              num_batches = num_state_batches_data[i];
          K2_CHECK_NE(num_batches, -1);
          int32_t space_needed = num_paths * num_batches;
          storage_row_splits_data[i] = space_needed;
          num_paths_sum_data[i] = num_paths;
        });
  ExclusiveSum(storage_row_splits, &storage_row_splits);
  // We copied num_paths first to guarantee that there is an extra element at
  // the end (avoid theoretically possible segfault, as ExclusiveSum would read
  // that not-needed element).
  ExclusiveSum(num_paths_sum, &num_paths_sum);
  int32_t tot_space_needed = storage_row_splits.Back(),
      tot_num_paths = num_paths_sum.Back();

  Array1<int32_t> paths_row_ids(c, tot_num_paths);
  RowSplitsToRowIds(num_paths_sum, &paths_row_ids);
  // (num_paths_sum, paths_row_ids) form respectively the row_splits and row_ids
  // of a ragged tensor with Dim0() == num_fsas and TotSize(1) == tot_num_paths.

  Array1<int32_t> path_storage(c, tot_space_needed);

  const int32_t *paths_row_ids_data = paths_row_ids.Data(),
      *paths_row_splits_data = num_paths_sum_data,  // an alias..
      *fsas_row_ids2_data = fsas.RowIds(2).Data(),
      *fsas_row_splits1_data = fsas.RowSplits(1).Data(),
      *fsas_row_splits2_data = fsas.RowSplits(2).Data();

  int32_t *path_storage_data = path_storage.Data();

  namespace cg = cooperative_groups;
  const unsigned int thread_group_size = 8;  // Can tune this.  Power of 2,
                                             // 1<=thread_group_size<=256
  const FloatType *arc_cdf_data = arc_cdf.Data();
  const Arc *arcs = fsas.values.Data();

  if (c->GetDeviceType() == kCuda) {
    auto lambda_set_paths = [=] __device__(
        cg::thread_block_tile<thread_group_size> g,  // or auto g..
        PathState<FloatType> *shared_data,
        int32_t i) ->void {
     // First get certain fixed information.  All threads get this, which might
     // not seem ideal but I think the GPU will consolidate the identical reads.
     int32_t fsa_idx = paths_row_ids_data[i],
         state_idx0x = fsas_row_splits1_data[fsa_idx],
         path_begin = paths_row_splits_data[fsa_idx],
         path_idx1 = i - path_begin,
         thread_idx = g.thread_rank(),
         final_state = fsas_row_splits1_data[fsa_idx + 1] - 1,
         num_batches = num_state_batches_data[fsa_idx];

     // path_storage_start[0] will contain the path length;
     // then the arcs in the path.  The maximum num-arcs equals
     // num_batches - 1.
     int32_t *path_storage_start = path_storage_data +
         storage_row_splits_data[fsa_idx] + path_idx1 * num_batches;

     if (thread_idx == 0) {  // Initialize shared_data
       // Note: FSAs with no states would have num_paths = 0, and we'd never
       // reach this code.
       int32_t start_state = state_idx0x,
           path_end = paths_row_splits_data[fsa_idx + 1],
           num_paths = path_end - path_begin,
           begin_arc = fsas_row_splits2_data[start_state],
           end_arc = fsas_row_splits2_data[start_state + 1];
       shared_data->begin_arc_idx01x = begin_arc;
       shared_data->num_arcs = end_arc - begin_arc;
       FloatType p = ((FloatType)0.5 + path_idx1) / num_paths;
       shared_data->p = p;
     }

     int32_t path_pos = 0;
     for (; ; path_pos++) {
       // if the following check fails, we'll have to start debugging: something
       // went wrong, but could be lots of things, e.g. we reached a state that
       // we shouldn't have reached (has no arcs) or some other circumstance
       // meant that no arc was chosen.
       K2_DCHECK_LT(path_pos, num_batches);

       g.sync();

       FloatType p = shared_data->p;
       int32_t begin_arc_idx01x = shared_data->begin_arc_idx01x,
           num_arcs = shared_data->num_arcs;

       if (num_arcs == 0) {
         // If we reach a state with no arcs leaving it, it should be the final
         // state; otherwise something went wrong.
         K2_CHECK_EQ(fsas_row_splits2_data[final_state], begin_arc_idx01x);
         if (thread_idx == 0) {
           int32_t path_length = path_pos;
           path_storage_start[0] = path_length;
         }
         return;
       }

       // Must sync again after read and before potential write..
       g.sync();

       int32_t arc_idx2 = thread_idx;
       for (; arc_idx2 < num_arcs; arc_idx2 += g.size()) {
         int32_t arc_idx012 = begin_arc_idx01x + arc_idx2;
         FloatType interval_start = arc_cdf_data[arc_idx012],
             interval_end = (arc_idx2 + 1 == num_arcs ? 1.0 :
                             arc_cdf_data[arc_idx012 + 1]);
         K2_DCHECK_GE(interval_end, interval_start);
         K2_DCHECK_LE(interval_end, 1.0);
         if (p >= interval_start && p <= interval_end &&
             interval_end > interval_start &&
             (p != interval_end || p == 1.0)) {
           // The above if-statement ensures that any 0.0 <= p <= 1.0 will be
           // inside exactly one interval.  The edges of the intervals form a
           // non-decreasing sequence from 0.0 to 1.0.  So every such p is
           // either completely inside an interval, or:
           //    - Is 1.0 and is at the upper boundary of one nonempty interval
           //    - Is 0.0 and is at the lower boundary of one nonempty interval
           //    - Satisfies 0.0 < p < 1.0 and is at the boundary of two
           //      nonempty intervals; we assign it to the higher of the two
           //      intervals.
           // The reason for the + 1 is that we'll put the path length in
           // element 0.
           path_storage_start[path_pos + 1] = arc_idx012;

           // The following will ensure that 0.0 <= p <= 1.0: we know
           // interval_end - interval_start > 0.0, p - interval_start >= 0.0,
           // and (interval_end - interval_start) >= (p - interval_start).
           p = (p - interval_start) / (interval_end - interval_start);

           int32_t next_state_idx01 = arcs[arc_idx012].dest_state + state_idx0x,
                   next_arc_idx01x = fsas_row_splits2_data[next_state_idx01],
                   next_arc_idx01x_next =
                       fsas_row_splits2_data[next_state_idx01 + 1];
           shared_data->begin_arc_idx01x = next_arc_idx01x;
           shared_data->num_arcs = next_arc_idx01x_next - next_arc_idx01x;
           shared_data->p = p;
           break;
         }
       }
     }
    };

    EvalGroupDevice<thread_group_size, PathState<FloatType>>(
        c, tot_num_paths, lambda_set_paths);
  } else {
    // CPU.
    for (int32_t fsa_idx = 0; fsa_idx < num_fsas; ++fsa_idx) {
      int32_t state_idx0x = fsas_row_splits1_data[fsa_idx],
          final_state = fsas_row_splits1_data[fsa_idx + 1] - 1,
          num_paths = num_paths_data[fsa_idx],
          num_batches = num_state_batches_data[fsa_idx];
      for (int32_t path_idx1 = 0; path_idx1 < num_paths; ++path_idx1) {
        int32_t *path_storage_start = path_storage_data +
            storage_row_splits_data[fsa_idx] + path_idx1 * num_batches;

        int32_t cur_state_idx01 = state_idx0x;  // Start state.  Note: start
                                                // state is never the final
                                                // state.
        FloatType p = (FloatType(0.5) + path_idx1) / num_paths;

        int32_t path_pos;
        for (path_pos = 0; path_pos <= num_batches; ++path_pos) {
          // Note: if things are working correctly we should break from this
          // loop before it naturally terminates.
          if (cur_state_idx01 == final_state) {  // Finalize..
            path_storage_start[0] = path_pos;
            break;
          }
          int32_t arc_idx01x = fsas_row_splits2_data[cur_state_idx01],
              arc_idx01x_next = fsas_row_splits2_data[cur_state_idx01 + 1];
          K2_DCHECK_GT(arc_idx01x_next, arc_idx01x);
          // std::upper_bound finds the first index i in the range
          //  [arc_idx01x+1 .. arc_idx01x_next-1] such that
          // arc_cdf_data[i] > p, and if it doesn't exist gives us
          // arc_idx01x_next (so p will be in the last interval).
          const FloatType *begin1 = arc_cdf_data + arc_idx01x + 1,
              *end = arc_cdf_data + arc_idx01x_next;
          int32_t arc_idx2 = std::upper_bound(begin1, end, p) - begin1;
          int32_t arc_idx012 = arc_idx01x + arc_idx2;
          FloatType interval_start = arc_cdf_data[arc_idx012],
              interval_end = (arc_idx012 + 1 == arc_idx01x_next ? 1.0 :
                              arc_cdf_data[arc_idx012 + 1]);
          K2_DCHECK_GE(p, interval_start);
          K2_DCHECK_LE(p, interval_end);
          p = (p - interval_start) / (interval_end - interval_start);

          // + 1 to leave space to store the path length.
          path_storage_start[path_pos + 1] = arc_idx012;
          int32_t next_state_idx01 = arcs[arc_idx012].dest_state + state_idx0x;
          cur_state_idx01 = next_state_idx01;
        }
        if (path_pos > num_batches)
          K2_LOG(FATAL)
              << "Bug in RandomPaths, please ask maintainers for help..";
      }
    }
  }

  Array1<int32_t> path_lengths(c, tot_num_paths + 1);
  int32_t *path_lengths_data = path_lengths.Data();
  K2_EVAL(c, tot_num_paths, lambda_get_path_lengths, (int32_t i) {
      int32_t fsa_idx = paths_row_ids_data[i],
          path_begin = paths_row_splits_data[fsa_idx],
          path_idx1 = i - path_begin,
          num_batches = num_state_batches_data[fsa_idx];
     int32_t *path_storage_start = path_storage_data +
         storage_row_splits_data[fsa_idx] + path_idx1 * num_batches;
     int32_t path_length = path_storage_start[0];
     K2_CHECK_GT(path_length, 0);
     K2_CHECK_LT(path_length, num_batches);
     path_lengths_data[i] = path_length;
    });

  ExclusiveSum(path_lengths, &path_lengths);
  Array1<int32_t> ans_row_splits2(path_lengths);

  Ragged<int32_t> ans(RaggedShape3(&num_paths_sum, &paths_row_ids,
                                   tot_num_paths, &ans_row_splits2, nullptr,
                                   -1));
  const int32_t *ans_row_ids2_data = ans.RowIds(2).Data(),
      *ans_row_splits2_data = ans.RowSplits(2).Data(),
      *ans_row_ids1_data = ans.RowIds(1).Data(),
      *ans_row_splits1_data = ans.RowSplits(1).Data();
  int32_t *ans_data = ans.values.Data();
  int32_t ans_tot_size = ans.shape.NumElements();
  // TODO: maybe optimize the following for CPU, would be quite slow.
  K2_EVAL(c, ans_tot_size, lambda_format_ans_data, (int32_t ans_idx012) {
      int32_t path_idx01 = ans_row_ids2_data[ans_idx012],
          ans_idx01x = ans_row_splits2_data[path_idx01],
          path_pos_idx2 = ans_idx012 - ans_idx01x,
          fsa_idx0 = ans_row_ids1_data[path_idx01],
          path_idx0x = ans_row_splits1_data[fsa_idx0],
          path_idx1 = path_idx01 - path_idx0x,
          num_batches = num_state_batches_data[fsa_idx0];

      int32_t *path_storage_start = path_storage_data +
          storage_row_splits_data[fsa_idx0] + path_idx1 * num_batches;
      ans_data[ans_idx012] = path_storage_start[1 + path_pos_idx2];
    });
  return ans;
}


template
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<float> &arc_cdf,
                            const Array1<int32_t> &num_paths,
                            Ragged<int32_t> &state_batches);
template
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<double> &arc_cdf,
                            const Array1<int32_t> &num_paths,
                            Ragged<int32_t> &state_batches);

template <typename FloatType>
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<FloatType> &arc_cdf,
                            int32_t num_paths,
                            const Array1<FloatType> &tot_scores,
                            Ragged<int32_t> &state_batches) {
  ContextPtr c = GetContext(fsas, arc_cdf, tot_scores, state_batches);
  int32_t num_fsas  = fsas.Dim0();
  Array1<int32_t> num_paths_array(c, num_fsas);
  int32_t *num_paths_data = num_paths_array.Data();
  const FloatType *tot_scores_data = tot_scores.Data();
  // Compiler optimization seems to defeat a comparison with -infinity, on CPU.
  // using std::numeric_limits<FloatType>::lowest() instead.
  FloatType minus_inf = -std::numeric_limits<FloatType>::infinity();
  K2_EVAL(c, num_fsas, lambda_set_num_paths, (int32_t i) {
      FloatType tot_score = tot_scores_data[i];
      // use num_paths=0 if tot_scores[i] == 0.
      int32_t this_num_paths = (tot_score > minus_inf ? num_paths : 0);
      num_paths_data[i] = this_num_paths;
    });
  return RandomPaths(fsas, arc_cdf, num_paths_array, state_batches);
}

template
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<float> &arc_cdf,
                            int32_t num_paths,
                            const Array1<float> &tot_scores,
                            Ragged<int32_t> &state_batches);
template
Ragged<int32_t> RandomPaths(FsaVec &fsas,
                            const Array1<double> &arc_cdf,
                            int32_t num_paths,
                            const Array1<double> &tot_scores,
                            Ragged<int32_t> &state_batches);

}  // namespace k2
