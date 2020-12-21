// k2/csrc/cuda/fsa.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)
//                      Mobvoi Inc.        (authors: Fangjun Kuang)

// See ../../LICENSE for // clarification regarding multiple authors

#include <string>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/tensor_ops.h"

namespace k2 {

std::ostream &operator<<(std::ostream &os, const Arc &arc) {
  static constexpr char kSep = ' ';
  os << arc.src_state << kSep << arc.dest_state << kSep << arc.label << kSep
     << arc.score;
  return os;
}

std::istream &operator>>(std::istream &is, Arc &arc) {
  InputFixer<float> score;  // helps deal with infinities correctly
  is >> arc.src_state >> arc.dest_state >> arc.label >> score;
  arc.score = score.t;
  return is;
}

std::string FsaPropertiesAsString(int32_t properties) {
  NVTX_RANGE(K2_FUNC);
  static constexpr char kSep = '|';
  std::ostringstream os;

  // clang-format off
  if (properties & kFsaPropertiesValid) os << kSep << "Valid";
  if (properties & kFsaPropertiesNonempty) os << kSep << "Nonempty";
  if (properties & kFsaPropertiesTopSorted) os << kSep << "TopSorted";
  if (properties & kFsaPropertiesTopSortedAndAcyclic) os << kSep << "TopSortedAndAcyclic"; // NOLINT
  if (properties & kFsaPropertiesArcSorted) os << kSep << "ArcSorted";
  if (properties & kFsaPropertiesArcSortedAndDeterministic) os << kSep << "ArcSortedAndDeterministic";  // NOLINT
  if (properties & kFsaPropertiesEpsilonFree) os << kSep << "EpsilonFree";
  if (properties & kFsaPropertiesMaybeAccessible) os << kSep << "MaybeAccessible";  // NOLINT
  if (properties & kFsaPropertiesMaybeCoaccessible) os << kSep << "MaybeCoaccessible";  // NOLINT
  // clang-format on

  size_t offset = (os.str().empty() ? 0 : 1);  // remove leading '|'
  os << '"';
  return std::string("\"") + std::string(os.str().c_str() + offset);
}

void GetFsaVecBasicProperties(FsaVec &fsa_vec, Array1<int32_t> *properties_out,
                              int32_t *tot_properties_out) {
  NVTX_RANGE(K2_FUNC);
  if (fsa_vec.NumAxes() != 3) {
    K2_LOG(FATAL) << "Input has wrong num-axes " << fsa_vec.NumAxes()
                  << " vs. 3.";
  }
  ContextPtr c = fsa_vec.Context();
  const int32_t *row_ids1_data = fsa_vec.shape.RowIds(1).Data(),
                *row_splits1_data = fsa_vec.shape.RowSplits(1).Data(),
                *row_ids2_data = fsa_vec.shape.RowIds(2).Data(),
                *row_splits2_data = fsa_vec.shape.RowSplits(2).Data();
  Arc *arcs_data = fsa_vec.values.Data();

  int32_t num_arcs = fsa_vec.values.Dim();

  // properties per arc; these will be and-ed over all arcs.
  Array1<int32_t> properties(c, num_arcs);
  int32_t num_states = fsa_vec.shape.RowIds(1).Dim(),
          num_fsas = fsa_vec.shape.Dim0();

  // `reachable[idx01]` will be true if the state with index idx01 has an arc
  // entering it or is state 0 of its FSA, not counting self-loops; it's a
  // looser condition than being 'accessible' in FSA terminology, simply meaning
  // it's reachable from some state (which might not itself be reachable).
  //
  // reachable[num_states + idx01] will be true if the state with index idx01 is
  // the final-state of its FSA (i.e. last-numbered) or has at least one arc
  // leaving it, not counting self-loops. Again, it's a looser condition than
  // being 'co-accessible' in FSA terminology.

  Array1<char> reachable_mem(c, num_states * 2 + num_fsas * 2,
                             static_cast<char>(0));

  Array1<char> reachable = reachable_mem.Range(0, num_states);
  Array1<char> co_reachable = reachable_mem.Range(num_states, num_states);

  int32_t *properties_data = properties.Data();
  char *reachable_data = reachable.Data();  // access co_reachable via this.

  K2_EVAL(
      c, num_arcs, lambda_get_properties, (int32_t idx012)->void {
        Arc arc = arcs_data[idx012];
        Arc prev_arc;
        if (idx012 > 0) prev_arc = arcs_data[idx012 - 1];
        int32_t idx01 = row_ids2_data[idx012], idx01x = row_splits2_data[idx01],
                idx2 = idx012 - idx01x, idx0 = row_ids1_data[idx01],
                idx0x = row_splits1_data[idx0],
                idx0x_next = row_splits1_data[idx0 + 1], idx1 = idx01 - idx0x,
                idx0xx = row_splits2_data[idx0x];
        int32_t this_fsa_num_states = idx0x_next - idx0x;

        int32_t neg_property = 0;
        if (arc.src_state != idx1) neg_property |= kFsaPropertiesValid;
        if (arc.dest_state < 0 || arc.dest_state >= this_fsa_num_states)
          neg_property |= kFsaPropertiesValid;
        if (arc.dest_state <= arc.src_state) {
          neg_property |= kFsaPropertiesTopSortedAndAcyclic;
          if (arc.dest_state < arc.src_state)
            neg_property |= kFsaPropertiesTopSorted;
        }
        if (arc.label == 0) neg_property |= kFsaPropertiesEpsilonFree;
        if (arc.label < 0) {
          if (arc.label != -1) {  // neg. symbols != -1 are not allowed.
            neg_property |= kFsaPropertiesValid;
          } else {
            if (arc.dest_state != this_fsa_num_states - 1)
              neg_property |= kFsaPropertiesValid;
          }
        }
        if (arc.label != -1 && arc.dest_state == this_fsa_num_states - 1)
          neg_property |= kFsaPropertiesValid;
        if (arc.dest_state < 0 || arc.dest_state >= this_fsa_num_states)
          neg_property |= kFsaPropertiesValid;
        else if (arc.dest_state != arc.src_state)
          reachable_data[idx0x + arc.dest_state] = static_cast<char>(1);

        if (idx0xx == idx012) {
          // first arc in this FSA (whether or not it's from state 0..)
          reachable_data[idx0x] =
              static_cast<char>(1);  // state 0 is reachable.
          // final state is always co-reachable.
          // Note: below, we're effectively accessing co_reachable_data.
          reachable_data[num_states + idx0x_next - 1] = static_cast<char>(1);
        }

        if (arc.dest_state != arc.src_state)
          // Note: below, we're effectively accessing co_reachable_data.
          reachable_data[num_states + idx01] = 1;
        if (idx2 != 0) {
          // this is not the first arc leaving this state...
          if (static_cast<uint32_t>(arc.label) <=
              static_cast<uint32_t>(prev_arc.label))
            neg_property |= kFsaPropertiesArcSortedAndDeterministic;
          if (static_cast<uint32_t>(arc.label) <
              static_cast<uint32_t>(prev_arc.label))
            neg_property |= kFsaPropertiesArcSorted;
          if (arc.label == prev_arc.label &&
              arc.dest_state < prev_arc.dest_state)
            neg_property |= kFsaPropertiesArcSorted;
        }
        properties_data[idx012] = ~neg_property;
      });

  // Figure out the properties per FSA.
  RaggedShape fsa_to_arcs_shape = RemoveAxis(fsa_vec.shape, 1),
              fsa_to_states_shape = RemoveAxis(fsa_vec.shape, 2);
  Ragged<int32_t> properties_ragged(fsa_to_arcs_shape, properties);
  Ragged<char> reachable_ragged(fsa_to_states_shape, reachable),
      co_reachable_ragged(fsa_to_states_shape, co_reachable);

  Array1<int32_t> properties_per_fsa_mem(c, num_fsas + 1),
      properties_per_fsa = properties_per_fsa_mem.Range(0, num_fsas),
      properties_total = properties_per_fsa_mem.Range(num_fsas, 1);

  Array1<char> reachable_per_fsa =
                   reachable_mem.Range(num_states * 2, num_fsas),
               co_reachable_per_fsa =
                   reachable_mem.Range(num_states * 2 + num_fsas, num_fsas);
  {
    ParallelRunner pr(c);
    {
      With(pr.NewStream());
      AndPerSublist(properties_ragged, static_cast<int32_t>(kFsaAllProperties),
                    &properties_per_fsa);
    }
    {
      With(pr.NewStream());
      AndPerSublist(reachable_ragged, static_cast<char>(1), &reachable_per_fsa);
    }
    {
      With(pr.NewStream());
      AndPerSublist(co_reachable_ragged, static_cast<char>(1),
                    &co_reachable_per_fsa);
    }
  }

  {
    int32_t *properties_per_fsa_data = properties_per_fsa.Data();
    char *reachable_per_fsa_data = reachable_per_fsa.Data(),
         *co_reachable_per_fsa_data = co_reachable_per_fsa.Data();

    K2_EVAL(
        c, num_fsas, lambda_finalize_properties, (int32_t i)->void {
          int32_t neg_properties = ~(properties_per_fsa_data[i]);
          char reachable = reachable_per_fsa_data[i],
               co_reachable = co_reachable_per_fsa_data[i];
          int32_t fsa_has_no_arcs = (row_splits2_data[row_splits1_data[i]] ==
                                     row_splits2_data[row_splits1_data[i + 1]]);
          neg_properties |= (!reachable * kFsaPropertiesMaybeAccessible) |
                            (!co_reachable * kFsaPropertiesMaybeCoaccessible) |
                            (fsa_has_no_arcs * kFsaPropertiesNonempty);
          properties_per_fsa_data[i] = ~neg_properties;
        });

    And(properties_per_fsa, static_cast<int32_t>(kFsaAllProperties),
        &properties_total);
    *tot_properties_out = properties_total[0];
    *properties_out = properties_per_fsa;
  }
}

FsaVec FsaToFsaVec(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  if (fsa.NumAxes() != 2) return fsa;
  ContextPtr &c = fsa.values.Context();
  RaggedShape first_axis = TrivialShape(c, fsa.shape.Dim0());
  RaggedShape fsa_vec_shape = ComposeRaggedShapes(first_axis, fsa.shape);
  return Ragged<Arc>(fsa_vec_shape, fsa.values);
}

int32_t GetFsaBasicProperties(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  if (fsa.NumAxes() != 2) return 0;
  FsaVec vec = FsaToFsaVec(fsa);
  Array1<int32_t> properties;
  int32_t ans;
  GetFsaVecBasicProperties(vec, &properties, &ans);
  return ans;
}

Fsa FsaFromArray1(Array1<Arc> &array, bool *error) {
  NVTX_RANGE(K2_FUNC);
  const Arc *arcs_data = array.Data();
  ContextPtr &c = array.Context();
  int32_t num_arcs = array.Dim();
  // We choose to return an Fsa with no states and no arcs.  We could also have
  // chosen to return an Fsa with 2 states and no arcs.
  if (num_arcs == 0)
    return Fsa(EmptyRaggedShape(c, 2), Array1<Arc>(c, 0));
  *error = false;

  // If the FSA has arcs entering the final state, that will
  // tell us what the final-state id is.
  // If there are no arcs entering the final-state, we let the final state be
  // (highest numbered state that has arcs leaving it) + 1, so num_states
  // (highest numbered state that has arcs leaving it) + 2.

  // element 0 is num-states, element 1 is error flag that's set to
  // 0 on error.

  Array1<int32_t> num_states_array(c, 2, -1);
  int32_t *num_states_data = num_states_array.Data();

  Array1<int32_t> row_ids1(c, num_arcs);  // maps arc->state.
  int32_t *row_ids1_data = row_ids1.Data();

  K2_EVAL(
      c, num_arcs, lambda_misc, (int32_t i)->void {
        row_ids1_data[i] = arcs_data[i].src_state;
        if (arcs_data[i].label == -1) {
          int32_t final_state = arcs_data[i].dest_state;
          int32_t old_value = num_states_data[0];
          if (old_value >= 0 && old_value != final_state + 1)
            num_states_data[1] = 0;  // set error flag.
          num_states_data[0] = final_state + 1;
        }
      });
  num_states_array = num_states_array.To(GetCpuContext());
  int32_t num_states = num_states_array[0], error_flag = num_states_array[1];
  if (error_flag == 0) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, there was a problem "
                       "working out the num-states in the FSA, num_states="
                    << num_states;
    *error = true;
    return Fsa();
  }
  if (num_states == -1) {
    // there was no final arc, so let the final state be the highest-numbered
    // state that is referenced, plus one.

    Array1<int32_t> max_state(c, num_arcs);
    int32_t *max_state_data = max_state.Data();
    K2_EVAL(
        c, num_arcs, lambda_get_dest_state, (int32_t i)->void {
          int32_t dest = arcs_data[i].dest_state, src = arcs_data[i].src_state;
          max_state_data[i] = (dest > src ? dest : src);
        });
    Array1<int32_t> max_state0 = max_state.Range(0, 1);
    Max(max_state, 0, &max_state0);
    num_states = max_state[0] + 2;
  }

  if (!ValidateRowIds(row_ids1)) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, "
                       "src_states of arcs were out of order";
    *error = true;
    return Fsa();
  }
  Array1<int32_t> row_splits1(c, num_states + 1);
  RowIdsToRowSplits(c, num_arcs, row_ids1_data, false, num_states,
                    row_splits1.Data());
#ifndef NDEBUG
  if (!ValidateRowSplitsAndIds(row_splits1, row_ids1, nullptr)) {
    K2_LOG(FATAL) << "Failure validating row-splits/row-ids, likely code error";
  }
#endif

  RaggedShape fsas_shape =
      RaggedShape2(&row_splits1, &row_ids1, row_ids1.Dim());
  Fsa ans = Ragged<Arc>(fsas_shape, array);
  int32_t tot_properties = GetFsaBasicProperties(ans);
  // TODO: check properties, at least
  int32_t required_props = (kFsaPropertiesValid | kFsaPropertiesNonempty);
  if ((tot_properties & required_props) != required_props) {
    K2_LOG(WARNING) << "Did not have expected properties "
                    << FsaPropertiesAsString(tot_properties & required_props)
                    << " vs. " << FsaPropertiesAsString(required_props)
                    << ", all properties were: "
                    << FsaPropertiesAsString(tot_properties);
    *error = true;
  }
  return ans;
}

Tensor FsaToTensor(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(fsa.NumAxes(), 2);
  Array2<int32_t> arcs_as_ints(fsa.values.Dim(), 4, 4, fsa.values.ByteOffset(),
                               fsa.values.GetRegion());
  return arcs_as_ints.ToTensor();
}

Fsa FsaFromTensor(Tensor &t, bool *error) {
  NVTX_RANGE(K2_FUNC);
  if (!t.IsContiguous()) t = ToContiguous(t);

  *error = false;
  if (t.GetDtype() != kInt32Dtype) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, wrong dtype, got "
                    << TraitsOf(t.GetDtype()).Name() << " but expected "
                    << TraitsOf(kInt32Dtype).Name();
    *error = true;
    return Fsa();  // Invalid, empty FSA
  }
  if (t.NumAxes() != 2 || t.Dim(1) != 4) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, shape was "
                    << t.Dims();
    *error = true;
    return Fsa();  // Invalid, empty FSA
  }
  K2_CHECK_EQ(sizeof(Arc), sizeof(int32_t) * 4);
  int32_t *tensor_data = t.Data<int32_t>();

  Array1<Arc> arc_array(t.Dim(0), t.GetRegion(), t.ByteOffset());
  return FsaFromArray1(arc_array, error);
}

FsaVec FsaVecFromTensor(Tensor &t, bool *error) {
  NVTX_RANGE(K2_FUNC);
  if (!t.IsContiguous()) t = ToContiguous(t);

  *error = false;
  if (t.GetDtype() != kInt32Dtype) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, wrong dtype, got "
                    << TraitsOf(t.GetDtype()).Name() << " but expected "
                    << TraitsOf(kInt32Dtype).Name();
    *error = true;
    return Fsa();  // Invalid, empty FSA
  }
  int32_t num_fsas;
  if (t.NumAxes() != 1) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, shape was "
                    << t.Dims();
    *error = true;
    return Fsa();
  }
  int32_t num_ints = t.Dim(0);
  Array1<int32_t> int_array(num_ints, t.GetRegion(), t.ByteOffset());
  num_fsas = int_array[0];

  int32_t arcs_start = 2 + (num_fsas + 1) * 2;
  if (num_fsas < 0 || num_ints < arcs_start ||
      (num_ints - arcs_start) % 4 != 0) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, num_ints = "
                    << num_ints << ", num_fsas = " << num_fsas;
    *error = true;
    return Fsa();
  }

  Array1<int32_t> row_splits1 = int_array.Range(2, num_fsas + 1),
                  row_splits12 =
                      int_array.Range(2 + num_fsas + 1, num_fsas + 1),
                  arcs_ints =
                      int_array.Range(arcs_start, num_ints - arcs_start);
  int32_t num_arcs = (num_ints - arcs_start) / 4;

  Array1<Arc> arcs(num_arcs, arcs_ints.GetRegion(), arcs_ints.ByteOffset());

  if (num_arcs != row_splits12[num_fsas]) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, num_arcs = "
                    << num_arcs << " vs. " << row_splits12[num_fsas];
    *error = true;
    return Fsa();
  }
  if (!ValidateRowSplits(row_splits1) || !ValidateRowSplits(row_splits12)) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, "
                       "row_splits were out of order";
    *error = true;
    return Fsa();
  }
  ContextPtr &c = int_array.Context();

  // TODO: would be nice to transfer this and row_splits12[num_fsas] at the same
  // time.
  int32_t num_states = row_splits1[num_fsas];
  Array1<int32_t> row_ids1(c, num_states),
      row_ids12(c, num_arcs),  // we'll modify row_ids12 to be row_ids2.
      row_splits2(c, num_states + 1);
  RowSplitsToRowIds(row_splits1, &row_ids1);
  RowSplitsToRowIds(row_splits12, &row_ids12);

  const int32_t *row_ids1_data = row_ids1.Data(),
                *row_splits12_data = row_splits12.Data(),
                *row_splits1_data = row_splits1.Data();
  int32_t *row_ids12_data = row_ids12.Data();
  Arc *arcs_data = arcs.Data();

  K2_EVAL(
      c, num_arcs, lambda_make_row_ids2, (int32_t arc_idx012)->void {
        int32_t fsa_idx0 = row_ids12_data[arc_idx012],
                state_idx0x = row_splits1_data[fsa_idx0];
        int32_t state_idx1 = arcs_data[arc_idx012].src_state,
                state_idx01 = state_idx0x + state_idx1;
        row_ids12_data[arc_idx012] =
            state_idx01;  // we're turning this into the row_ids2.
      });

  Array1<int32_t> &row_ids2 =
      row_ids12;  // we overwrote the data in the lambda above.
  RowIdsToRowSplits(row_ids2, &row_splits2);

  if (!ValidateRowSplitsAndIds(row_splits2, row_ids2,
                               &row_splits12)) {  // last arg is temp space
    K2_LOG(WARNING)
        << "Could not convert tensor to FSA, problem validating "
           "row-splits and row-ids (likely data corruption or code bug)";
    *error = true;
    return Fsa();
  }

  return FsaVec(RaggedShape3(&row_splits1, &row_ids1, num_states, &row_splits2,
                             &row_ids2, num_arcs),
                arcs);
}

Tensor FsaVecToTensor(const FsaVec &fsa_vec) {
  NVTX_RANGE(K2_FUNC);
  if (fsa_vec.NumAxes() != 3) {
    K2_LOG(FATAL) << "Expected num-axes == 3. Given: " << fsa_vec.NumAxes();
  }
  Array1<int32_t> row_splits1 = fsa_vec.shape.RowSplits(1),
                  row_splits12 = fsa_vec.shape.RowSplits(2)[row_splits1];
  int32_t num_fsas = fsa_vec.shape.Dim0();
  ContextPtr &c = row_splits1.Context();
  // vector containing: [ num_fsas, 0 ]
  Array1<int32_t> meta_info = Range(c, 2, num_fsas, -num_fsas);
  const Array1<Arc> &arcs = fsa_vec.values;
  Array1<int32_t> arcs_linearized(arcs.Dim() * 4, arcs.GetRegion(),
                                  arcs.ByteOffset());
  int32_t byte_offset = arcs.ByteOffset();
  // The next if-statement detects when this FSA was previously serialized.
  if (byte_offset == (2 + (num_fsas + 1) * 2) * 4 &&
      row_splits1.ByteOffset() == 2 * 4) {
    Array1<int32_t> meta_info_orig(2, arcs.GetRegion(), 0),
        row_splits12_orig(num_fsas + 1, arcs.GetRegion(),
                          (2 + (num_fsas + 1)) * 4);
    if (Equal(meta_info_orig, meta_info) &&
        Equal(row_splits12, row_splits12_orig)) {
      return Array1<int32_t>(2 + (num_fsas + 1) * 2 + arcs.Dim() * 4,
                             arcs.GetRegion(), 0)
          .ToTensor();
    }
  }

  Array1<int32_t> *arrays[4] = {&meta_info, &row_splits1, &row_splits12,
                                &arcs_linearized};
  return Append(4, (const Array1<int32_t> **)arrays).ToTensor();
}

std::ostream &operator<<(std::ostream &os, const DenseFsaVec &dfsavec) {
  DenseFsaVec d_cpu = dfsavec.To(GetCpuContext());
  int32_t num_fsas = d_cpu.shape.Dim0();
  const int32_t *row_splits = d_cpu.shape.RowSplits(1).Data();
  os << "DenseFsaVec{ ";
  for (int32_t i = 0; i < num_fsas; i++) {
    int32_t start = row_splits[i], end = row_splits[i + 1];
    os << dfsavec.scores.RowArange(start, end);
  }
  return os << " }";
}

DenseFsaVec DenseFsaVec::operator[] (const Array1<int32_t> &indexes) {
  Array1<int32_t> elem_indexes;
  RaggedShape ans_shape = Index(this->shape, indexes,
                                &elem_indexes);
  bool allow_minus_one = false;
  Array2<float> ans_scores = IndexRows(this->scores, elem_indexes,
                                       allow_minus_one);
  return DenseFsaVec(ans_shape, ans_scores);
}

}  // namespace k2
