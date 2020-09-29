// k2/csrc/cuda/fsa.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for // clarification regarding multiple authors

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa.h"

namespace {
/* Will be used in FsaVecFromTensor to call ExclusiveSum (which calls
   cub::DeviceScan::ExclusiveSum internally).

   This is a kind of pseudo-vector (that we don't have to allocate
   memory for) It behaves like a pointer to a vector of size
   `num_arcs`, of 'tails' (see `tails concept` in utils.h) which
   tells us if this is the last arc within this FSA.
 */
struct IsLastArcOfFsa {
  int32_t num_arcs;
  const k2::Arc *arcs;

  explicit IsLastArcOfFsa(int32_t num_arcs, const k2::Arc *arcs):
      : num_arcs(num_arcs), arcs(arcs) {}
  __host__ __device__ IsLastArcOfFsa(const IsLastArcOfFsa &other)
      : num_arcs(num_arcs), arcs(other.arcs)  {}

  // operator[] and operator+ are required by cub::DeviceScan::ExclusiveSum
  __host__ __device__ bool operator[](int32_t i) const {
    return (i + 1 >= num_arcs || arcs[i + 1].src_state < arcs[i].src_state);
  }
  __host__ __device__ IsLastArcOfFsa operator+(int32_t n) const {
    IsLastArcOfFsa tmp(*this);
    tmp.arcs += n;
    return tmp;
  }
};

struct IsLastArcOfState {
  int32_t num_arcs;
  const k2::Arc *arcs;

  explicit IsLastArcOfState(int32_t num_arcs, const k2::Arc *arcs):
      : num_arcs(num_arcs), arcs(arcs) {}
  __host__ __device__ IsLastArcOfState(const IsLastArcOfState &other)
      : num_arcs(num_arcs), arcs(other.arcs)  {}

  // operator[] and operator+ are required by cub::DeviceScan::ExclusiveSum
  __host__ __device__ bool operator[](int32_t i) const {
    return (i + 1 >= num_arcs || arcs[i + 1].src_state < arcs[i].src_state);
  }
  __host__ __device__ IsLastArcOfState operator+(int32_t n) const {
    IsLastArcOfState tmp(*this);
    tmp.arcs += n;
    return tmp;
  }
};



}  // namespace

namespace std {
// value_type is required by cub::DeviceScan::ExclusiveSum
template <>
struct iterator_traits<::IsLastArcOfState> {
  typedef bool value_type;
};
template <>
struct iterator_traits<::IsLastArcOfFsa> {
  typedef bool value_type;
};
}  // namespace std

namespace k2 {

int32_t GetFsaVecBasicProperties(FsaVec &fsa_vec) {
  if (fsa_vec.NumAxes() != 3) {
    K2_LOG(FATAL) << "FsaVec has wrong number of axes: " << fsa_vec.NumAxes()
                  << ", expected 3.";
  }
  ContextPtr c = fsa_vec.Context();
  const int32_t *row_ids1_data = fsa_vec.shape.RowIds(1).Data(),
                *row_splits1_data = fsa_vec.shape.RowSplits(1).Data(),
                *row_ids2_data = fsa_vec.shape.RowIds(2).Data(),
                *row_splits2_data = fsa_vec.shape.RowSplits(2).Data();
  Arc *arcs_data = fsa_vec.values.Data();

  int32_t num_arcs = fsa_vec.values.Dim();

  // `neg_` means negated. It's more convenient to do it this way.
  Array1<int32_t> neg_properties(c, num_arcs);
  int32_t num_states = fsa_vec.shape.RowIds(1).Dim(),
          num_fsas = fsa_vec.shape.Dim0();

  // `reachable[idx01]` will be true if the state with index idx01 has an arc
  // entering it or is state 0 of its FSA, not counting self-loops; it's a
  // looser condition than being 'accessible' in FSA terminlogy, simply meaning
  // it's reachable from some state (which might not itself be reachable).
  //
  // reachable[num_states + idx01] will be true if the state with index idx01 is
  // the final-state of its FSA (i.e. last-numbered) or has at least one arc
  // leaving it, not counting self-loops. Again, it's a looser condition than
  // being 'co-accessible' in FSA terminology.
  Array1<char> reachable(c, num_states * 2 + 1, static_cast<char>(0));
  Array1<char> flag = reachable.Range(num_states * 1, 1);
  Array1<char> co_reachable = reachable.Range(num_states, num_states);
  reachable = reachable.Range(0, num_states);
  int32_t *neg_properties_data = neg_properties.Data();
  char *reachable_data = reachable.Data();  // access co_reachable via this.

  auto lambda_get_properties = [=] __host__ __device__(int32_t idx012) -> void {
    Arc arc = arcs_data[idx012];
    Arc prev_arc;
    if (idx012 > 0) prev_arc = arcs_data[idx012 - 1];
    int32_t idx01 = row_ids2_data[idx012], idx01x = row_splits2_data[idx012],
            idx2 = idx012 - idx01x, idx0 = row_ids1_data[idx01],
            idx0x = row_splits1_data[idx0],
            idx0x_next = row_splits1_data[idx0 + 1], idx1 = idx01 - idx0x,
            idx0xx = row_splits2_data[idx0x];
    int32_t this_fsa_num_states = idx0x_next - idx0x;

    int32_t neg_property = 0;
    if (arc.src_state != idx1) neg_property |= kFsaPropertiesValid;
    if (arc.dest_state <= arc.src_state) {
      neg_property |= kFsaPropertiesTopSortedAndAcyclic;
      if (arc.dest_state < arc.src_state)
        neg_property |= kFsaPropertiesTopSorted;
    }
    if (arc.symbol == 0) neg_property |= kFsaPropertiesEpsilonFree;
    if (arc.symbol < 0) {
      if (arc.symbol != -1) {  // neg. symbols != -1 are not allowed.
        neg_property |= kFsaPropertiesValid;
      } else {
        if (arc.dest_state != this_fsa_num_states - 1)
          neg_property |= kFsaPropertiesValid;
      }
    }
    if (arc.symbol != -1 && arc.dest_state == this_fsa_num_states - 1)
      neg_property |= kFsaPropertiesValid;
    if (arc.dest_state < 0 || arc.dest_state >= this_fsa_num_states)
      neg_property |= kFsaPropertiesValid;
    else if (arc.dest_state != arc.src_state)
      reachable_data[idx0x + arc.dest_state] = static_cast<char>(1);

    if (idx0xx == idx012) {
      // first arc in this FSA (whether or not it's from state 0..)
      reachable_data[idx0x] = static_cast<char>(1);  // state 0 is reachable.
      // final state is always co-reachable.
      reachable_data[num_states + idx0x_next - 1] = static_cast<char>(1);
      // there was a problem with the state-indexes which makes this
      // impossible to deserialize from a list of arcs.
      if (idx012 > 0 && prev_arc.src_state <= arc.src_state)
        neg_property |= kFsaPropertiesSerializable;
    }

    if (idx2 == 0) {
      // First arc leaving this state record that this state has arcs leaving
      // it.
      if (arc.dest_state != arc.src_state)
        reachable_data[num_states + idx01] = 1;
    }
    neg_properties_data[idx012] = ~neg_property;
  };
  Eval(c, num_arcs, lambda_get_properties);
  // Note: eventually, for more diagnostics, we could use AndPerSublist to get
  // the properties one by  one.

  Array1<int32_t> and_properties = neg_properties.Range(0, 1);  // ok to overlap
  Array1<char> and_reachable = reachable.Range(0, 1);
  Array1<char> and_co_reachable = reachable.Range(0, 1);
  ParallelRunner pr(c);
  {
    With(pr.NewStream());
    And(neg_properties, static_cast<int32_t>(kFsaAllProperties),
        &and_properties);
  }
  {
    With(pr.NewStream());
    And(reachable, static_cast<char>(1), &and_reachable);
  }
  {
    With(pr.NewStream());
    And(co_reachable, static_cast<char>(1), &and_co_reachable);
  }
  {
    char *flag_data = flag.Data();
    auto lambda_find_empty_fsas = [=] __host__ __device__(int32_t i) -> void {
      if (row_ids1_data[i + 1] == row_ids1_data[i])
        *flag_data = 1;  // There is an empty FSA.
    };
    Eval(pr.NewStream(), num_fsas, lambda_find_empty_fsas);
  }

  int32_t properties =
      and_properties[0] |
      (and_reachable[0] ? kFsaPropertiesMaybeAccessible : 0) |
      (and_co_reachable[0] ? kFsaPropertiesMaybeCoaccessible : 0);
  if (flag[0])  // not serializable because has empty FSA.
    properties &= ~kFsaPropertiesSerializable;

  // probably tons of bugs in this :-(
  return properties;
}


Fsa FsaFromArray1(Array1<Arc> &array, bool *error) {
  const Arc *arcs_data = reinterpret_cast<const Arc *>(array.Data());
  ContextPtr c = array.Context();
  const int32_t num_arcs = array.Dim();

  // num-arcs per state
  Array1<int32_t> num_arcs_array(c, 1, num_arcs);
  const int32_t *num_arcs_data = num_arcs_array.Data();
  IsLastArcOfState tails(num_arcs, num_arcs_data);
  ExclusiveSum(c, num_arcs + 1, tails, fsa_ids.Data());

  int32_t num_fsas = fsa_ids[num_arcs];
  fsa_ids = fsa_ids.Range(0, num_arcs);
  int32_t *fsa_ids_data = fsa_ids.Data();

  // Get the num-states per FSA, including the final-state which must be
  // numbered last.  If the FSA has arcs entering the final state, that will
  // tell us what the final-state id is.
  //   num_state_per_fsa has 2 parts, for 2 different methods of finding the
  //   right FSA (we'll later shorten it after disambiguating).
  // The very last element has an error flag that we'll set to 0 if there is
  // an error.
  Array1<int32_t> num_states_per_fsa(c, 2 * num_fsas + 1, -1);
  int32_t *num_states_per_fsa_data = num_states_per_fsa.Data();
  auto lambda_get_num_states_a = [=] __host__ __device__(int32_t i) -> void {
    if (arcs_data[i].symbol == -1) {
      int32_t final_state = arcs_data[i].dest_state, fsa_id = fsa_ids_data[i];
      num_states_per_fsa_data[fsa_id] = final_state + 1;
    } else if (i + 1 == num_arcs || arcs_data[i].src_state < arcs_data[i + 1].src_state) {
      // This is the last arc in this FSA.
      // The final state cannot have arcs leaving it, so the smallest
      // possible num-states (counting the final-state as a state) is
      // (last state that has an arc leaving it) + 2.
      int32_t final_state = arcs_data[i].src_state + 1, fsa_id = fsa_ids_data[i];
      num_states_per_fsa_data[fsa_id] = final_state + 1;
    }
  };
  Eval(c, num_arcs, lambda_get_num_states_a);
  auto lambda_get_num_states_b = [=] __host__ __device__(int32_t i) -> void {
    int32_t num_states_1 = num_states_per_fsa_data[i],
            num_states_2 = num_states_per_fsa_data[i + num_fsas];
    if (num_states_2 < 0 || (num_states_1 < 0 && num_states_2 > num_states_1)) {
      // Note: num_states_2 is a lower bound on the final-state, something is
      // wrong if num_states_1 != -1 and num_states_2  is greater than
      // num_states_1.
      num_states_per_fsa_data[2 * num_fsas] = 0;  // Error
    }
    int32_t num_states = (num_states_1 < 0 ? num_states_2 : num_states_1);
    num_states_per_fsa_data[i] = num_states;
  };
  Eval(c, num_arcs, lambda_get_num_states_b);
  if (num_states_per_fsa[2 * num_fsas] == 0) {
    K2_LOG(WARNING)
        << "Could not convert tensor to FSA, there was a problem "
           "working out the num-states in the FSAs, num_states_per_fsa="
        << num_states_per_fsa;
  }
  num_states_per_fsa = num_states_per_fsa.Range(0, num_fsas + 1);
  // fsa_state_offsets is of size num_fsas + 1.
  // Note: fsa_state_offsets could be called row_splits1.
  Array1<int32_t> fsa_state_offsets = ExclusiveSum(num_states_per_fsa);
  int32_t tot_num_states = fsa_state_offsets[num_fsas];

  const int32_t *fsa_state_offsets_data = fsa_state_offsets.Data();

  // by `row_ids2` we mean row_ids for axis=2. This is the second
  // of two row_ids vectors. It maps from idx012 to idx01.
  Array1<int32_t> row_ids2(c, num_arcs);
  int32_t *row_ids2_data = row_ids2.Data();
  auto lambda_set_row_ids2 = [=] __host__ __device__(int32_t i) -> void {
    int32_t src_state = arcs_data[i].src_state, fsa_id = fsa_ids_data[i];
    row_ids2_data[i] = fsa_state_offsets_data[fsa_id] + src_state;
  };
  Eval(c, num_arcs, lambda_set_row_ids2);

  Array1<int32_t> row_splits2(c, tot_num_states + 1);
  RowIdsToRowSplits(c, num_arcs, row_ids2_data, false, tot_num_states,
                    row_splits2.Data());
#ifndef NDEBUG
  if (!ValidateRowSplitsAndIds(
          row_splits2, row_ids2,
          &num_states_per_fsa)) {  // last arg is temp space
    K2_LOG(FATAL) << "Failure validating row-splits/row-ids, likely code error";
  }
#endif
  RaggedShape fsas_shape =
      RaggedShape3(&fsa_state_offsets, &fsa_ids, fsa_ids.Dim(), &row_splits2,
                   &row_ids2, row_ids2.Dim());
  Array1<Arc> arcs_array(num_arcs, t.GetRegion(), 0);
  FsaVec ans = Ragged<Arc>(fsas_shape, arcs_array);
  int32_t properties = GetFsaVecBasicProperties(ans);
  // TODO: check properties, at least
  int32_t required_props = (kFsaPropertiesValid | kFsaPropertiesNonempty |
                            kFsaPropertiesSerializable);
  if (properties & required_props) {
    K2_LOG(WARNING) << "Did not have expected properties "
                    << (properties & required_props) << " vs. "
                    << required_props;
    // TODO: better way of displaying properties.
    *error = true;
  }
  return ans;
}

Fsa FsaFromTensor(Tensor &t, bool *error) {
  if (!t.IsContiguous())
    t = ToContiguous(t);

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
  }
  K2_CHECK_EQ(sizeof(Arc), sizeof(int32_t) * 4);
  int32_t *tensor_data = t.Data<int32_t>();

  Array1<Arc> arc_array(t.Dim(0), t.GetRegion(), t.ByteOffset());
  return FsaFromArray1(arc_array, error);
}


Fsa FsaVecFromArray1(Array1<Arc> &array, bool *error) {
  const Arc *arcs_data = reinterpret_cast<const Arc *>(array.Data());
  ContextPtr c = array.Context();
  const int32_t num_arcs = array.Dim();
  Array1<int32_t> row_ids12(c, num_arcs + 1);  // maps arc->fsa_id, like row_ids1[row_ids2]
  IsLastArcOfFsa fsa_tails(num_arcs, arcs_data);
  ExclusiveSum(c, num_arcs + 1, fsa_tails, row_ids12.Data());
  int32_t num_fsas = row_ids12[num_arcs];
  row_ids12 = row_ids12.Range(0, num_arcs);

  Array1<int32_t> row_ids2(c, 1, num_arcs + 1);
  const int32_t *row_ids2_data = row_ids2.Data();
  IsLastArcOfState state_tails(num_arcs, arcs_data);
  ExclusiveSum(c, num_arcs + 1, state_tails, row_ids2_data);

  int32_t *fsa_ids_data = row_ids12.Data();

  // Get the num-states per FSA, including the final-state which must be
  // numbered last.  If the FSA has arcs entering the final state, that will
  // tell us what the final-state id is.  (that goes in num_states_per_fsa).
  // If there are no arcs entering the final-state, we let the final state be
  // (highest numbered state that has arcs leaving it) + 1, so num_states
  // (highest numbered state that has arcs leaving it) + 2.
  //
  // num_states_per_fsa[num_fsas] is an error flag that gets set to 0 on error.
  Array1<int32_t> num_states_per_fsa(c, num_fsas + 1, -1);
  int32_t *num_states_per_fsa_data = num_states_per_fsa.Data();
  auto lambda_get_num_states_a = [=] __host__ __device__(int32_t i) -> void {
    if (arcs_data[i].symbol == -1) {
      int32_t final_state = arcs_data[i].dest_state, fsa_id = fsa_ids_data[i];
      num_states_per_fsa_data[fsa_id] = final_state + 1;
    }
  };
  Eval(c, num_arcs, lambda_get_num_states_a);

  Array1<int32_t> row_splits12(c, num_fsas + 1);
  int32_t * row_splits12_data = row_splits12.Data();
  RowIdsToRowSplits(c, num_arcs, row_ids12.Data(), true,
                    num_fsas, row_splits12.Data());
  auto lambda_get_num_states_b = [=] __host__ __device__(int32_t i) -> void {
     int32_t num_states_1 = num_state_per_fsa_data[i],
         num_states_2 = arcs_data[row_splits_12_data[i+1] - 1].src_state + 2;
    if (num_states_2 <= 0 || (num_states_1 >= 0 && num_states_2 > num_states_1)) {
      // Note: num_states_2 is a lower bound on the final-state, something is
      // wrong if num_states_1 != -1 and num_states_2  is greater than
      // num_states_1.
      num_states_per_fsa_data[2 * num_fsas] = 0;  // Error
    } else {
      int32_t num_states = (num_states_1 < 0 ? num_states_2 : num_states_1);
      num_states_per_fsa_data[i] = num_states;
    }
  };
  Eval(c, num_arcs, lambda_get_num_states_b);
  if (num_states_per_fsa[2 * num_fsas] == 0) {
    K2_LOG(WARNING)
        << "Could not convert tensor to FSA, there was a problem "
           "working out the num-states in the FSAs, num_states_per_fsa="
        << num_states_per_fsa;
    *error = true;
    return Fsa();
  }
  num_states_per_fsa = num_states_per_fsa.Range(0, num_fsas + 1);
  // row_splits1 is of size num_fsas + 1.
  // TODO(dan): make this in-place?
  Array1<int32_t> row_splits1 = ExclusiveSum(num_states_per_fsa);
  int32_t tot_num_states = row_splits1[num_fsas];

  const int32_t *row_splits1_data = row_splits1.Data();

  // by `row_ids2` we mean row_ids for axis=2. This is the second
  // of two row_ids vectors. It maps from idx012 to idx01.
  Array1<int32_t> row_ids2(c, num_arcs);
  int32_t *row_ids2_data = row_ids2.Data();
  auto lambda_set_row_ids2 = [=] __host__ __device__(int32_t i) -> void {
    int32_t src_state = arcs_data[i].src_state, fsa_id = fsa_ids_data[i];
    row_ids2_data[i] = row_splits1_data[fsa_id] + src_state;
  };
  Eval(c, num_arcs, lambda_set_row_ids2);

  if (!ValidateRowIds(row_ids2)) {
    K2_LOG(WARNING)
        << "Could not convert tensor to FSA, src_states of arcs were out of "
        "order";
    *error = true;
    return Fsa();
  }

  Array1<int32_t> row_splits2(c, tot_num_states + 1);
  RowIdsToRowSplits(c, num_arcs, row_ids2_data, false, tot_num_states,
                    row_splits2.Data());
#ifndef NDEBUG
  if (!ValidateRowSplitsAndIds(
          row_splits2, row_ids2,
          &num_states_per_fsa)) {  // last arg is temp space
    K2_LOG(FATAL) << "Failure validating row-splits/row-ids, likely code error";
  }
#endif

  // row_splits1 maps from idx0 to idx01.
  // row_splits12 maps from idx0 to idx012.  row_ids2 maps from idx012 to idx01.
  Array1<int32_t> row_splits1 = row_ids2[row_splits12];
  // row_ids1 maps from idx01 to idx0.
  // row_ids12 maps from idx012 to idx0.  row_splits2 maps from idx01 to idx012.
  Array1<int32_t> row_ids1 = row_ids12[row_splits2];


  RaggedShape fsas_shape =
      RaggedShape3(&row_splits1, &row_ids1, row_ids1.Dim(),
                   &row_splits2, &row_ids2, row_ids2.Dim());
  FsaVec ans = Ragged<Arc>(fsas_shape, array);
  int32_t properties = GetFsaVecBasicProperties(ans);
  // TODO: check properties, at least
  int32_t required_props = (kFsaPropertiesValid | kFsaPropertiesNonempty |
                            kFsaPropertiesSerializable);
  if (properties & required_props) {
    K2_LOG(WARNING) << "Did not have expected properties "
                    << (properties & required_props) << " vs. "
                    << required_props;
    // TODO: better way of displaying properties.
    *error = true;
  }
  return ans;
}



FsaVec FsaVecFromTensor(Tensor &t, bool *error) {
  if (!t.IsContiguous())
    t = ToContiguous(t);

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
  }
  K2_CHECK_EQ(sizeof(Arc), sizeof(int32_t) * 4);
  int32_t *tensor_data = t.Data<int32_t>();

  Array1<Arc> arc_array(t.Dim(0), t.GetRegion(), t.ByteOffset());
  return FsaVecFromArray1(arc_array, error);
}

}  // namespace k2
