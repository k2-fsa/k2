// k2/csrc/cuda/fsa.cu

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors


#include "k2/csrc/fsa.h"

namespace k2 {

Fsa FsaFromTensor(const Tensor &t, bool *error) {
  *error = false;
  if (t.Dtype() != kInt32Dtype) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, wrong dtype, got "
                    << TraitsOf(t.Dtype()).Name() << " but expected "
                    << TraitsOf(kInt32Dtyt.Dtype()).Name();
    *error = true;
    return Fsa();  // Invalid, empty FSA
  }
  if (t.NumAxes() != 2 || t.Dim(1) != 4) {
    // ...
  }

}





Fsa FsaVecFromTensor(Tensor t, bool *error) {
  if (!t.IsContiguous())
    t = ToContiguous(t);

  *error = false;
  if (t.Dtype() != kInt32Dtype) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, wrong dtype, got "
                    << TraitsOf(t.Dtype()).Name() << " but expected "
                    << TraitsOf(kInt32Dtyt.Dtype()).Name();
    *error = true;
    return Fsa();  // Invalid, empty FSA
  }
  if (t.NumAxes() != 2 || t.Dim(1) != 4) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, shape was "
                    << t.Dims();
  }
  K2_CHECK(sizeof(Arc) == sizeof(int32_t) * 4);
  Arc *arcs = static_cast<Arc*>(t.Data<int32_t>());
  ContextPtr c = t.GetContext();
  int32_t num_arcs = t.Dim(0);

  /* This is a kind of pseudo-vector (that we don't have to allocate memory for)
     It behaves like a pointer to a vector of size `num_arcs`, of 'tails'
     (see `tails concept` in utils.h) which tells us if this is the last arc
     within this FSA.
   */
  struct IsTail {
    __host__ __device__ operator [](int32_t i) {
      return (i + 1 >= num_arcs ||
              arcs[i+1].src_state < arcs[i].src_state);
    }
    IsTail(Arc *arcs): arcs(arcs)  {}
    __host__ __device__ IsTail(const IsTail &other) = default;
    const Arc *arcs;
  };
  Array1<int32_t> fsa_ids(c, num_arcs + 1);
  // `fsa_idss` will be the exclusive sum of `tails`
  IsTail tails(arcs);
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
  Array1<int32> num_states_per_fsa(c, 2 * num_fsas + 1, -1)
  int32_t *num_states_per_fsa_data = num_states_per_fsa.Data();
  auto lambda_find_final_state_a = [=] __host__ __device (int32_t i) -> void {
    if (arcs[i].symbol == -1) {
      int32_t final_state = arcs[i].dest_state,
         fsa_id = fsa_ids_data[i];
      num_states_per_fsa_data[i] = final_state + 1;
    } else if (i + 1 == num_arcs ||
               arcs[i].src_state < arcs[i+1].src_state) {
      // This is the last arc in this FSA.
      // The final state cannot have arcs leaving it, so the smallest
      // possible num-states (counting the final-state as a state) is
      // (last state that has an arc leaving it) + 2.
      int32_t final_state = arcs[i].src_state + 2,
         fsa_id = fsa_ids_data[i];
      num_states_per_fsa_data[i + num_fsas] = fsa_id;
    }
  }
  Eval(c, num_arcs, lambda_get_num_states_a);
  auto lambda_get_num_states_b = [=] __host__ __device (int32_t i) -> void {
    int32_t num_states_1 = num_states_per_fsa_data[i],
       num_states_2 = num_states_per_fsa_data[i + num_fsas];
    if (num_states_2 < 0 ||
        (num_states_1 < 0 && num_states_2 > num_states_1)) {
      // Note: num_states_2 is a lower bound on the final-state, something is
      // wrong if num_states_1 != -1 and num_states_2  is greater than num_states_1.
      num_states_per_fsa_data[2 * num_fsas] = 0;  // Error
    }
    int32_t num_states = (num_states_1 < 0 ?
                          num_states_2 : num_states_1);
    num_states_per_fsa_data[i] = num_states;
  }
  Eval(c, num_arcs, lambda_get_num_states_b);
  if (num_states_per_fsa[2 * num_fsas] == 0) {
    K2_LOG(WARNING) << "Could not convert tensor to FSA, there was a problem "
        "working out the num-states in the FSAs, num_states_per_fsa="
                    << num_states_per_fsa;
  }
  num_states_per_fsa = num_states_per_fsa.Range(0, num_fsas);

  Array1<int32_t> fsa_state_offsets = ExclusiveSum(num_states_per_fsa);
  const int32_t *fsa_state_offsets_data = fsa_state_offsets.Data();

  Array1<int32_t> row_ids2(num_arcs);
  int32_t *row_ids2_data = row_ids2.Data();
  auto set_row_ids2 = [=] __host__ __device (int32_t i) -> void {
    int32_t src_state = arcs[i].src_state,
      fsa_id = fsa_ids_data[i];
    row_ids2_data[i] = fsa_state_offsets_data[fsa_id] + src_state;
  }



        (num_states_2 < != -1 && fsa_id_2 != fsa_id_1)

    if (fsa_id_1 == -1 && fsa_id_2 == -1) {
      num_states_per_fsa_data[2 * num_fsas] = 0;  // Error
    } else if (fsa_id_1 != -1 && fsa_id_2 != -1 &&
               fsa_id_2 ==
  }

  int32_t *num_states_per_fsa_b = num_states_per_fsa.Data();

}


}  // namespace k2
