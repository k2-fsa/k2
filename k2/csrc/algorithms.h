/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef K2_CSRC_ALGORITHMS_H_
#define K2_CSRC_ALGORITHMS_H_

#include <algorithm>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"

//  this really contains various utilities that are useful for k2 algorithms.
namespace k2 {

class Renumbering {
 public:
  Renumbering() = default;
  // copy constructor
  Renumbering(const Renumbering &src) = default;
  // move constructor
  Renumbering(Renumbering &&src) = default;
  // move assignment
  Renumbering &operator=(Renumbering &&) = default;
  // copy assignment
  Renumbering &operator=(const Renumbering &) = default;


  /*
     This constructor will allocate memory for `keep_` array with size
     `num_old_elems`. User then can call `Keep` to get `keep_` and set values in
     it (1 if kept, 0 if not kept), then call `New2Old` to map new indexes to
     old indexes or `Old2New` to map old indexes to new indexes.

        @param [in] c The context this Renumbering object will work on.
        @param [in] num_old_elems  The number of old indexes (i.e.
                      keep_.Dim() or NumOldElems()).
        @param [in] init_keep_with_zero  If true, we will initialize `keep_`
                      with 0 when creating; if false, we just allocate memory
                      and don't initialize it. CAUTION: usually user should
                      leave it as false (the default value), as coalesced
                      writing to an array (i.e `keep_`) would be much more
                      efficient than writing individual 1's, especially
                      considering that we have used another kernel to
                      initialize the `keep` array with 0 if
                      `init_keep_with_zero` = true. We suggest users should
                      set it with true only when it's hard for them to set
                      both 1s (kept indexes) and 0s (not kept indexes) in one
                      kernel.
  */
  Renumbering(ContextPtr c, int32_t num_old_elems,
              bool init_keep_with_zero = false) {
    Init(c, num_old_elems, init_keep_with_zero);
  }

  /*
    This constructor is not intended for use by users; it is used by
    IdentityRenumbering().  Just sets members to the provided arrays and
    num_new_elems_ to new2old.Dim().
  */
  Renumbering(const Array1<char> &keep,
              const Array1<int32_t> &old2new,
              const Array1<int32_t> &new2old);


  void Init(ContextPtr c, int32_t num_old_elems,
            bool init_keep_with_zero = false) {
    NVTX_RANGE(K2_FUNC);
    // make the underlying region allocate an extra element as we'll do
    // exclusive sum in New2Old() and Old2New()
    Array1<char> temp = Array1<char>(c, num_old_elems + 1);
    if (init_keep_with_zero) temp = 0;
    keep_ = temp.Range(0, num_old_elems);
  }

  int32_t NumOldElems() const { return keep_.Dim(); }

  int32_t NumNewElems() {
    NVTX_RANGE(K2_FUNC);
    if (!old2new_.IsValid()) ComputeOld2New();
    return num_new_elems_;
  }

  // 0 if not kept, 1 if kept (user will write to here).  Its dimension is the
  // `num_old_elems` provided in the constructor (the internal array has an
  // extra element because ExclusiveSum reads one past the end (since we get the
  // result with 1 extra element).
  Array1<char> &Keep() { return keep_; }

  /* Return a mapping from new index to old index.  This is created on
     demand (must only be called after the Keep() array has been populated).

       @return    Returns an array mapping the new indexes to the old
                  (pre-renumbering) indexes. Its dimension is the number of
                  new indexes (i.e. the number of 1 in keep_), but internally
                  it has one extra element which contains the number of old
                  elements, so it's OK to read one past the end.
  */
  Array1<int32_t> &New2Old() {
    NVTX_RANGE(K2_FUNC);
    if (!new2old_.IsValid()) ComputeNew2Old();
    return new2old_;
  }

  /*
    Return a mapping from new index to old index, with one extra element
    containing the total number of kept elements if extra_element == true.
  */
  Array1<int32_t> New2Old(bool extra_element) {
    Array1<int32_t> &new2old_part = New2Old();
    if (!extra_element) {
      return new2old_part;
    } else {
      // This is a little perverse, using low-level interfaces to increase the
      // dimension of the array; but we know it does have one more element.
      // Because we normally use New2Old() with no arg (equivalent to false),
      // the overloaded version of this function returns a reference for
      // efficiency.
      return Array1<int32_t>(new2old_part.Dim() + 1,
                             new2old_part.GetRegion(), 0);
    }
  }

  /* Return a mapping from old index to new index. This is created on demand
     (must only be called after the Keep() array has been populated).

       @param [in] extra_element  If true, will return the array of size
                  NumOldElems() + 1, which includes one more element;
                  otherwise it will return an array of size NumOldElems().


       @return    Returns an array mapping the old indexes to the new indexes.
                  This array is just the exclusive sum of Keep().
                  It gives the mapping for indexes that are kept; element
                  i is kept if `Old2New()[i+1] > Old2New()[i]`.
  */
  Array1<int32_t> Old2New(bool extra_element = false) {
    NVTX_RANGE(K2_FUNC);
    if (!old2new_.IsValid()) ComputeOld2New();
    if (extra_element) return old2new_;

    return old2new_.Arange(0, old2new_.Dim() - 1);
  }

 private:
  void ComputeOld2New();
  // ComputeNew2Old() also computes old2new_ if needed.
  void ComputeNew2Old();

  Array1<char> keep_;  // array of elements to keep; dimension is the
                       // `num_old_elems` provided in the constructor but it
                       // was allocated with one extra element.
  Array1<int32_t> old2new_;  // note: dimension is num-old-elems + 1.
  int32_t num_new_elems_;  // equals last element of old2new_; set when
                           // old2new_ is created.
  Array1<int32_t> new2old_;
};

// returns a Renumbering object that is the identity map.  Caution; its Keep()
// elements are not set up.
Renumbering IdentityRenumbering(ContextPtr c, int32_t size);


/**
   GetNew2Old() is an alternative to a Renumbering object for cases where the
   temporary arrays it uses might exhaust GPU memory (i.e. when num_old_elems
   might be extremely large, like more than a million).

      @param [in] c              Context to use
      @param [in] num_old_elems  Number of elements that we're computing
                      a subset of.  Presumably this might potentially be
                      quite large, otherwise it would probably be easier
                      to use a Renumbering object.
     @param [in] lambda  A __host__ __device__  lambda of type like:
                          "bool keep_this_elem(int32_t index);"
                      that says whether to keep a particular array element.
     @param [in,optional] max_array_size  The chunk size to be used;
                      an array containing this many int32_t's will be
                      allocated.
     @return  Returns an Array1<int32_t> that maps from the new indexes
                      (i.e. those indexes we're keeping) to the old indexes.
                      Contains elements 0 <= ans[] < num_old_elems.
 */
template <typename LambdaT>
__forceinline__ Array1<int32_t> GetNew2Old(
    ContextPtr c, int32_t num_old_elems, LambdaT &lambda,
    int32_t max_array_size = (1 << 20)) {
  NVTX_RANGE(K2_FUNC);
  int32_t num_arrays = (num_old_elems + max_array_size - 1) / max_array_size;
  std::vector<Array1<int32_t> > new2old(num_arrays);
  for (int32_t i = 0; i < num_arrays; i++) {
    int32_t old_elem_start = i * max_array_size,
        this_array_size = std::min<int32_t>(max_array_size,
                                            num_old_elems - old_elem_start);
    Renumbering renumbering(c, this_array_size);
    char *subset_keep = renumbering.Keep().Data();

    K2_EVAL(c, this_array_size, lambda_offset, (int32_t i) {
      subset_keep[i] = lambda(i + old_elem_start);
      });
    new2old[i] = renumbering.New2Old();
  }
  if (num_arrays == 1) {
    return new2old[0];
  } else {
    Array1<int32_t> offsets = Arange(c, 0, num_arrays * max_array_size,
                                     max_array_size);
    std::vector<const Array1<int32_t>* > new2old_ptrs(num_arrays);
    for (int32_t i = 0; i < num_arrays; i++)
      new2old_ptrs[i] = &(new2old[i]);
    return CatWithOffsets(offsets, new2old_ptrs.data());
  }
}



/**
   GetNew2OldAndRowIds() is a utility for a specific programming pattern that
   you might want to use when you need to prune indexes associated with a
   row_ids array that is too large to fit in memory.

     @param [in] row_splits   A row_splits vector from which we'll be
                computing (in chunks) a row_ids vector and then keeping
                selected elements of the row_ids vector (indexed
                by a new2old array).
     @param [in] num_elems   This should equal row_splits.Back().  It's
                supplied to this function in case you already obtained it
                for another purpose (since d2h transfer is slow).
     @param [in] lambda   Lambda that returns true for elementrs that
                are to be kept.  It should be of type:
                 [=] __host__ __device__ lambda(int32_t i, int32_t row) -> bool;
                where i is the element index with 0 <= i < row_splits.Back(),
                and row is the index of the row to which this element belongs,
                with 0 <= row < row_splits.Back() - 1.
     @param [out] new2old_out  The new2old vector will be output to here;
                it is as if the return status of the lambda for each i
                was assigned to the Keep() vector of a Renumbering
                object and you obtained the New2Old() vector of the
                Renumbering object.
     @param [out] new_row_ids_out  The row_ids, subsampled with the
               renumbering, will be written to here.  It's as if
               you got the row_ids corresponding to `row_splits` and
               did: `*new_row_ids_out = row_ids[*new2old_out]`.
     @param [in,optional] max_array_size  The chunk size to be used;
                arrays containing (up to) approximately this many int32_t's will be
                allocated, assuming `row_splits` is fairly evenly
                distributed.
*/
template <typename LambdaT>
__forceinline__ void GetNew2OldAndRowIds(
    Array1<int32_t> &row_splits,
    int32_t num_elems,
    LambdaT &lambda,
    Array1<int32_t> *new2old_out,
    Array1<int32_t> *new_row_ids_out,
    int32_t max_array_size = (1 << 20)) {
  NVTX_RANGE(K2_FUNC);
  ContextPtr c = row_splits.Context();
  if (num_elems == 0) {
    *new2old_out = Array1<int32_t>(c, 0);
    *new_row_ids_out = Array1<int32_t>(c, 0);
    return;
  }
  // After determining `num_arrays` based on the number of elements,
  // we divide the rows of `row_splits` evenly (except for possibly
  // the last piece, which may be shorter).
  int32_t num_arrays = (num_elems + max_array_size - 1) / max_array_size,
            num_rows = row_splits.Dim() - 1,
       num_rows_part = (num_rows + num_arrays - 1) / num_arrays;
  // the following fixes a kind of edge case that can happen when
  // num_rows_part is less than num_arrays.
  if ((num_arrays - 1) * num_rows_part >= num_rows)
    num_arrays--;

  Array1<int32_t> elem_starts;
  Array1<int32_t> elem_starts_cpu;
  int32_t *elem_starts_cpu_data = nullptr;

  if (num_arrays > 1) {
    elem_starts = Array1<int32_t>(c, num_arrays);
    int32_t *elem_starts_data = elem_starts.Data(),
         *row_splits_data = row_splits.Data();
    K2_EVAL(c, num_arrays, lambda_set_elem_starts, (int32_t i) {
        elem_starts_data[i] = row_splits_data[i * num_rows_part];
      });
    elem_starts_cpu = elem_starts.To(GetCpuContext());
    elem_starts_cpu_data = elem_starts_cpu.Data();
  }

  std::vector<Array1<int32_t> > new2old(num_arrays),
      row_ids(num_arrays);

  for (int32_t i = 0; i < num_arrays; i++) {
    int32_t this_row_start, this_num_rows, this_elem_start, this_num_elems;
    if (num_arrays == 1) {  // we can avoid d2h transfer
      this_row_start = 0;
      this_num_rows = num_rows;
      this_elem_start = 0;
      this_num_elems = num_elems;
    } else {
      this_row_start = i * num_rows_part;
      this_num_rows = std::min<int32_t>(num_rows - this_row_start,
                                        num_rows_part);
      this_elem_start = elem_starts_cpu_data[i];
      this_num_elems = (i + 1 < num_arrays ?
                        elem_starts_cpu_data[i + 1] : num_elems)
                       - this_elem_start;
    }

    Renumbering renumbering(c, this_num_elems);
    char *subset_keep = renumbering.Keep().Data();
    Array1<int32_t> row_splits_part = row_splits.Range(this_row_start,
                                                       this_num_rows + 1);
    Array1<int32_t> row_ids_part(c, this_num_elems);
    if (i == 0) {
      RowSplitsToRowIds(row_splits_part, &row_ids_part);
    } else {
      RowSplitsToRowIdsOffset(row_splits_part, &row_ids_part);
    }
    const int32_t *row_ids_part_data = row_ids_part.Data();

    K2_EVAL(c, this_num_elems, lambda_offset, (int32_t i) {
        subset_keep[i] = lambda(i + this_elem_start,
                                row_ids_part_data[i] + this_row_start);
      });
    new2old[i] = renumbering.New2Old();
    row_ids[i] = row_ids_part[new2old[i]];
  }
  if (num_arrays == 1) {
    *new2old_out = new2old[0];
    *new_row_ids_out = row_ids[0];
  } else {
    std::vector<const Array1<int32_t>* > new2old_ptrs(num_arrays),
        row_ids_ptrs(num_arrays);
    for (int32_t i = 0; i < num_arrays; i++) {
      new2old_ptrs[i] = &(new2old[i]);
      row_ids_ptrs[i] = &(row_ids[i]);
    }
    *new2old_out = CatWithOffsets(elem_starts, new2old_ptrs.data());
    Array1<int32_t> row_offsets = Arange(c, 0, num_arrays * num_rows_part,
                                         num_rows_part);
    *new_row_ids_out = CatWithOffsets(row_offsets, row_ids_ptrs.data());
  }
}

}  // namespace k2

#endif  // K2_CSRC_ALGORITHMS_H_
