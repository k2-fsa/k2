/**
 * Copyright      2022  Xiaomi Corporation (authors: Daniel Povey)
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

#ifndef K2_CSRC_ARRAY_OF_RAGGED_H_
#define K2_CSRC_ARRAY_OF_RAGGED_H_

#include <string>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"
#include "k2/csrc/log.h"

namespace k2 {

/*
  ArrayOfRagged<T> is a 1-dimensional array of Ragged<T>.
  It is intended for situations where you want to do some operations on
  arrays of ragged arrays, without explicitly concatenating them (e.g. to
  save time).   This is a fairly low-level interface, intended to
  be used mostly by CUDA/C++ implementation code.  It is a convenience
  wrapper that saves you the trouble of creating arrays of pointers.
 */


/*
  ArrayOfRaggedShape is a convenience function that gives you easy access
  to pointers-of-pointers for an array of ragged shapes.
 */
class Array1OfRaggedShape {
 public:

  /*
    Constructor.
    Args:
       srcs: pointers to the source shapes, a CPU pointer
      num_srcs: the number of source shapes.  All shapes must have the
              same NumAxes() and must be on the same device.

   TODO: we'll likely, later, add optional args which dictate which of
   the MetaRowSplits() and MetaRowIds() are to be pre-populated; this should
   enable us to save kernels by combining certain operations across the
   axes.

  */
  ArrayOfRaggedShape(RaggedShape *src,
                     int32_t num_srcs);


  int32_t NumSrcs() const;
  int32_t NumAxes() const;

  // Returns device-accessible array of row-splits for the individual shapes,
  // indexed [axis-1][src], with 0 <= src < num_srcs.  The shape of this
  // Array2 is [NumAxes() - 1][NumSrcs()].
  Array2<int32_t*> *RowSplits();

  // Returns device-accessible vector of row-splits for a particular
  // axis, indexed by 0 <= src < num_srcs.
  int32_t **RowSplits(int32_t axis) {
  }

  // Returns device-accessible array of row-ids for the individual shapes
  // indexed [axis-1][src], with 0 <= src < num_srcs.  The shape of this
  // Array2 is [NumAxes() - 1][NumSrcs()].
  Array2<int32_t*> *RowIds();


  // Returns device-accessible vector of row-splits for a particular
  // axis, indexed by 0 <= src < num_srcs.
  int32_t **RowIds(int32_t axis) {
  }


  /* Return the  total size on this axis, which is the sum of the TotSize() of
     the individual shapes.  Requires 0 <= axis < NumAxes() and
     for axis=0 the returned value is the same as Dim0().
  */
  int32_t TotSize(int32_t axis) const;

  // equivalent to TotSize(0).
  int32_t Dim0() const { return TotSize(0); }


  /* Return the device-accessible meta-row-splits, which is the cumulative sum,
     along the src axis, of the tot-sizes of the individual arrays.
     This Array2 is of shape [NumAxes()][NumSrcs() + 1], indexed [axis][src];
     caution, the indexing is different from RowSplits(), there is no offset.
     Also, the meta_row_splits0 is a thing, unlike with regular row-splits
     which start from 1.

     Caution: the lengths of the arrays pointed to by the elements of this
     Array2 (which contains pointers!) are of course all different, and
     these lengths are currently only available

     Implementation note: we can probably just populate this on CPU and transfer
     to GPU, this will be faster than invoking an extra kernel in normal cases
     when the NumSrcs() is small.  [Also: see GetRowInfoMulti()].
   */
  Array2<int32_t> MetaRowSplits();

  // could POSSIBLY add this so this code could be used in functions like Stack().
  // would be like MetaRowSplits but with an extra 1st row containing 0,1,2,...
  // We could perhaps create it with 1 extra initial row so this is always
  // convenient to output.
  Array2<int32_t> Offsets();

  /*
    Returns the meta-row-splits for a particular axis, with 0 <= axis < NumAxes();
    this is the cumulative sum of the TotSize(axis) for all of the sources,
    with MetaRowSplits(axis).Dim() == NumSrcs() + 1.

    Note: in ragged_opts.cu we refer to this as composed_row_splits
  */
  Array1<int32_t> MetaRowSplits(int32_t axis);

  /* Return the device-accessible meta-row-ids, which are the row-ids corresponding
     to MetaRowSplits(); this tells us, for indexes into the appended/concatenated
     array, which source array they belong to, i.e. elements are in [0,NumSrcs()-1].

     This cannot be an Array2 because unlike the MetaRowSplits(), all the row-ids
     arrays are of different lengths.

     Note: in ragged_ops.cu we refer to this as composed_row_ids.
  */
  Array1<int32_t*> MetaRowIds();

  /*
    Returns the meta-row-ids for a particular axis, with 0 <= axis < NumAxes();
    this is the row-ids corresponding to MetaRowSplits(axis), and its elements
    gives, for indexes into the concatentated shape (concatenated on axis 0),m
    which source they come from.  E.g. element 100 of MetaRowIds(2)
    would tell us which source an idx012 with value 100 into axis 2 of
    concatenated array would come from.
  */
  Array1<int32_t> MetaRowIds(int32_t axis);
};



template <typename T>
struct Array1OfRagged {

  Array1OfRaggedShape shape;

  // Array of the individual values pointers of the source arrays, indexed by
  // shape
  Array1<T*> values;

  int32_t NumSrcs() { return values.Dim(); }

  ArrayOfRagged(Ragged<T> *srcs,
                int32_t num_srcs);

}



}  // namespace k2

#endif  // K2_CSRC_ARRAY_OF_RAGGED_H_
