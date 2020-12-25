/**
 * @brief
 * ragged_utils
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_RAGGED_UTILS_H_
#define K2_CSRC_RAGGED_UTILS_H_

#include <utility>
#include <vector>

#include "k2/csrc/algorithms.h"
#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/utils.h"

// ragged_utils.h is intended for operations that are somewhat internal to the
// ragged implementation, and not user-facing-- for example, operations that are
// limited to a RaggedShape with NumAxes() == 2 (i.e. a single ragged axis).
namespace k2 {

/*
  This checks that the RaggedShapeLayer indexed `axis`, i.e. src.Axes()[axis],
  is the same for all sources (i.e. represents the same ragged shape).

  This function may do a quick or more thorough check depending on the debug
  level k2 was compiled with; it is intended for situations where it's required
  for the axes to be the same and the program needs to exit or raise an
  exception if not.

  Note: you can use the function Equal(RaggedShape&, RaggedShape&) if you
  want to check the return status directly.

    @param [in] layer       Layer to check, e.g. 0; must satisfy
                           `0 <= axis < src[0]->NumAxes() - 1.`
    @param [in] num_srcs   Length of the list of sources.
    @param [in] srcs      The sources to check

 */
void CheckLayerEqual(int32_t layer,
                     int32_t num_srcs,
                     RaggedShape **src);


/*
   Stacking operation on a single layer of an array of RaggedShape sources,
   appending inthe order "all rows for source 0" then "all rows for source 1".

   (Check the usage message carefully, because this interface is not very
   intuitive).

      @param [in] layer      Layer to operate on, viewed as index into src[i]->Layers().
                             Must satisfy 0 <= axis < src[0]->NumAxes()  - 1.
      @param [in] num_srcs   Number of sources to append/merge (size of array `src`)
      @param [in] src        Array of sources, must have the same device and num-axes.
      @param [out,optional]  merge_map    If not nullptr, will be set to an array
                             that indicates the source of each element on axis
                             `axis+1`, with merge_map->Dim() == the sum of
                             src[i]->TotSize(axis+1).  If `m = (*merge_map)[i]`,
                             then `m % num_srcs` indicates the source for this
                             item and `m / num_srcs` indicates the position of this
                             item within its source.

      @return               Return a RaggedShape with `NumAxes() == 2`, i.e. `Layers().size() == 1`,
                            that is the result of appending the sources together; its
                            TotSize(0) will be the sum of src[i]->TotSize(axis),
                            and its TotSize(1) will be the sum of src[i]->TotSize(axis+1).
                            Its order will be: all elements of *src[0], all elements of
                            *src[1], and so on.

    EXAMPLE: suppose num_srcs == 2, and axis == 0, and src[0]->NumAxes() == 2.
    And suppose *src[0] == [ x x x ] [ x ] and *src[1] = [ ] [ x x ].  Then
    ans == [ x x x ] [ x ] [ ] [ x x ], and merge_map (if requested) will
    be [ 0 2 4 6 1 3 ].
 */
RaggedShape AppendRaggedLayer(int32_t axis,
                              int32_t num_srcs,
                              RaggedShape **src,
                              Array1<uint32_t> *merge_map = nullptr);



/*
   Intersperses rows from a single layer of an array of RaggedShape sources,
   using the order: Row 0 of source 0; row 0 of source 1, etc.,
   i.e. row 0 of all sources, then row 1 of all sources.

   (Check the usage message carefully, because this interface is not very
   intuitive).

      @param [in] layer      Layer to operate on, viewed as index into src[i]->Layers().
                             Must satisfy 0 <= layer < src[0]->NumAxes()  - 1.
      @param [in] num_srcs   Number of sources to intersperse (size of array `src`).
                             Must be >= 1 (otherwise there would be no way to
                             determine the context).
      @param [in] src        Array of sources; must have the same device and num-axes,
                             and src[i]->TotSize(layer) must be the same for all
                             i.
      @param [out,optional]  merge_map    If not nullptr, will be set to an array
                             that indicates the source of each element of this
                             layer, with merge_map->Dim() == the sum of
                             src[i]->TotSize(layer+1).  If `m = (*merge_map)[i]`,
                             then `m % num_srcs` indicates the source for this
                             element and `m / num_srcs` indicates the position of this
                             element within its source.

      @return               Return a RaggedShape with `NumAxes() == 2`, i.e. one layer,
                            that is the result of appending the
                            sources together; its
                            TotSize(0) will be the sum of src[i]->TotSize(layer),
                            i.e. `num_srcs times src[0]->TotSize(layer)` since they
                            are all the same;
                            and its TotSize(1) will be the sum of `src[i]->TotSize(layer+1)`.
                            The rows of the source shape are interspersed.

    EXAMPLE: suppose num_srcs == 2, and layer == 0, and src[0]->NumAxes() == 2.
    And suppose *src[0] == [ x x x ] [ x x ] and *src[1] = [ x ] [ x x x ].  Then
    ans == [ x x x ] [ x ] [ x x ] [ x x x ], and merge_map (if requested) will
    be [ 0 2 4 1 6 8 3 5 7 ].
 */
RaggedShape IntersperseRaggedLayer(int32_t layer,
                                  int32_t num_srcs,
                                   RaggedShape **src,
                                   Array1<uint32_t> *merge_map = nullptr);

/*
  Merge a ragged axis given a 'merge_map' obtained from an operation on a previous
  layer, creating a merge_map for the next layer.

     @param [in] layer     Layer to operate on, viewed as index into src[i]->Layers(),
                           with 0 <= layer < src[i]->NumAxes() - 1
     @param [in] num_srcs  Number of RaggedShapes being merged
     @param [in] src       src   Array of pointers to RaggedShapes to be merged;
                           we will merge the contents of `*src[0]`, `*src[1]` and so on.
     @param [in] merge_map Merge map likely obtained from a previous operation on
                           the same inputs.  merge_map.Dim() must equal the sum
                           of src[i]->TotSize(layer).  If merge_map[i] == m, then
                           we must take the i'th row of this axis from source m
                           % num_srcs, at position within that source equal to  m / num_srcs.
     @param [out] merge_map_out  If not nullptr, will be set to a newly allocated
                          Array1 with Dim() equal to the sum of src[i]->TotSize(layer+1),
                          indicating the sources of the rows of the next layer.

     @return              Return a RaggedShape with `NumAxes() == 2`, i.e. one layer,
                          that is the result of merging layer numbered `layer` of the
                          sources together as dictated by `merge_map`.  Its
                          TotSize(0) will be the sum of src[i]->TotSize(layer),
                          and its TotSize(1) will be the sum of `src[i]->TotSize(layer+1)`.

   See also MergeRagged() in ragged_ops.h, which is supposed to be a more user-facing
   version of this function.
 */
RaggedShape MergeRaggedLayer(int32_t layer,
                             int32_t num_srcs,
                             RaggedShape **src,
                             const Array1<uint32_t> &merge_map,
                             Array1<uint32_t> *merge_map_out = nullptr);

/*
  An operation on one layer of a ragged shape that subsamples the rows.

    @param [in] src   Source RaggedShape; only layer `layer` will be read
                      (viewed as index into src.Layers()).
    @param [in] layer  Layer to subsample; must satisfy 0 <= layer < src.NumAxes() - 1.
    @param [in] subsample_factor  Factor by which we subsample the rows;
                      must be a divisor of src.TotSize(layer).
 */
RaggedShape SubsampleRaggedLayer(RaggedShape &src, int32_t layer,
                                 int32_t subsample_factor);





}  // namespace k2

#define IS_IN_K2_CSRC_RAGGED_UTILS_H_
#include "k2/csrc/ragged_utils_inl.h"
#undef IS_IN_K2_CSRC_RAGGED_UTILS_H_

#endif  // K2_CSRC_RAGGED_UTILS_H_
