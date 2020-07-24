#include "k2/csrc/cuda/array1.h"





// Shape of a 2-dimensional ragged array ( e.g. [ [ 0 ] [ 3 4 1 ] [] [ 5 ] ])
class RaggedShape2 {
  // return dim on 0th axis.
  int32_t Size0() { return row_splits0_.size - 1; }
  Array1Tpl<int32_t> &RowSplits1() { return row_splits1_; }
  Array1Tpl<int32_t> &RowIds1();
  // TODO: make TotSize1() more efficient for GPU vectors via cached_tot_size1_ or row_splits_.size().
  // size1() is the *total* size of dimension one, summed across all rows.
  int TotSize1() { return row_splits1_[-1]; }
  // max_size1() returns the maximum of any element of Sizes1().
  int MaxSize1();

  // Returns a lambda which behaves like a pointer to the data of Sizes1()
  // (except use () not []).  The size of this virtual array is Size0().
  auto Sizes1Data() {
    int *data = this->row_splits1_.data;
    return __host__ __device__ [data] (int i) { return data[i+1]-data[i]; };
  }

  Array1Tpl<int,T> Sizes1();  // Caution: this is slow as it creates a new array.

  template <typename D2>
  RaggedShape2<D2> &CastTo() {
    if (D2 != kUnk) {
      assert(region->device == kUnk ||  // would only happen if empty region
             region->device == D2);
    }
    return reinterpret_cast<RaggedShape2<D2> > (*this);
  }

  ContextPtr &Context() { return row_splits1_.Context(); }

 protected:
  Array1<int> row_splits1_;
  Array1<int> row_ids1_;  // populated on demand.
  int cached_tot_size1_;  // TODO?  can use to avoid gpu access when
};


template <DeviceType D>
class RaggedShape3: public RaggedShape2<D>: {
  Array1<int> &RowSplits2() { return row_splits2_; }
  Array1<int> &RowIds2();

  // Mm, might not use this?
  auto Sizes2Data() {  // sizes of 2nd level of the array, indexable (i,j) ?
    int *data1 = row_splits1_.data, *data2 = row_splits2_.data;
    return __host__ __device__ [data] (int i, int j) {
                                 int idx2 = data1[i];
                                 assert(idx2 + j < data1[i+1]);
                                 return data2[idx2 + 1] - data2[idx2];
                               };
  }

  auto FlatSizes2Data() {  // sizes of 2nd level of the array; an array of
                           // TotSize1().
    int *data = row_splits2_.data;
    return __host__ __device__ [data] (int i) {
                                 return data[i+1] - data[i];
                               }
  };


  int TotSize2() { return row_splits2_[-1]; }
  int MaxSize2();

  ContextPtr &Context() { return row_splits1_.Context(); }
 protected:
  Array1<int> row_splits2_;
  int cached_size2_;  // TODO?  can use to avoid gpu access ..
  Array1<int> row_ids2_;
};



/*
  Create a RaggedShape2 from an array of elems and a array of row-ids
  (which may each element to its corresponding row).  The row-ids must
  be a nonempty vector, nonnegative and no-decreasing.

    @param [in]  num_rows   The number of rows (Size0()) of the object to be created.
                 If a value <= 0 is supplied, it will use row_ids[-1]+1
                 if row_ids.size > 0, else 0.
    @param [in]  row_ids   The row-ids of the elements; must be nonnegative
                 and non-decreasing.
 */
RaggedShape2 RaggedShape2FromRowIds(int num_rows,
                                    const Array<int> &row_ids);



/*
  Create a new RaggedShape3 that is a subsampling
 */
template <typename T>
RaggedShape3<T> SubsampleFromRowIds(const RaggedShape3<D> &src,
                                    const Array1<int> &row_ids);
