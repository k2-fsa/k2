// k2/csrc/cuda/ragged.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/cuda/ragged.h"

namespace k2 {

RaggedShapeFromTotSizes(ContextPtr &c, int32_t num_axes, int32_t *tot_sizes) {
  std::vector<RaggedShapeDim> axes(num_axes - 1);
  // In future we might choose to allocate everything in one big array, to avoid
  // multiple allocations, but for now just do it the simple way.
  for (int32_t axis = 1; axis < num_axes; axis++) {
    axes[axis-1].row_splits = Array1<int32_t>(c, tot_sizes[axis - 1] + 1);
    axes[axis-1].row_ids = Array1<int32_t>(c, tot_sizes[axis] + 1);
    axes[axis-1].cached_tot_size = tot_sizes[axis];
  }
  return RaggedShape(axes);
}


RaggedShape RaggedShape2(Array1<int32_t> *row_splits,
                         Array1<int32_t> *row_ids,
                         int32_t cached_tot_size) {
  if (!row_splits && !row_ids) {
    LOG(FATAL) << "At least one of row_splits and row_ids must be defined";
  }
  if (cached_tot_size != -1) {
    if (row_ids != nullptr)
      CHECK(cached_tot_size == row_ids->Size()-1);
    if (row_splits != nullptr)  // caution: next check may be slow...
      CHECK(cached_tot_size == row_splits[row_splits->Size()-1]);
  }
  axes_.resize(1);
  if (row_splits)
    axes_[0].row_splits = *row_splits;
  if (row_ids)
    axes_[0].row_ids = *row_ids;
  axes_[0].cached_tot_size = cached_tot_size;
}

RaggedShape ComposeRaggedShapes(RaggedShape &a,
                                RaggedShape &b) {
  if (a.NumElements() != b.Dim0()) {
    LOG(FATAL) << "ComposeRaggedShapes: shape mismatch: "
               << a.NumElements() << " vs. " << b.Dim0();
  }
  std::vector<RaggedShapeDim> axes(a.axes_.size() + b.axes_.size());
  size_t a_size = a.axes_.size(), b_size = b.axes_.size();
  for (size_t i = 0; i < a_size; i++)
    axes[i] = a.axes_[i];
  for (size_t i = 0; i < b_size; i++)
    axes[i + a_size] = b.axes_[i];
  return RaggedShape(axes);
}


RaggedShape RaggedShape3(Array1<int32_t> *row_splits1,
                         Array1<int32_t> *row_ids1, int32_t cached_tot_size1,
                         Array1<int32_t> *row_splits2,
                         Array1<int32_t> *row_ids2, int32_t cached_tot_size2) {
  // This is a slightly lazy implementation, could save a couple copies of metadata by
  // implementing it directly.
  return ComposeRaggedShapes(RaggedShape2(row_splits1, row_ids1, cached_tot_size1),
                             RaggedShape2(row_splits2, row_ids2, cached_tot_size2));
}


// this is for transfer to GPU, it is the meta-info for one axis of a sparse array.
struct RowPtrs {
  int32_t *row_splits;
  int32_t *row_ids;
  int32_t num_rows;
  int32_t num_elems;
};

/*
  Extract meta-info from the shape (this will include populating any
  row_ids and row_splits that were not already populated).  This is used
  inside algorithms when we need to transfer meta-info to GPU.

     @param [in] shape   Ragged shape that we're extracting meta-info from
     @param [in] zeros   Vector of dimension at least 1 + shape.Dim0(),
                         used for the RowPtrs info for axis 0 of the output
                         which contains all zeros.
     @param [out] ptrs   Host (i.e. CPU memory) array of RowPtrs that we're writing to; must
                         be the start of an array of length shape.NumAxes().
                         (each element refers in some sense to the next axis).
*/
void GetRowPtrs(RaggedShape &shape, const Array1<int32_t>&zeros, RowPtrs *ptrs) {
  // TODO (dan)
}


/*
  Returns a CPU array of shape src[0]->NumAxes() by (src_size + 1), where each
  row is the exclusive-sum of the TotSize() of the respective sources, on that
  axis.

     @param [in] src_size  The number of `RaggedShape`s in `src`
     @param [in] src    The shapes whose sizes we want.  Must all have the
                      same NumAxes().
     @return   Returns a freshly allocated Array2<int32_t> of dimension
               src[0]->NumAxes() by (src_size + 1), where each
               row is the exclusive-sum of the TotSize() of the respective
               sources, on that axis.  Its last column contains the totals.

 */
inline Array2<int32_t> GetOffsets(int32_t src_size, RaggedShape **src) {
  //  src_offsets[i,j]  == src_offsets.Data()[i*num_axes_in + j] contains:
  //          sum(0 <= k < i) src[k]->TotSize(j).
  int23_t num_axes_in = src[0]->NumAxes()
  Array2<int32_t> src_offsets(CpuContext(), src_size + 1, num_axes_in);
  int32_t *src_offsets_data = src_offsets.Data();
  int32_t src_offsets_stride0 = src_size + 1;
  DCHECK_EQ(src_offsets.Stride0(), src_offsets_stride0);

  for (int32_t axis = 0; axis  < num_axes_in; axis++) {
    int32_t sum = 0;
    for (int32_t i = 0; i <= src_size; i++) {
      src_offsets_data[i*src_offsets_stride0 + axis] = sum;
      if (i < src_size)
        sum += src->TotSize(axis);
    }
  }
  return src_offsets;
}


struct RowPtrsWithOffsets {
  int32_t *row_splits;
  int32_t *row_ids;
  int32_t num_rows;
  int32_t num_elems;
  int32_t row_splits_offset;
  int32_t row_ids_offset;
};



RaggedShape Stack(int32_t axis, int32_t src_size, RaggedShape **src) {
  CHECK_GT(src_size, 0);
  ContextPtr c = src[0]->GetContext();
  int32_t num_axes_in = src[0]->NumAxes(),
      max_src_dim0 = src_size;  // will be the maximum of (src_size, and the
                                // largest num-axes on any input; it's the size
                                // of an array of zeros that we'll need for a
                                // couple purposes.
  for (int32_t i = 1; i < src_size; i++) {
    // Check they have the same num-axes.
    CHECK_EQ(num_axes_in, src[i]->NumAxes());
    CHECK(IsCompatible(*src[0], *src[i]));
    if (src[i]->Dim0() > max_src_dim0)
      max_src_dim0 = src[i]->Dim0();
  }

  // This could definitly be more efficient on the CPU side.  TODO(dan): make a separate
  // CPU implementation.

  // This array will be transferred to the GPU where it will contain the
  // pointers to the row_splits and row_ids of the source arrays,
  // and their sizes.
  Array2<RowPtrs> src_row_ptrs(CpuContext(), src_size, num_axes_in);
  auto src_row_ptrs_acc = src_row_ptrs.Accessor();
  RowPtrs *src_row_ptrs_data = src_row_ptrs.Data();
  Array1<int32_t> zeros(c, max_src_dim0 + 1, 0);
  for (int32_t i = 0; i < src_size; i++) {
    GetRowPtrs(*(src[i]), zeros, &(src_row_ptrs_acc(i, 0)));
  }

  // offsets has shape  (src[0]->NumAxes(), (src_size + 1))
  Array2<int32_t> offsets = GetOffsets(src_size, src);
  auto offsets_acc = offsets.Accessor();

  std::vector<int32_t> tot_sizes_out(num_axes_in + 1);
  tot_sizes_out[0] = src_size;
  for (int32_t axis = 0; axis < num_axes_in; axis++)
    tot_sizes_out[axis + 1] = offsets_acc(axis, src_size);

  RaggedShape ans = RaggedShapeFromTotSizes(c, tot_sizes_out);
  Array1<RowPtrs> dest_row_ptrs(CpuContext(), num_axes_in);
  GetRowPtrs(ans, dest_row_ptrs.Data());

  if (c.DeviceType() != kCpu) {
    offsets = offsets.To(c);
    dest_row_ptrs = dest_row_ptrs.To(c);
    src_row_ptrs = src_row_ptrs.To(c);
  }

  RowPtrs *dest_row_ptrs_data = dest_row_ptrs.Data();
  auto src_row_ptrs_acc = src_row_ptrs.Accessor();
  offsets_acc = offsets.Accessor();

  // ParallelRunner will eventually allow us to run things in child streams.
  ParallelRunner pr(c);
  // `axis` here corresponds to the axis on the input side, or the axis on the
  // output side minus one (i.e. the index into the axes_ member of Ragged).

  std::vector<cudaStream_t> streams(num_axes_in + 1);
  int32_t num_jobs = src_size * 2;
  Array2<TaskRedirect> task_redirects(c, num_axes_in + 1, num_jobs);
  auto task_redirects_acc = task_redirects.Accessor();
  // populate task_redirects (these allocate blocks of threads roughly
  // proportionally to the amount of data to process from this source.
  for (int32_t axis = 0; axis <= num_axes_in; axis++) {
    streams[axis] = pr.NewStream();

    const float *offsets = (axis == 0 ? zeros.Data() :
                            &(offsets_acc(axis - 1, 0)));

    GetTaskRedirect(stream[axis], src_size, offsets,
                    &(task_redirects_acc(axis, 0)));
  }

  for (int32_t axis = 0; axis < num_axes_in; axis++) {
    RowPtrs *dest_info = dest_row_ptrs_data + axis,
        *src_info = src_row_ptrs_data + axis;

    {
      // first do the row-splits.
      TaskRedirect *tr = &(task_redirects_acc(axis, 0));
      cudaStream_t stream = streams[axis];

      auto lambda_set_prev_row_ids row_splits = [=] __host__ __device__ (
          int32_t src_idx, int32_t num_threads, int32_t thread_idx) -> void {
          RowPtrs src_row = *src_info, dest_row = *dst_info;
          int32_t num_rows = src_info->num_rows;
          const int32_t *src_row_splits = src_info->row_splits;
          int32_t *dest_row_splits = dest_info->row_splits;
          int32_t offset = offsets_acc(axis, src_idx);
          // HERE...
              next_offsets
          for (; thread_idx < num_rows; thread_idx += num_threads) {
          }
      };



    RowPtrs *dest_row = dest_row_ptrs_data + axis + 1;

    int32_t num_rows = tot_sizes_out[axis];
    // Below, the `+ 1` is to ensure we have at least one block.  This block size
    // won't necessarily correspond to the CUDA block size.  We assign one
    // block to each source array.  If one of the sources is significantly
    // larger than the average it will loop in the kernels, which of course
    // is not ideal.
    int32_t min_block_size = 32,
        target_loops_per_block = 2,
        block_size = (min_block_size * (num_rows / (num_jobs * target_loops_per_block * min_block_size))) + 1;

      auto lambda_set_row_splits = [=] __host__ __device__ (int32_t job, int32_t j) -> void {
         TaskRedirect tr = tr_data[job];
         int32_t s = tr.task_id;  // the source, 0 <= s < src_size
         int32_t num_threads = block_size * tr.num_jobs_this_task,
             this_thread = j + block_size * tr.job_idx;

    // Handle the row-ids for this output axis.


    if (axis > 0) {  // TODO: handle row splits for axis == 0.

      auto lambda_set_row_splits = [=] __host__ __device__ (int32_t job, int32_t j) -> void {
         TaskRedirect tr = tr_data[job];
         int32_t s = tr.task_id;  // the source, 0 <= s < src_size
         int32_t num_threads = block_size * tr.num_jobs_this_task,
             this_thread = j + block_size * tr.job_idx;
         // `num_threads` is the total number of threads that are processing
         // this source-index, 0 <= this_thread < num_threads is the index
         // within those threads.


         // s is the member of `src` we're processing, 0 <= a < src_size.
         // 0 <= j < block_size is the number of threads processing this source
         // array.
         // 1 <= axis < num_axes_in (captured) is the output axis we're getting the
         //       row_split for; the corresponding input axis is axis-1.
         RowPtrs dest_row_ptrs = dest_row_ptrs_data[axis];
         RowPtrs src_row_ptrs = src_row_ptrs_acc(s, axis - 1);
         int32_t dest_offset_begin = offsets_acc(axis, s),
             dest_offset_end = offsets_acc(axis, s + 1);
         assert(dest_offset_end - dest_offset_begin == src_row_ptrs.num_rows);
         for (; j < src_row_ptrs.num_rows; j += block_size) {

         }



      };
      int32_t *offsets = &(offsets_acc(i,0));
      int32_t row = RowIdFromRowSplits(src_size, offsets,
                                       i, num_rows + 1);

      pr.Eval(num_jobs, num_rows + 1, lambda_set_row_splits);


    }

    pr.Eval()


  }


  // What we mean by 'alldim' is the row-splits/row-ids of `ans` if we were to
  // collapse the dimensions of `ans` so its Dim0() is src_size and its
  // NumElements() is unchanged.  So row_splits_alldim points to an array
  // of size src_size+1 and row_ids_alldim points to an array of size
  // (sum of NumElements() in source arrays) + 1.
  // Note: `row_ids_alldim` is a temporary repurposing of the final
  // row_ids of the output array.  We'll overwrite it in the kernel.
  int32_t *row_splits_alldim = &(offsets_acc(num_axes_in - 1,0));
  int32_t *row_ids_alldim = ans.RowIds(num_axes_in);

  int32_t *ans.RowSplits0().Data();





  //  src_offsets[i,j]  == src_offsets.Data()[i*num_axes_in + j] contains:
  //          sum(0 <= k < i) src[k]->TotSize(j).
  // It is used to find the appropriate parts of the output row_splits and
  //  row_ids to write to, for each axis.
  // It also gives us the sizes of the arrays we need to allocate.
  Array2<int32_t> src_offsets(CpuContext(), src_size + 1, num_axes_in);
  int32_t *src_offsets_data = src_offsets.Data();
  int32_t src_offsets_stride0 = src_offsets.Stride0();

  // The format is: row_splits1 row_ids1 row_splits2 row_ids2 ...
  // note: the number of these equals num_axes_out -1 == num_axes_in.
  Array1<int32_t*> row_splits_and_ids(CpuContext(), 2 * num_axes_in);
  int32_t **row_splits_and_ids_data = row_splits_and_ids.Data();

  // Set src_offsets and src_row_splits_and_ids.
  for (int32_t axis = 0; axis  < num_axes_in; axis++) {
    int32_t sum = 0;
    for (int32_t i = 0; i <= src_size; i++) {
      src_offsets_data[i*src_offsets_stride0 + axis] = sum;
      if (i < src_size) {
        sum += src->TotSize(axis);
        if (axis > 0) {
          src_row_splits_and_ids_data[i*src_row_splits_and_ids_stride0 + 2*(axis-1)] = src->RowSplits(axis);
          src_row_splits_and_ids_data[i*src_row_splits_and_ids_stride0 + 2*(axis-1) + 1] = src->RowIds(axis);
        }
      }
    }
  }



  std::vector<int32> row_splits0_vec(src_size + 1);
  // std::vector<Array1<int32_t>* >...
  int32_t sum = 0;
  for (int32_t i = 0; i < src_size; i++) {
    row_splits0_vec[i] = sum;
    sum += src[i]->Dim();
  }
  Array1<int32_t> row_splits0(c, row_splits0_vec);
  Array1<int32_t> row_ids0(c, sum + 1);
  // NOTE: the following is not nearly as efficient as it could be, e.g. we
  // could avoid any reduction operation if the max size (of any element of the
  // sum) was not too much larger than the average.  Can optimize later.
  RowSplitsToRowIds(c, src_size, row_splits0.Data(),
                    sum + 1, row_ids0.Data());




}



RaggedShape4 MergeToAxis1(const std::vector<const RaggedShape3*> &src) {
  // TODO, check this.
  assert(src.size() != 0);
  Context c = src[0]->Context();

  RaggedShape3 temp = MergeToAxis1(reinterpret_cast<const std::vector<const RaggedShape2*> &>(src));

  Array1<int32_t*> src_row_splits1(c_cpu, n), src_row_splits2(c_cpu, n);
  // init arrays, convert to device tensors.

  int32_t **src_row_splits1_data = src_row_splits1.Data(),
      **src_row_splits2_data = src_row_splits2.Data();

  const int32_t *row_splits1_data = temp.RowSplits1().Data(),
      *row_ids1 = temp.RowIds1().Data(),
      *row_splits2 = temp.RowSplits2().Data(),
      *row_ids2 = temp.RowIds2().Data();

  // row_splits3_out_01 will be indexed by an idx01 of the output array, but actually
  // contains row-indexes on axis 3 (rather than on axis 2, which
  // something indexed by an idx01 normally would).  So it's like
  // idx01xx = row_splits3_out_01[idx01].
  // We want to do the
  // exclusive-sum on the smallest size we can, and are using the fact that
  // within given indexes 0,1 of the output, the input and output layouts are
  // the same.
  Array1<int32_t> row_splits3_out_01(temp.TotSize1() + 1);
  // sorry this naming is weird, but they are the sizes on axis 3, but the
  // row-splits for axis 2 (i.e. indexed by an idx012).
  int32_t *sizes3_out_data = row_splits3_out_01.Data();

  __host__ __device__ lambda_set_sizes3 = [=] (int output_idx01) -> void {
     int32_t output_idx0 = row_ids1_data[output_idx01],
         output_idx0x = row_splits0_data[output_idx0],
         output_idx1 = output_idx01 - output_idx0x;

     int32_t *input_row_splits1 = src_row_splits1_data[output_idx1],
         *input_row_splits2 = src_row_splits2_data[output_idx1];

     int32_t input_idx0xx = input_row_splits2[input_row_splits1[output_idx0]],
         input_idx0xx_next = input_row_splits2[input_row_splits1[output_idx0 + 1]],
         input_size0xx = input_idx0xx_next - input_idx0xx;
     // input_size0xx is the total size on axis 2, but spanning one element on
     // axis 0, including whatever elements there are on axis1.  So the
     // num-elements in a list of lists.

     // Index 1 in the output is known as well; it corresponds to the index into 'src'.
     // So this is the total size on axis 3 of the output, given values on axes 0 and
     // 1 (but not 2; span all) of the input.
     int32_t output_size01xx = input_size0xx;
     // the size on axis 2 of the input becomes the size on axis 3 of the
     // output.
     sizes3_out_data[output_idx01] = output_size01xx;
  };
  Eval(c, temp.TotSize1(), lambda_set_sizes3);
  ExclusiveSum(row_splits3_out_01, &row_splits3_out_01);
  // the entries of row_splits3_out_01 would be written as output_idx01x,
  // i.e. they have the magnitude of an output_idx012 but where the last index
  // (axis 2) is always zero so we write x.
  int32_t *row_splits3_out_01 = row_splits3_out_01.Data();

  int32_t row_splits3_out_size = temp.TotSize2() + 1;
  Array1<int32_t> row_splits3_out(c, row_splits3_out_size);

  int32_t *row_splits3_out_data = row_splits3_out.Data();

  __host__ __device__ lambda_set_row_splits3 = [=] (int output_idx012) -> void {
     // 'offset' below is to avoid reading invalid memory, one past the end of
     // row_ids2_data; it'll still do the right thing.
     int32_t offset = (output_idx012 == row_splits3_size-1 ? -1 : 0);

     int32_t output_idx01 = row_ids2_data[output_idx012 + offset],
         output_idx0 = row_ids1_data[output_idx01],
         output_idx0x = row_splits1_data[output_idx0],
         output_idx1 = output_idx01 - output_idx0x,
         output_idx01x = row_splits2_data[output_idx01],
         output_idx2 = output_idx012 - output_idx01x;

     // Note: axis 0 of the output corresponds to axis 0 of the input, but axes
     // 2 and 3 of the output correspond to axes 1 and 2 of the input
     // respectively (axis 1 of the output corresponds to the index into 'src').
     int32_t *input_row_splits1 = src_row_splits1_data[output_idx1],
         *input_row_splits2 = src_row_splits2_data[output_idx1];

     int32_t input_idx_0x = input_row_splits0[output_idx0],
         input_idx01 = input_idx0x + output_idx2,
         input_idx01x = input_row_splits2[input_idx01],
         input_idx0xx = input_row_splits2[input_idx0x],
         input_idxx1x = input_idx01x - input_idx0xx;

     // index 1 of input becomes index 2 of the output.  the extra "x" is because
     // we inserted output_idx1.  Index  xx2x means it's dimensionally an index into
     // the elems of a Ragged4 array (includes 4 indexes) but it's a difference of
     // such things, namely like an 012x minus an 01xx.
     int32_t output_idxxx2x = input_idxx1x;

     int32_t output_idx01xx = row_splits3_out_data[output_idx01],
         output_idx012x = output_idx01xx + output_idxxx2x;

     row_splits3_out_data[output_idx012] = output_idx012x;
   };
  Eval(c, row_splits3_out_size, lambda_set_row_splits3);
}



  int32_t osize0 = src[0]->Size0(),
      osize1 = src.size();   // note, osize1 is not the same as its TotSize1(),
                             // it is the size of each sub-list.
  // TODO: assert src[n]->Size0() == size0 for all n.

  // tot_size2 and tot_size3 are the total sizes on axes 2 and 3 of the
  // result, corresponding to the totals on axes 1 and 2 of src.
  int32_t tot_size2 = 0, tot_size3 = 0;
  for (int32_t i = 0; i < size1; i++) {
    tot_size2 += src[i]->TotSize1();
    tot_size3 += src[i]->TotSize2();
  }

  // we will transpose these arrays later; it's better for consolidated writes
  // to have it this way around initially.  TODO: need to make sure we can read
  // one past the end.
  // TODO: ensure that memory one past the end is readable.  May be easiest to
  // just ensure that this is the case always, in the constructor of Array2.
  Array2<int32_t> osizes2(c, osize1, osize0),
      tot_osizes3(c, osize1, osize0);
  // Note, we rely on the fact that a freshly-initialized Array2 will be contiguous, so
  // we can index an array x with size (A,B) as x.data()[a*B + b]
  int32_t *osizes2_data = osizes2.data(),
      *tot_osizes3_data = tot_osizes3.data();
  for (int32_t i = 0; i < osize1; i++) {
    RaggedShape3 &shape = *(src[i]);
    int32_t *row_splits1_data = shape.RowSplits1(),
        *row_splits2_data = shape.RowSplits2();
    // the j below is the top-level index0 into the shape; we want the total
    // number of index1's and index2's/elements associated with this index0,
    // which will be written to respectively sizes2_data and sizes3_data.
    auto lambda_get_sizes = __host__ __device__ [=] (int32_t j) -> void {
        int32_t begin1 = row_splits1_data[j], end1 = row_splits1_data[j+1];
        int32_t begin2 = row_splits2_data[begin1], end2 = row_splits1_data[end1];
        int32_t num_indexes1 = end1 - begin1,
            num_indexes2 = end2 - begin2;
        tot_osizes2_data[i * osize0 + j] = num_indexes1;
        tot_osizes3_data[i * osize0 + j] = num_indexes2;
    };
    Eval(c.Child(), osize0, lambda_get_sizes);
  }
  c.WaitForChildren();

  // Transpose those total sizes, so the indexes are the right way
  // around, i.e. [index0,index1].  Note: conversion to Array2 will
  // ensure the result is contiguous.
  Array2<int32_t> osizes2_t(osizes2.ToTensor().Transpose(0, 1)),
      tot_osizes3_t(osizes2.ToTensor().Transpose(0, 1));
  // osizes2_t and tot_osizes3_t are contiguous, so osizes2_t and tot_osizes3_t
  // will share the underlying memory.
  Array1<int32_t> osizes2_t_linear = osizes2_t.Flatten(),
      tot_osizes3_t_linear = osizes3_t.Flatten();
  Array1<int32_t> osizes1(c, osize0, osize1);

  Array1<int32_t> row_splits2_linear(size0*size1 + 1),
      row_splits3_linear(size0*size1 + 1);
  ExclusiveSum(c.Child(), osizes2_t_linear, &row_splits2_linear);
  ExclusiveSum(c.Child(), tot_osizes3_t_linear, &row_splits3_linear);

  // row_splits1_linear has size `size0`.
  Array1<int32_t> row_splits1_linear(
      row_splits2_linear.ToTensor().Range(0, size0, size1));

  c.WaitForChildren();

  return RaggedShape4FromRowSplits(row_splits1_linear,
                                   row_splits2_linear,
                                   row_splits3_linear);

}




}



}  // namespace k2
