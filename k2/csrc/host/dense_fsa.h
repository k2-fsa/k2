/**
 * @brief
 * dense_fsa
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_DENSE_FSA_H_
#define K2_CSRC_HOST_DENSE_FSA_H_

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/log.h"

namespace k2host {

/*
  DenseFsa represents an FSA stored as a matrix, representing something
  like CTC output from a neural net.  `data` is a (T+1) by N
  matrix, where N is the number of symbols (including blank/zero).
  The last row of this matrix contains only zeros; this is where it
  gets the (zero) weight for the final-arc.  It may seem odd to
  actually have to store the zero, but for the autograd to work
  correctly we need all arcs to have an arc-index.

  Physically, we would access weights[t,n] as weights[t * t_stride + n].

  This FSA has T + 2 states, with state 0 the start state and state T + 2
  the final state.  (Caution: if we formulated our FSAs more normally we
  would have T + 1 states, but because we represent final-probs via an
  arc with symbol kFinalSymbol on it to the last state, we need one
  more state).   For 0 <= t < T, we have an arc with symbol n on it for
  each 0 <= n < N, from state t to state t+1, with weight equal to
  weights[t,n].
 */
struct DenseFsa {
  int32_t T;
  int32_t num_symbols;
  int32_t arc_offset;

  const float *data;  // Would typically be a log-prob or unnormalized log-prob

  /*
    The next few functions provide an interface more similar to struct
    Fsa.  We don't necessarily recommending using these functions much;
    it may be more efficient to use what is known about the structure
    of this object.  But they may be useful for documentation and testing.
   */
  int32_t NumStates() const { return T + 2; }
  int32_t ArcIndexes(int32_t state_index) const {
    return arc_offset +
           (state_index <= T ? state_index * num_symbols : T * num_symbols + 1);
  }
  Arc arc(int32_t arc_index) const {
    arc_index -= arc_offset;
    int32_t state_index = (arc_index - arc_offset) / num_symbols;
    return Arc(state_index, state_index + 1,
               (state_index < T ? (arc_index % num_symbols) : kFinalSymbol));
  }

  /* Constructor
     @param [in] T  number of frames / sequence length.  This FSA has T+2
                    states; state T+1 is the final-state, and state T has
                    only a single arc, with symbol kFinalSymbol, with
                    arc-index `arc_index = arc_offset+(T*num_symbols)`,
                    to state T+1; the weight on this is
                    data[arc_index] == 0.

                   All other states t<T have `num_symbols` arcs to
                   state t+1; the arc with symbol s will have arc-index
                   `arc_index = arc_offset+(t*num_symbols)+s`, and weight
                   data[arc_index].
     @param [in] num_symbols   The number of symbols in the vocabulary,
                               including epsilon/blank; equals num-cols
                               of `data` matrix
     @param [in] data   Pointer to the raw data, which is a contiguous (T+1) by
                        num_symbols matrix with stride `stride`, containing
                        logprobs for frames 0..T-1 followed by zeros on
                        frame T.


      CAUTION: we may later enforce that stride == num_symbols, in order to
      be able to know the layout of a phantom matrix of arcs.  (?)
   */
  DenseFsa(int32_t T, int32_t num_symbols, int32_t arc_offset, float *data)
      : T(T), num_symbols(num_symbols), arc_offset(arc_offset), data(data) {}
};

/*  struct DenseFsaVecHeader documents/defines the format for storing
    the meta-info for DenseFsa.  The actual float data is stored separately,
    indexed [frame][symbol].  Note: these `frame` indexes are not the
    same as what you'd have used to index the original sequences, they
    are potentially much larger indexes into an array that you get
    by concatenating all the segments with one zero-valued frame
    in between each one (to store the loglike of the final arc).

 */
struct DenseFsaVecMeta {
  int32_t num_segs;     // The number of segments
  int32_t max_seg_len;  // The largest number of frames (not counting the zero
                        // padding frame) in any segment
  int32_t
      num_symbols;  // the number of symbols (== num-cols in features matrix)

  int32_t seg_frame_index[];  // size equals num_segs + 1; look at the next
                              // element for the last-plus-one frame.
                              // seg_frame_index[num_segs] is the total number
                              // of frames, which will equal \sum_segment
                              // (length(segment) + 1) where length(segment) is
                              // the number of frames of nnet output that that
                              // segment uses.

  // and after the frame_info, imagine we have: DenseFsaVecFrameInfo
  // frame_info[num_frames_padded] where num_frames_padded ==
  // seg_frame_index[num_segs].

  // returns the start of an array of dim `frame_info_dim()`
  DenseFsaVecFrameInfo *frame_info() {
    return reinterpret_cast<DenseFsaVecFrameInfo *>(seg_frame_index +
                                                    (num_segs + 1));
  }
  int32_t frame_info_dim() const { return seg_frame_index[num_segs]; }
  int32_t num_frames_padded() const { return seg_frame_index[num_segs]; }

  // and next we have the following, which will be used from
  // the Python calling code to copy the neural-net output to the correct
  // location.  This lists the elements of `frame_info` but with the
  // padding frames removed.
  //
  // DenseFsaVecFrameCopyInfo copy_info[num_frames];
  // where num_frames == \sum_segment length(segment) == total number of nnet
  //   output frames over all segments.
  //
  // int32_t frame_index[num_frames]; # frame_index will be an index into
  // `frame_info`;
  //                                  # this will be of the form 0 1 2 4 5 6 7 8
  //                                  9 11 ... # (note the gaps where the zero
  //                                  padding was!)
  int32_t *frame_index() {
    return reinterpret_cast<int32_t *>(frame_info() + frame_info_dim());
  }
  int32_t frame_index_dim() const {
    int32_t num_frames_padded = seg_frame_index[num_segs],
            num_frames = num_frames_padded - num_segs;
    return num_frames;
  }

  // The total size of this object in int32_t elements will equal:
  //   3 + # for first 3 elements
  //   num_segs + 1  +  # for seg_frame_index[]
  //   4 * (num_segs + num_frames) +  # == 4*num_frames_padded, for frame_info[]
  //   num_frames      # for frame_index[]
  //
  //   4 + 5*num_segs + 5*num_frames
};

struct DenseFsaVecFrameInfo {
  int32_t seg_id;  // The segment-id that this frame is part of (0 <= seg_id <=
                   // num_segs)... will be of the form 0 0 0 0 1 1 1 1 1 1 1 1 2
                   // 2 2 3 3 3 ...
  int32_t seq_id;  // The sequence-id that the `seg_id`'th segment was part of.
                   // Would equal seg_id in the case where it was one segment
                   // per sequence.
  int32_t frame_in_seg;  // The frame-index within the segment, so would be 0
                         // for the 1st frame of each segment, and
                         // `this_seg_num_frames` for the last (which could
                         // contain all zeros). Will be of the form  0 1 2 3 0 1
                         // 2 3 4 5 6 0 1 2 0 1 2....
  int32_t
      frame_in_seq;  // The frame-index within the sequence that this segment
                     // is a part of.  Would be the same as `frame_in_seg` if
                     // this segment starts at frame zero of its sequence.
};

/**
   Creates meta-info for DenseFsaVec (DenseFsaVecMeta) as one block in memory.
   For some of the terminology, see the comment above the definition class
   DenseFsaVec.

   First, some terminology.  Note: some of this is more relevant to the
   Python level here.  Please note that seq == sequence and seg == segment.
   The neural-network outputs, consisting of log likes, would be in a tensor
   of shape (num_seqs, num_frames, num_symbols).  I.e. we have
   `num_seqs` sequences; each sequence has `num_frames` outputs, and
   each frame of output has `num_symbols` symbols (e.g. phones or letters).

   There are `num_segs` segments.  Each segment is a subset of the frames
   in a sequence.

     @param [in] num_seqs  Number of sequences of (e.g.) phone
                           posteriors/loglikes from the neural net
     @param [in] frames_per_seq  Number of frames in each sequence
     @param [in] num_symbols  Dimension of the neural network output,
                              interpreted for instance as epsilon/blank
                              and the rest are phones or letters.
     @param [in] num_segs  Number of segments.  Each segment represents a range
                           of frames within a sequence.  There will in general
                           be at least as many segments as sequences.
     @param [in] seq_id    Indexed by 0 <= seg_id < num_segs, seq_id[seg_id]
                           contains the sequence index 0 <= s < num_seqs to
                           which this segment belongs
     @param [in] frame_begin Indexed by 0 <= seg_id < num_segs,
                             frame_begin[seg_id] contains the index of the
                             first frame of that segment.
     @param [in] frame_end Indexed by 0 <= seg_id < num_segs, frame_end[seg_id]
                           contains the index of the last-plus-one frame of that
                           segment.
     @param [in] storage_size  Size of `storage` array, in int32_t elements.
                               Defining num_frames =  sum(frame_end) -
                               sum(frame_begin), storage_size must equal
                               4 + 5*num_segs + 5*num_frames. It is provided as
                               an arg for checking purposes.
     @param [out] storage  Pointer to an array of int32_t where we put the
                           meta-info (probably part of a torch.Tensor).  It will
                           be interpreted internally as type DenseFsaVecMeta.
 */
void CreateDenseFsaVecMeta(int32_t num_seqs, int32_t frames_per_seq,
                           int32_t num_symbols, int32_t num_segs,
                           const int32_t *seq_id, const int32_t *frame_begin,
                           const int32_t *frame_end, ssize_t storage_size,
                           int32_t *storage);

/**
   DenseFsaVec represents a vector of FSAs with a special regular
   structure.  Suppose there are N FSAs, numbered n=0 .. N-1;
   and suppose the symbol space has S symbols numbered 0, ... S-1
   (yes, 0 represents epsilon; and we're not including the "final symbol"
   numbered -1).

   The n'th FSA corresponds to a log-likelihood matrix (call this M_n with M a
   matrix) with T_n frames.  Below, we'll just call this T for clarity.  This
   FSA has T+2 states, numbered 0, .. T+1.  For 0 < t < T and 0 <= s < S, there
   is an arc from state t to state t+1 with symbol s and log-like/weight
   equal to M_n(t, s).  From state T to T+1 there is a single arc with
   symbol -1=kFinalSymbol and log-like/weight equal to 0.0.  (Of course, state
   T+1 is the final state.. this is how our framework works).

 */
struct DenseFsaVec {
  /*
     Constructor.

       @param [in] meta   The meta-info, as written to by
     CreateDenseFsaVecMeta().
       @param [in] data      A contiguous, row-major matrix of shape
                        (meta_info->num_frames_padded(),meta_info->num_symbols),
                        containing the neural-net outputs for each segment with
                        zero rows in between for padding.
   */
  DenseFsaVec(const DenseFsaVecMeta *meta, const float *data)
      : meta(meta), data(data) {
    Check();
  }

  void Check();  // Sanity check (spot check, not thorough) on `meta_info`

  const DenseFsaVecMeta *meta;
  const float *data;

  DenseFsa operator[](int32_t seg_id) const {
    K2_CHECK_LT(seg_id, meta->num_segs);
    int32_t start_frame_index = meta->seg_frame_index[seg_id],
            end_frame_index = meta->seg_frame_index[seg_id + 1];
    // below, the -1 is to exclude the zero-padding frame.
    int32_t T = end_frame_index - start_frame_index - 1;
    int32_t arc_offset = meta->num_symbols * start_frame_index;
    return DenseFsa(T, num_symbols, arc_offset, this->data);
  }
};

/*
  Version of Intersect where `a` is dense?
 */
void Intersect(const DenseFsa &a, const Fsa &b, Fsa *c,
               std::vector<int32_t> *arc_map_a = nullptr,
               std::vector<int32_t> *arc_map_b = nullptr);

/*
  Version of Intersect where `a` is dense, pruned with pruning beam `beam`.
  Suppose states in the output correspond to pairs (s_a, s_b), and have
  forward-weights w(s_a, s_b), i.e. best-path from the start state...
  then if a state has a forward-weight w(s_a, s_b) that is less than
  (the largest w(s_a, x) for any x) minus the beam, we don't expand it.

  This is the same as time-synchronous Viterbi beam pruning.
*/
void IntersectPruned(const DenseFsa &a, const Fsa &b, float beam, Fsa *c,
                     std::vector<int32_t> *arc_map_a = nullptr,
                     std::vector<int32_t> *arc_map_b = nullptr);

/* Convert DenseFsa to regular Fsa (for testing purposes) */
void DenseToFsa(const DenseFsa &a, Fsa *b);

}  // namespace k2host

#endif  // K2_CSRC_HOST_DENSE_FSA_H_
