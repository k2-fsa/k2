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


#include <algorithm>
#include <vector>

#include "k2/csrc/rnnt_decode.h"

namespace k2 {
namespace rnnt_decoding {

/*
  Prune the incoming scores based on beam, max-states and max-contexts.
  Actually the beam part is not realy necessary, as we already pruned
  with the beam, but it doesn't cost anything extra.

  Args:
     incoming_scores [in]  The ragged array of scores to be pruned, indexed
           [stream][context][state][arc].  The scores are per arc, but
           it's at the state and context level
           that we prune, based on settings in this->config, so entire
           sub-lists of arcs will be deleted.
     arcs_new2old [out]  The new2old map of the pruned arcs will be
           written to here.

   Returns: pruned array of incoming scores, indexed
      [stream][context][state][arc].
 */
Ragged<double> RnntDecodingStream::PruneTwice(Ragged<double> &incoming_scores,
                                              Array1<int32_t> *arcs_new2old) {
  // TODO: could introduce a max-arcs per stream to prune on... this would
  // be done at this point, before pruning on states...  thus avoiding any problems
  // created by empty lists, although it's perhaps not an optimal way to prune.

  // states_prune is a renumbering on the states axis.
  Renumbering states_prune = PruneRagged(incoming_scores, 2,
                                         self.config.beam,
                                         self.config.max_states);

  Array1<double> arcs_new2old1;
  Ragged<double> temp_scores = SubsampleRagged(incoming_scores,
                                               states_prune,
                                               2, &arcs_new2old1);

  Renumbering context_prune = PruneRagged(temp_scores, 1,
                                          self.config.beam,
                                          self.config.max_contexts);

  Array1<double> arcs_new2old2;
  Ragged<double> ans_scores = SubsampleRagged(temp_scores,
                                              context_prune,
                                              1, &arcs_new2old2);
  *arcs_new2old = arcs_new2old1[arcs_new2old2];
  return ans_scores;
}





void RnntDecodingStream::Advance(Array2<float> &logprobs) {
  ContextPtr c = logprobs.Context();

  // arg-checking..

  int32_t num_streams = this->states.Dim0(),
      num_states = this->states.NumElements(),



  Array1<int32_t> num_arcs(c, num_states + 1);
  // populate array of num-arcs, indexed by idx012 into `states`.
  // These num-arcs are the num-arcs leaving the state in the corresponding
  // graph, plus one for the implicit epsilon self-loop.


  // Compute exclusive sum of num-arcs above.
  ExclusiveSum(num_arcs, &num_arcs);
  RaggedShape states2arcs_shape(&num_arcs, nullptr, -1);

  Array1<double> max_scores_per_stream(c, num_streams);
  {
    Ragged<int64> scores_per_stream = scores.RemoveAxis(0).RemoveAxis(1);
    MaxPerSublist(scores_per_stream, &max_scores_per_stream);
  }

  // unpruned_arcs_shape has 4 axes: [stream][context][state][arc]
  RaggedShape unpruned_arcs_shape = ComposeRaggedShapes(this->states.shape,
                                                        states2arcs_shape);



  Renumbering pass1_renumbering;
  // Do initial pruning pass on the arcs (because it will be quite a large array),
  // populating the `keep` array of a Renumbering object..
  // The pruning rule is:
  //   (1) keep all epsilon transitions to the next frame, to ensure there is
  //       no way we can have no states surviving.
  //   (2) for all other arcs, keep the it if the forward scores after the
  //       arc would be >= the max_scores_per_stream entry for this stream, minus
  //       the beam from the config.
  RaggedShape pass1_arcs_shape = SubsampleRaggedShape(
      unpruned_arcs_shape,
      pass1_renumbering);  // [stream][state][context][arc]

  // stream_arc_shape is pass1_arcs indexed [stream][arc].
  // We need to rearrange so it's by destination context and state, not source.
  RaggedShape stream_arc_shape = pass1_arcs_shape.RemoveAxis(2).RemoveAxis(1);

  Ragged<ArcInfo> arcs(pass1_arcs_shape); // arcs, indexed [stream][state][context][arc].  Might eventually make this non-materialized.
  Ragged<int64_t> states(stream_arc_shape);  // dest-states of arcs, incexed [stream][arc]
  Ragged<double> scores(stream_arc_shape);  // final-scores after arcs, indexed [stream][arc]

  // A kernel, here, populates arcs, states and scores; it computes
  // the destination state for each arc and puts its in 'states',
  // and the after-the-arc scores for each arc and puts them in
  // 'scores'.
  int32_t cur_num_arcs = arcs.NumElements();

  Array1<int32_t> dest_state_sort_new2old(c, states.NumElements());
  SortSublists(&states, &dest_state_sort_new2old);
  // We will delay doing:
  //  arcs.values = arcs.values[dest_state_sort_new2old];
  // .. until later.
  scores.values = scores.values[dest_state_sort_new2old];

  // state_boundaries and context_boundaries are Renumbering objects
  // that we use in a slightly different way from normal.
  // We populate their Keep() arrays with:
  //  for context_boundaries: a 1 if next_stream != this_stream,
  //     or next_context != this_context.
  //  for state_boundaries: a 1 if next_stream != this_stream or
  //     next_state (i.e. the 64-bit state index) != this_state.
  Renumbering context_boundaries(c, cur_num_arcs);
  Renumbering state_boundaries(c, cur_num_arcs);

  Array1<int32_t> arc2state_row_ids_extra = state_boundaries.Old2New(true),
      arc2state_row_ids = state_boundaries.Old2New(),
      state2arc_row_splits = state_boundaries.New2Old(true);

  RaggedShape state_arc_shape(&state2arc_row_splits, &arc2state_row_ids, -1);

  Array1<int32_t> arc2ctx_row_ids_extra = context_boundaries.Old2New(true),
      arc2ctx_row_ids = context_boundaries.Old2New(),
      ctx2arc_row_splits = context_boundaries.New2Old(true);

  Array1<int32_t> state2ctx_row_ids = arc2ctx_row_ids[state2arc_row_splits[:-1]],
      ctx2state_row_splits = arc2state_row_ids_extra[ctx2arc_row_splits];

  RaggedShape ctx_state_shape(&ctx2state_row_splits, state2ctx_row_ids);

  Array1<int32_t> &arc2stream_row_ids = stream_arc_shape.RowIds(1),
      stream2arc_row_splits = stream_arc_shape.RowSplits(1);

  Array1<int32_t> ctx2stream_row_ids = arc2stream_row_ids[ctx2arc_row_splits[:-1]],
      stream2ctx_row_splits = arc2ctx_row_ids_extra[stream2arc_row_splits];

  RaggedShape stream_ctx_shape(&ctx2stream_row_ids, stream2ctx_row_splits);

  // incoming_arcs_shape has indexes [stream][context][state][arc].
  // It represents the incoming arcs sorted by destination state.
  RaggedShape incoming_arcs_shape = ComposeRaggedShapes(stream_ctx_shape,
                                                        ctx_state_shape,
                                                        state_arc_shape);

  Ragged<double> incoming_scores(incoming_arcs_shape, scores.values);

  Array1<int32_t> arcs_prune2_new2old;
  Ragged<double> pruned_incoming_scores = PruneTwice(incoming_scores,
                                                     &arcs_prune2_new2old);

  Ragged<int64_t> pruned_dest_states(pruned_incoming_scores.shape,
                                     states.values[arcs_prune2_new2old]);

  // arcs_new2old is new2old map from indexes in `incoming_scores` or `pruned_dest_states`,
  // to indexes into `arcs` (remember, we did not renumber arcs, it is in the original
  // order after pass1 pruning).
  Array1<int32_t> arcs_new2old = dest_state_sort_new2old[arcs_prune2_new2old];

  // prune_orig_arcs is a renumbering object that will renumber the original arcs.
  // We can't use `prune_scores` because that one is after sorting the arcs by
  // destination state.
  Renumbering prune_orig_arcs(...);

  // the next line may not be real code.  not sure if it's even needed,
  // or if Keep() is initialized to 0.
  prune_orig_arcs.Keep() = 0;
  // the next line is not real code, we'd do it with a kernel.
  prune_orig_arcs.Keep()[arcs_new2old] = 1;

  // pruned_arcs is indexed [stream][src_state][context][arc].
  Ragged<ArcInfo> pruned_arcs = SubsampleRagged(arcs, prune_orig_arcs);

  // arcs_dest2src maps from an arc-index in `pruned_incoming_scores` to an
  // arc-index in `pruned_arcs`.  This is a permutation of integers
  // 0..num_pruned_arcs-1.
  Array1<int32_t> arcs_dest2src = pruned_arcs.Old2New()[arcs_new2old];

  // Here, a kernel sets the dest_state of the arcs in pruned_arcs.
  // It works as follows:
  //  For each arc_idx0123 in `pruned_dest_states`:
  //    work out the state_idx2, which will be the `dest_state`
  //    for the corresponding ArcInfo.
  //  Work out the arc-index (arc_idx0123) in `pruned_arcs`, which
  //  is just arcs_dest2src[arc_idx0123], and then set the dest_state
  //  in `pruned_arcs`.

  // Here, use MaxPerSublist to reduce `pruned_incoming_scores` to be per
  // state not per arc.  (Need to remove last axis from the shape)


  int32_t num_dest_states = pruned_incoming_scores.TotSize(2);
  Array1<double> dest_state_scores_values(c, num_dest_states);
  MaxPerSublist(pruned_incoming_scores, -infinity, &dest_state_scores_values);

  // dest_state_scores will be the 'scores' held by this object on the next frame
  Ragged<double> dest_state_scores(pruned_incoming_scores.RemoveAxis(3),
                                   dest_state_scores_values);
  // dest_states will be the `states` held by this object on the next frame.
  Ragged<int32_t> dest_states(dest_state_scores.shape,
                              pruned_states_dups.values[pruned_dest_states.RowSplits(3)[:-1]]);


  this->prev_frames.push_back(std::make_unique<Ragged<ArcInfo> >(pruned_arcs));

  this->states = dest_states;
  this->scores = dest_state_scores;





  // states_prune is a renumbering on the states axis.
  Renumbering states_prune = PruneRagged(incoming_scores, 2, self.config.beam,
                                         self.config.max_states);

  Renumbering
  incoming_scores_pruned1 = SubsampleRagged(incoming_scores,
                                            states_prune, 2,



  // Now use the pruning functions I mentioned in a separate PR to prune
  // incoming_arcs_shape based on the max-active at the context and
  // at the state level, per stream of course.
  //

  // suppose the prune_scores object below is somehow the composition of
  // the Renumbering objects obtained from the two invocations of Prune()
  // above.  It may not be convenient to make this an actual
  // Renumbering object, it could just be a new2old or something like
  // that.
  Renumbering prune_scores(...);

  // Note: pruned_scores still has an extra axis on it that's not needed,
  // we can remove that below.
  Ragged<double> pruned_scores_dups = SubsampleRaggedShape(scores, prune_scores);

  // pruned_states_dups contains states with duplicates, it has an unnecessary last axis.
  Ragged<int64_t> pruned_states_dups(pruned_scores_dups.shape,
                                     states.values[prune_scores_dups.New2Old()]);


  Array1<double> pruned_scores_values(c, pruned_scores_dups.TotSize(2));
  MaxPerSublist(pruned_states_dups, -infinity, &pruned_scores_values);

  // pruned_scores will be the 'scores' held by this object on the next frame
  Ragged<double> pruned_scores(pruned_scores_dups.RemoveAxis(3),
                               pruned_scores_values);

  // pruned_states will be the 'states' held by this object on the next frame
  Ragged<int32_t> pruned_states(pruned_scores.shape,
                                pruned_states_dups.values[pruned_states_dups.RowSplits(3)[:-1]])




  // pruned_arcs still indexed: [stream][state][context][arc]
  Ragged<ArcInfo> pruned_arcs(SubsampleRaggedShape(pass1_arcs_shape,
                                                   prune_orig_arcs));

  Ragged<ArcInfo> arcs = Subsample



  // Now set the dest-state of the arcs to the appropriate state
  // index, would be the idx2 within incoming_arcs_shape.

  // Now we need to renumber the incoming arcs so that they correspond
  // to the







      ctx2arc_row_splits

  RaggedShape state2arc(&state2arc_row_splits, &state2arc_row_ids, -1);




  Array1<int32_t> arc2context_row_ids = context_boundaries.Old2New();
  // caution: translate next line from python, the [:-1]
  Array1<int32_t> state2context_row_ids = arc2context_row_ids[state2arc.RowSplits(1)[:-1]];

  RaggedShape context2state(nullptr, &state2context_row_ids, state2arc.Dim0()); // 2 axes: [context][state]


  Array1<int32_t> stream2context_row_splits = context_boundaries.N
  {
    Array1<int32_t> arc2context_row_splits(context_boundaries.NumNewElems() + 1);
    RowIdsToRowSplits(arc2context_row_ids, &arc2context_row_splits);

  }






  // For
  Array1<int32_t> context_row_ids = context_boundaries.Old2New();



      state_row_ids = state_
      state_row_ids = state_boundaries.Old2New(true);





}

}  // namespace rnnt_decoding
}  // namespace k2

#endif  // K2_CSRC_RNNT_DECODE_H_
