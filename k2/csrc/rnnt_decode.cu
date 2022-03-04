/**
 * Copyright      2022  Xiaomi Corporation (authors: Daniel Povey, Wei kang)
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

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/rnnt_decode.h"

namespace k2 {
namespace rnnt_decoding {

std::shared_ptr<RnntDecodingStream> CreateStream(
    const std::shared_ptr<Fsa> &graph) {
  K2_CHECK_EQ(graph->shape.NumAxes(), 2);
  ContextPtr &c = graph->shape.Context();
  RnntDecodingStream stream;
  stream.graph = graph;
  stream.num_graph_states = graph->shape.Dim0();
  stream.states = Ragged<int64_t>(RegularRaggedShape(c, 1, 1),
                                  Array1<int64_t>(c, std::vector<int64_t>{0}));
  stream.scores = Ragged<double>(RegularRaggedShape(c, 1, 1),
                                 Array1<double>(c, std::vector<double>{0.0}));
  return std::make_shared<RnntDecodingStream>(stream);
}

RnntDecodingStreams::RnntDecodingStreams(
    std::vector<std::shared_ptr<RnntDecodingStream>> &srcs,
    const RnntDecodingConfig &config)
    : attached_(true), srcs_(srcs), num_streams_(srcs.size()), config_(config) {
  K2_CHECK_GE(num_streams_, 1);
  c_ = srcs_[0]->graph->shape.Context();

  Array1<int32_t> num_graph_states(GetCpuContext(), num_streams_);

  int32_t *num_graph_states_data = num_graph_states.Data();

  std::vector<Ragged<int64_t> *> states_ptr(num_streams_);
  std::vector<Ragged<double> *> scores_ptr(num_streams_);
  std::vector<Fsa> graphs(num_streams_);

  for (int32_t i = 0; i < num_streams_; ++i) {
    K2_CHECK(c_->IsCompatible(*(srcs_[i]->graph->shape.Context())));
    num_graph_states_data[i] = srcs_[i]->num_graph_states;
    states_ptr[i] = &(srcs_[i]->states);
    scores_ptr[i] = &(srcs_[i]->scores);
    graphs[i] = *(srcs_[i]->graph);
  }

  num_graph_states_ = num_graph_states.To(c_);
  states_ = Stack(0, num_streams_, states_ptr.data());
  scores_ = Stack(0, num_streams_, scores_ptr.data());
  graphs_ = Array1OfRagged<Arc>(graphs.data(), num_streams_);

  // We don't combine prev_frames_ here, will do that when needed, for example
  // when we need all prev_frames_ to format output fsas.
}

void RnntDecodingStreams::Detach() {
  if (!attached_ || prev_frames_.empty()) return;
  std::vector<Ragged<int64_t>> states;
  std::vector<Ragged<double>> scores;
  Unstack(states_, 0, &states);
  Unstack(scores_, 0, &scores);
  K2_CHECK_EQ(static_cast<int32_t>(states.size()), num_streams_);
  K2_CHECK_EQ(static_cast<int32_t>(scores.size()), num_streams_);

  for (int32_t i = 0; i < num_streams_; ++i) {
    srcs_[i]->states = states[i];
    srcs_[i]->scores = scores[i];
  }

  // detatch prev_frames_
  std::vector<Ragged<ArcInfo> *> frames_ptr;
  for (size_t i = 0; i < prev_frames_.size(); ++i) {
    frames_ptr.emplace_back(prev_frames_[i].get());
  }
  // statck_frames has a shape of [t][stream][state][arc]
  auto stack_frames = Stack(0, prev_frames_.size(), frames_ptr.data());
  // stack_frames now has a shape of [strams][state][arc]
  // its Dim0(0) equals to `num_streams_ * prev_frames_.size()`
  stack_frames = stack_frames.RemoveAxis(0);

  std::vector<Ragged<ArcInfo>> frames;
  Unstack(stack_frames, 0, &frames);

  K2_CHECK_EQ(num_streams_ * prev_frames_.size(), frames.size());

  for (size_t i = 0; i < prev_frames_.size(); ++i) {
    for (int32_t j = 0; j < num_streams_; ++j) {
      srcs_[j]->prev_frames.emplace_back(
          std::make_shared<Ragged<ArcInfo>>(frames[i * num_streams_ + j]));
    }
  }
  attached_ = false;
  prev_frames_.clear();
}

void RnntDecodingStreams::GetContexts(RaggedShape *shape,
                                      Array2<int32_t> *contexts) {
  K2_CHECK(shape);
  K2_CHECK(contexts);
  K2_CHECK_EQ(states_.NumAxes(), 3);
  *shape = RemoveAxis(states_.shape, 2);
  int32_t num_contexts = shape->TotSize(1),
          decoder_history_len = config_.decoder_history_len,
          vocab_size = config_.vocab_size;
  *contexts = Array2<int32_t>(c_, num_contexts, decoder_history_len);
  const int32_t *shape_row_ids1_data = shape->RowIds(1).Data(),
                *states_row_splits2_data = states_.RowSplits(2).Data();
  auto contexts_acc = contexts->Accessor();

  const int32_t *num_graph_states_data = num_graph_states_.Data();
  const int64_t *states_values_data = states_.values.Data();

  K2_EVAL2(
      c_, num_contexts, decoder_history_len, lambda_set_contexts,
      (int32_t row, int32_t col) {
        int32_t idx0 = shape_row_ids1_data[row],
                num_graph_states = num_graph_states_data[idx0],
                state_idx01x = states_row_splits2_data[row];
        int64_t state_value = states_values_data[state_idx01x];
        int32_t exp_v = decoder_history_len - col;
        int64_t context_state = state_value / num_graph_states,
                state = context_state % (int64_t)pow(vocab_size, exp_v);
        state = state / pow(vocab_size, exp_v - 1);
        contexts_acc(row, col) = state;
      });
}

Ragged<double> RnntDecodingStreams::PruneTwice(Ragged<double> &incoming_scores,
                                               Array1<int32_t> *arcs_new2old) {
  K2_CHECK_EQ(incoming_scores.NumAxes(), 4);
  K2_CHECK_EQ(incoming_scores.Dim0(), num_streams_);
  // TODO: could introduce a max-arcs per stream to prune on... this would
  // be done at this point, before pruning on states...  thus avoiding any
  // problems created by empty lists, although it's perhaps not an optimal way
  // to prune.

  // incoming_scores has a shape of [stream][context][state][arc]
  // states_prune is a renumbering on the states axis.
  Renumbering states_prune = PruneRagged(incoming_scores, 2 /*axis*/,
                                         config_.beam, config_.max_states);

  Array1<int32_t> arcs_new2old1;
  Ragged<double> temp_scores =
      SubsetRagged(incoming_scores, states_prune, 2 /*axis*/, &arcs_new2old1);

  // incoming_scores has a shape of [stream][context][state][arc]
  // context_prune is a renumbering on the states context.
  Renumbering context_prune =
      PruneRagged(temp_scores, 1 /*axis*/, config_.beam, config_.max_contexts);
  Array1<int32_t> arcs_new2old2;
  Ragged<double> ans_scores =
      SubsetRagged(temp_scores, context_prune, 1 /*axis*/, &arcs_new2old2);

  if (arcs_new2old) *arcs_new2old = arcs_new2old1[arcs_new2old2];

  return ans_scores;
}

void RnntDecodingStreams::Advance(Array2<float> &logprobs) {
  K2_CHECK(attached_) << "Streams detached.";
  ContextPtr c = logprobs.Context();
  K2_CHECK(c_->IsCompatible(*c));

  int32_t num_streams = states_.Dim0(), num_states = states_.NumElements();

  Array1<int32_t> num_arcs(c_, num_states + 1);
  // populate array of num-arcs, indexed by idx012 into `states`.
  // These num-arcs are the num-arcs leaving the state in the corresponding
  // graph, plus one for the implicit epsilon self-loop.
  const int32_t *this_states_row_ids2_data = states_.RowIds(2).Data(),
                *this_states_row_ids1_data = states_.RowIds(1).Data(),
                *num_graph_states_data = num_graph_states_.Data();

  const int64_t *this_states_values_data = states_.values.Data();
  int32_t **graph_row_splits1_ptr_data = graphs_.shape.RowSplits(1);
  int32_t *num_arcs_data = num_arcs.Data();

  K2_EVAL(
      c_, num_states, lambda_set_num_arcs, (int32_t idx012) {
        int64_t state_value = this_states_values_data[idx012];
        int32_t
            idx0 = this_states_row_ids1_data[this_states_row_ids2_data[idx012]],
            num_graph_states = num_graph_states_data[idx0],
            graph_state = state_value % num_graph_states;

        const int32_t *graph_row_split1_data = graph_row_splits1_ptr_data[idx0];
        // plus one for the implicit epsilon self-loop
        num_arcs_data[idx012] = graph_row_split1_data[graph_state + 1] -
                                graph_row_split1_data[graph_state] + 1;
      });

  // Compute exclusive sum of num-arcs above.
  ExclusiveSum(num_arcs, &num_arcs);
  RaggedShape states2arcs_shape = RaggedShape2(&num_arcs, nullptr, -1);

  // unpruned_arcs_shape has 4 axes: [stream][context][state][arc]
  RaggedShape unpruned_arcs_shape =
      ComposeRaggedShapes(states_.shape, states2arcs_shape);

  Array1<double> max_scores_per_stream(c_, num_streams);
  double minus_inf = -std::numeric_limits<double>::infinity();
  {
    // scores_ has 3 axes: [stream][context][score]
    Ragged<double> scores_per_stream = scores_.RemoveAxis(1);
    MaxPerSublist<double>(scores_per_stream, minus_inf, &max_scores_per_stream);
  }

  // Do initial pruning pass on the arcs (because it will be quite a large
  // array), populating the `keep` array of a Renumbering object.. The pruning
  // rule is:
  //   (1) keep all epsilon transitions to the next frame, to ensure there is
  //       no way we can have no states surviving.
  //   (2) for all other arcs, keep the it if the forward scores after the
  //       arc would be >= the max_scores_per_stream entry for this stream,
  //       minus the beam from the config.
  Renumbering pass1_renumbering(c_, unpruned_arcs_shape.NumElements());
  char *pass1_keep_data = pass1_renumbering.Keep().Data();
  const auto logprobs_acc = logprobs.Accessor();
  const double *this_scores_data = scores_.values.Data(),
               *max_scores_per_stream_data = max_scores_per_stream.Data();
  double beam = config_.beam;
  // "uas" is short for unpruned_arcs_shape
  const int32_t *uas_row_ids3_data = unpruned_arcs_shape.RowIds(3).Data(),
                *uas_row_splits3_data = unpruned_arcs_shape.RowSplits(3).Data(),
                *uas_row_ids2_data = unpruned_arcs_shape.RowIds(2).Data(),
                *uas_row_ids1_data = unpruned_arcs_shape.RowIds(1).Data();

  Arc **graphs_arcs_data = graphs_.values.Data();

  K2_EVAL(
      c_, unpruned_arcs_shape.NumElements(), lambda_pass1_pruning,
      (int32_t idx0123) {
        int32_t idx012 = uas_row_ids3_data[idx0123],
                idx012x = uas_row_splits3_data[idx012],
                idx3 = idx0123 - idx012x;
        // keep the implicit epsilon self-loop
        if (idx3 == 0) {
          pass1_keep_data[idx0123] = 1;
          return;
        }
        int32_t idx01 = uas_row_ids2_data[idx012],
                idx0 = uas_row_ids1_data[idx01],
                num_graph_states = num_graph_states_data[idx0];

        const Arc *graph_arcs_data = graphs_arcs_data[idx0];
        const int32_t *graph_row_split1_data = graph_row_splits1_ptr_data[idx0];
        int64_t this_state = this_states_values_data[idx012];
        int32_t graph_state = this_state % num_graph_states,
                graph_idx0x = graph_row_split1_data[graph_state],
                graph_idx01 =
                    graph_idx0x + idx3 - 1;  // minus 1 as the implicit epsilon
                                             // self-loop takes the position 0.
        Arc arc = graph_arcs_data[graph_idx01];

        // keep the epsilon transitions
        if (arc.label == 0) {
          pass1_keep_data[idx0123] = 1;
          return;
        }
        if (arc.label == -1) {
          pass1_keep_data[idx0123] = 0;
          return;
        }

        double this_score = this_scores_data[idx012], arc_score = arc.score,
               log_prob = logprobs_acc(idx01, arc.label),
               score = this_score + arc_score + log_prob,
               max_score = max_scores_per_stream_data[idx0];
        // prune with beam
        if (score >= max_score - beam) {
          pass1_keep_data[idx0123] = 1;
        } else {
          pass1_keep_data[idx0123] = 0;
        }
      });

  RaggedShape pass1_arcs_shape =
      SubsetRaggedShape(unpruned_arcs_shape,
                        pass1_renumbering);  // [stream][context][state][arc]

  // stream_arc_shape is pass1_arcs indexed [stream][arc].
  // We need to rearrange so it's by destination context and state, not source.
  RaggedShape stream_arc_shape = RemoveAxis(pass1_arcs_shape, 2);
  stream_arc_shape = RemoveAxis(stream_arc_shape, 1);

  // arcs, indexed [stream][context][state][arc].  Might eventually make this
  // non-materialized.
  Ragged<ArcInfo> arcs(pass1_arcs_shape);
  // dest-states of arcs, incexed [stream][arc]
  Ragged<int64_t> states(stream_arc_shape);
  // final-scores after arcs, indexed [stream][arc]
  Ragged<double> scores(stream_arc_shape);

  int32_t cur_num_arcs = arcs.NumElements();
  // This renumber will be used for renumbering the arcs after we fiishing
  // pruning.
  Renumbering renumber_arcs(c_, cur_num_arcs);
  char *renumber_arcs_keep_data = renumber_arcs.Keep().Data();

  const int32_t *pass1_new2old_data = pass1_renumbering.New2Old().Data();
  int64_t *states_data = states.values.Data();
  double *scores_data = scores.values.Data();
  ArcInfo *arcs_data = arcs.values.Data();
  int32_t vocab_size = config_.vocab_size,
          decoder_history_len = config_.decoder_history_len;

  // Populate arcs, states and scores; it computes
  // the destination state for each arc and puts its in 'states',
  // and the after-the-arc scores for each arc and puts them in
  // 'scores'.
  K2_EVAL(
      c_, cur_num_arcs, lambda_populate_arcs_states_scores, (int32_t arc_idx) {
        // Init renumber_arcs to 0, place here to save one kernel.
        renumber_arcs_keep_data[arc_idx] = 0;
        int32_t idx0123 = pass1_new2old_data[arc_idx],
                idx012 = uas_row_ids3_data[idx0123],
                idx012x = uas_row_splits3_data[idx012],
                idx3 = idx0123 - idx012x, idx01 = uas_row_ids2_data[idx012],
                idx0 = uas_row_ids1_data[idx01],
                num_graph_states = num_graph_states_data[idx0];
        int64_t this_state = this_states_values_data[idx012];
        double this_score = this_scores_data[idx012];

        // handle the implicit epsilon self-loop
        if (idx3 == 0) {
          states_data[arc_idx] = this_state;
          scores_data[arc_idx] = this_score + logprobs_acc(idx01, 0);
          ArcInfo ai;
          ai.graph_arc_idx01 = -1;
          ai.score = logprobs_acc(idx01, 0);
          arcs_data[arc_idx] = ai;
          return;
        }

        const Arc *graph_arcs_data = graphs_arcs_data[idx0];
        const int32_t *graph_row_split1_data = graph_row_splits1_ptr_data[idx0];

        int64_t this_context_state = this_state / num_graph_states;
        int32_t graph_state = this_state % num_graph_states,
                graph_idx0x = graph_row_split1_data[graph_state],
                graph_idx01 = graph_idx0x + idx3 - 1;  // minus 1 here as
                                                       // epsilon self-loop
                                                       // takes the position 0.

        Arc arc = graph_arcs_data[graph_idx01];
        int64_t context_state = this_context_state;

        // non epsilon transitions, update context_state
        if (arc.label != 0) {
          context_state = this_context_state %
                          (int64_t)pow(vocab_size, decoder_history_len - 1);
          context_state = context_state * vocab_size + arc.label;
        }

        // next state is the state current arc pointting to.
        int64_t state = context_state * num_graph_states + arc.dest_state;
        states_data[arc_idx] = state;

        double arc_score = arc.score, log_prob = logprobs_acc(idx01, arc.label),
               score = this_score + arc_score + log_prob;
        scores_data[arc_idx] = score;

        ArcInfo ai;
        ai.graph_arc_idx01 = graph_idx01;
        ai.score = arc_score + log_prob;
        arcs_data[arc_idx] = ai;
      });

  // sort states so that we can group states by context-state
  Array1<int32_t> dest_state_sort_new2old(c, states.NumElements());
  SortSublists(&states, &dest_state_sort_new2old);

  scores.values = scores.values[dest_state_sort_new2old];

  // state_boundaries and context_boundaries are Renumbering objects
  // that we use in a slightly different way from normal.
  // We populate their Keep() arrays with:
  //  for context_boundaries: a 1 if next_stream != this_stream,
  //     or next_context != this_context.
  //  for state_boundaries: a 1 if next_stream != this_stream or
  //     next_state (i.e. the 64-bit state index) != this_state.
  Renumbering context_boundaries(c_, cur_num_arcs);
  Renumbering state_boundaries(c_, cur_num_arcs);
  const int32_t *states_row_ids1_data = states.RowIds(1).Data();
  char *context_boundaries_keep_data = context_boundaries.Keep().Data(),
       *state_boundaries_keep_data = state_boundaries.Keep().Data();

  K2_EVAL(
      c_, cur_num_arcs, lambda_set_boundaries, (int32_t idx01) {
        int32_t context_keep = 0, state_keep = 0;

        if (idx01 != cur_num_arcs - 1) {
          int32_t idx0 = states_row_ids1_data[idx01],
                  idx0_next = states_row_ids1_data[idx01 + 1],
                  num_graph_states = num_graph_states_data[idx0],
                  next_num_graph_states = num_graph_states_data[idx0_next];
          int64_t state_value = states_data[idx01],
                  next_state_value = states_data[idx01 + 1],
                  context_state = state_value / num_graph_states,
                  next_context_state = next_state_value / next_num_graph_states;
          if (idx0 != idx0_next || context_state != next_context_state)
            context_keep = 1;
          if (idx0 != idx0_next || state_value != next_state_value)
            state_keep = 1;
        } else {
          context_keep = 1;
          state_keep = 1;
        }

        context_boundaries_keep_data[idx01] = context_keep;
        state_boundaries_keep_data[idx01] = state_keep;
      });

  Array1<int32_t> arc2state_row_ids_extra = state_boundaries.Old2New(true),
                  arc2state_row_ids = state_boundaries.Old2New(),
                  state_boundaries_new2old = state_boundaries.New2Old();

  RaggedShape state_arc_shape =
      RaggedShape2(nullptr, &arc2state_row_ids, arc2state_row_ids.Dim());

  Array1<int32_t> arc2ctx_row_ids_extra = context_boundaries.Old2New(true),
                  arc2ctx_row_ids = context_boundaries.Old2New(),
                  context_boundaries_new2old = context_boundaries.New2Old();

  Array1<int32_t> state2ctx_row_ids = arc2ctx_row_ids[state_boundaries_new2old];

  RaggedShape ctx_state_shape =
      RaggedShape2(nullptr, &state2ctx_row_ids, state2ctx_row_ids.Dim());

  Array1<int32_t> &arc2stream_row_ids = stream_arc_shape.RowIds(1),
                  &stream2arc_row_splits = stream_arc_shape.RowSplits(1);

  Array1<int32_t> ctx2stream_row_ids =
                      arc2stream_row_ids[context_boundaries_new2old],
                  stream2ctx_row_splits =
                      arc2ctx_row_ids_extra[stream2arc_row_splits];

  RaggedShape stream_ctx_shape = RaggedShape2(
      &stream2ctx_row_splits, &ctx2stream_row_ids, ctx2stream_row_ids.Dim());

  // incoming_arcs_shape has indexes [stream][context][state][arc].
  // It represents the incoming arcs sorted by destination state.
  RaggedShape incoming_arcs_shape =
      ComposeRaggedShapes3(stream_ctx_shape, ctx_state_shape, state_arc_shape);

  Ragged<double> incoming_scores(incoming_arcs_shape, scores.values);

  Array1<int32_t> arcs_prune2_new2old;
  Ragged<double> pruned_incoming_scores =
      PruneTwice(incoming_scores, &arcs_prune2_new2old);

  Ragged<int64_t> pruned_dest_states(pruned_incoming_scores.shape,
                                     states.values[arcs_prune2_new2old]);

  // arcs_new2old is new2old map from indexes in `incoming_scores` or
  // `pruned_dest_states`, to indexes into `arcs` (remember, we did not renumber
  // arcs, it is in the original order after pass1 pruning).
  Array1<int32_t> arcs_new2old = dest_state_sort_new2old[arcs_prune2_new2old];

  // Renumber the original arcs, we create and initialize the renumbering object
  // when we create the arcs, see above.
  // arcs has a shape of [stream][context][state][arc]
  int32_t *arcs_new2old_data = arcs_new2old.Data();
  K2_EVAL(
      c_, arcs_new2old.Dim(), lambda_renumber_arcs, (int32_t idx) {
        int32_t arc_idx0123 = arcs_new2old_data[idx];
        renumber_arcs_keep_data[arc_idx0123] = 1;
      });

  // pruned_arcs is indexed [stream][context][src_state][arc].
  Ragged<ArcInfo> pruned_arcs = SubsetRagged(arcs, renumber_arcs);

  // arcs_dest2src maps from an arc-index in `pruned_incoming_scores` to an
  // arc-index in `pruned_arcs`.  This is a permutation of integers
  // 0..num_pruned_arcs-1.
  Array1<int32_t> arcs_dest2src = renumber_arcs.Old2New()[arcs_new2old];

  auto reduce_pruned_dest_states = RemoveAxis(pruned_dest_states, 1);
  // "rpds" is short for reduce_pruned_dest_states
  const int32_t *rpds_row_ids2_data =
                    reduce_pruned_dest_states.RowIds(2).Data(),
                *rpds_row_ids1_data =
                    reduce_pruned_dest_states.RowIds(1).Data(),
                *rpds_row_splits1_data =
                    reduce_pruned_dest_states.RowSplits(1).Data(),
                *arcs_dest2src_data = arcs_dest2src.Data();
  ArcInfo *pruned_arcs_data = pruned_arcs.values.Data();

  // Set the dest_state of the arcs in pruned_arcs.
  // It works as follows:
  //  For each arc_idx0123 in `pruned_dest_states`:
  //    work out the state_idx2, which will be the `dest_state`
  //    for the corresponding ArcInfo.
  //  Work out the arc-index (arc_idx0123) in `pruned_arcs`, which
  //  is just arcs_dest2src[arc_idx0123], and then set the dest_state
  //  in `pruned_arcs`.

  // reduce_pruned_dest_states has a shape of [stream][state][arc]
  K2_EVAL(
      c_, reduce_pruned_dest_states.NumElements(), lambda_set_dest_states,
      (int32_t idx012) {
        int32_t idx01 = rpds_row_ids2_data[idx012],
                idx0 = rpds_row_ids1_data[idx01],
                idx0x = rpds_row_splits1_data[idx0], idx1 = idx01 - idx0x,
                pruned_arc_idx0123 = arcs_dest2src_data[idx012];

        ArcInfo info = pruned_arcs_data[pruned_arc_idx0123];
        info.dest_state = idx1;
        pruned_arcs_data[pruned_arc_idx0123] = info;
      });

  // Here, use MaxPerSublist to reduce `pruned_incoming_scores` to be per
  // state not per arc.  (Need to remove last axis from the shape)
  int32_t num_dest_states = pruned_incoming_scores.TotSize(2);
  Array1<double> dest_state_scores_values(c_, num_dest_states);
  MaxPerSublist(pruned_incoming_scores, minus_inf, &dest_state_scores_values);

  // dest_state_scores will be the 'scores' held by this object on the next
  // frame
  Ragged<double> dest_state_scores(RemoveAxis(pruned_incoming_scores.shape, 3),
                                   dest_state_scores_values);
  // dest_states will be the `states` held by this object on the next frame.
  auto pruned_row_split3 = pruned_dest_states.RowSplits(3);
  Ragged<int64_t> dest_states(
      dest_state_scores.shape,
      pruned_dest_states
          .values[pruned_row_split3.Arange(0, pruned_row_split3.Dim() - 1)]);

  states_ = dest_states;
  scores_ = dest_state_scores;
  prev_frames_.emplace_back(
      std::make_shared<Ragged<ArcInfo>>(pruned_arcs.RemoveAxis(1)));
}

void RnntDecodingStreams::UpdatePrevFrames(std::vector<int32_t> &num_frames) {
  K2_CHECK(!attached_) << "Please call Detach() first.";
  K2_CHECK_EQ(num_streams_, static_cast<int32_t>(num_frames.size()));
  std::vector<Ragged<ArcInfo> *> frames_ptr;
  Array1<int32_t> stream2t_row_splits(GetCpuContext(), num_frames.size() + 1);

  for (size_t i = 0; i < num_frames.size(); ++i) {
    stream2t_row_splits.Data()[i] = num_frames[i];
    K2_CHECK_LE(num_frames[i],
                static_cast<int32_t>(srcs_[i]->prev_frames.size()));
    for (int32_t j = 0; j < num_frames[i]; ++j) {
      frames_ptr.push_back(srcs_[i]->prev_frames[j].get());
    }
  }

  // frames has a shape of [t][state][arc],
  // its Dim0() equals std::sum(num_frames)
  auto frames = Stack(0, frames_ptr.size(), frames_ptr.data());

  stream2t_row_splits = stream2t_row_splits.To(c_);
  ExclusiveSum(stream2t_row_splits, &stream2t_row_splits);
  auto stream2t_shape = RaggedShape2(&stream2t_row_splits, nullptr, -1);

  // now frames has a shape of [stream][t][state][arc]
  frames = Ragged<ArcInfo>(ComposeRaggedShapes(stream2t_shape, frames.shape),
                           frames.values);

  std::vector<Ragged<ArcInfo>> prev_frames;
  Unstack(frames, 1, "left", &prev_frames);

  prev_frames_.resize(prev_frames.size());
  for (size_t i = 0; i < prev_frames.size(); ++i) {
    prev_frames_[i] = std::make_shared<Ragged<ArcInfo>>(prev_frames[i]);
  }
}

void RnntDecodingStreams::FormatOutput(std::vector<int32_t> &num_frames,
                                       FsaVec *ofsa, Array1<int32_t> *out_map) {
  NVTX_RANGE("FormatOutput");

  K2_CHECK(!attached_) << "You can only get outputs after calling Detach()";

  K2_CHECK(ofsa);
  K2_CHECK(out_map);

  UpdatePrevFrames(num_frames);

  int32_t frames = prev_frames_.size();

  auto last_frame_shape = prev_frames_[frames - 1]->shape;
  auto pre_final_arcs_shape = ComposeRaggedShapes(
      RemoveAxis(last_frame_shape, 1),
      RegularRaggedShape(c_, last_frame_shape.NumElements(), 1));

  auto stream_state_shape = RegularRaggedShape(c_, num_streams_, 1);
  auto state_arc_shape =
      RegularRaggedShape(c_, stream_state_shape.NumElements(), 0);
  auto final_arcs_shape =
      ComposeRaggedShapes(stream_state_shape, state_arc_shape);

  RaggedShape oshape;
  // see documentation of Stack() in ragged_ops.h for explanation.
  Array1<uint32_t> oshape_merge_map;

  Array1<ArcInfo *> arcs_data_ptrs(GetCpuContext(), frames);
  ArcInfo **arcs_data_ptrs_data = arcs_data_ptrs.Data();

  {
    // each of these have 3 axes.
    std::vector<RaggedShape *> arcs_shapes(frames + 2);
    for (int32_t t = 0; t < frames; t++) {
      arcs_shapes[t] = &(prev_frames_[t]->shape);
      arcs_data_ptrs_data[t] = prev_frames_[t]->values.Data();
    }

    arcs_shapes[frames] = &pre_final_arcs_shape;
    arcs_shapes[frames + 1] = &final_arcs_shape;

    // oshape is a 4-axis ragged tensor which is indexed:
    //   oshape[stream][t][state_idx][arc_idx]
    int32_t axis = 1;
    oshape = Stack(axis, frames + 2, arcs_shapes.data(), &oshape_merge_map);
  }

  int32_t num_arcs = oshape.NumElements();

  // transfer to GPU if we're using a GPU
  arcs_data_ptrs = arcs_data_ptrs.To(c_);
  arcs_data_ptrs_data = arcs_data_ptrs.Data();
  uint32_t *oshape_merge_map_data = oshape_merge_map.Data();

  *out_map = Array1<int32_t>(c_, num_arcs);
  int32_t *out_map_data = out_map->Data();

  int32_t *oshape_row_ids3 = oshape.RowIds(3).Data(),
          *oshape_row_ids2 = oshape.RowIds(2).Data(),
          *oshape_row_ids1 = oshape.RowIds(1).Data(),
          *oshape_row_splits2 = oshape.RowSplits(2).Data(),
          *oshape_row_splits1 = oshape.RowSplits(1).Data();

  Array1<Arc> arcs_out(c_, num_arcs);
  Arc *arcs_out_data = arcs_out.Data();
  Arc **graphs_arcs_data = graphs_.values.Data();

  K2_EVAL(
      c_, num_arcs, lambda_set_arcs, (int32_t oarc_idx0123) {
        int32_t oarc_idx012 = oshape_row_ids3[oarc_idx0123],
                oarc_idx01 = oshape_row_ids2[oarc_idx012],
                oarc_idx0 = oshape_row_ids1[oarc_idx01],
                oarc_idx0x = oshape_row_splits1[oarc_idx0],
                oarc_idx0xx = oshape_row_splits2[oarc_idx0x],
                oarc_idx1 = oarc_idx01 - oarc_idx0x,
                oarc_idx01x_next = oshape_row_splits2[oarc_idx01 + 1];

        int32_t m = oshape_merge_map_data[oarc_idx0123],
                // actually we won't get t == frames + 1
                // here since those frames have no arcs.
            t = m % (frames + 2),
                // arc_idx012 into prev_frames_ arcs on time t, index of the arc
                // on that frame.
            arcs_idx012 = m / (frames + 2);

        K2_CHECK_EQ(t, oarc_idx1);

        ArcInfo arc_info;
        Arc arc;

        // all arcs in t == frames point to final state
        if (t == frames) {
          arc.src_state = oarc_idx012 - oarc_idx0xx;
          arc.dest_state = oarc_idx01x_next - oarc_idx0xx;
          arc.label = -1;
          arc.score = 0;
          arc_info.graph_arc_idx01 = -1;
        } else {
          const Arc *graph_arcs_data = graphs_arcs_data[oarc_idx0];
          const ArcInfo *arcs_data = arcs_data_ptrs_data[t];

          arc_info = arcs_data[arcs_idx012];
          arc.src_state = oarc_idx012 - oarc_idx0xx;
          // Note: the idx1 w.r.t. the frame's `arcs` is an idx2 w.r.t.
          // `oshape`.
          int32_t dest_state_idx012 = oarc_idx01x_next + arc_info.dest_state;
          arc.dest_state = dest_state_idx012 - oarc_idx0xx;

          // graph_arc_idx01 == -1 means this is a implicit epsilon self-loop
          if (arc_info.graph_arc_idx01 == -1) {
            arc.label = 0;
          } else {
            arc.label = graph_arcs_data[arc_info.graph_arc_idx01].label;
          }
          arc.score = arc_info.score;
        }
        out_map_data[oarc_idx0123] = arc_info.graph_arc_idx01;
        arcs_out_data[oarc_idx0123] = arc;
      });

  // Remove axis 1, which corresponds to time.
  *ofsa = FsaVec(RemoveAxis(oshape, 1), arcs_out);
}

}  // namespace rnnt_decoding
}  // namespace k2