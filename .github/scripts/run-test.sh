#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"

git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"
git lfs pull --include "data/lang_bpe_500/words.txt"

git lfs pull --include "data/lm/G_4_gram.pt"
popd

log "Test CTC decode (librispeech)"

./build/bin/ctc_decode \
  --use_gpu false \
  --nn_model $repo/exp/cpu_jit.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav


log "Test HLG decode (librispeech)"

./build/bin/hlg_decode \
  --use_gpu false \
  --nn_model $repo/exp/cpu_jit.pt \
  --hlg $repo/data/lang_bpe_500/HLG.pt \
  --word_table $repo/data/lang_bpe_500/words.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

if [ $(uname) == "Darwin" ]; then
  # GitHub only provides 7 GB RAM for Linux/Windows
  # It has 14 GB RAM for macOS. This test requires a lot of RAM.
  log "Test n-gram LM rescore (librispeech)"
  ./build/bin/ngram_lm_rescore \
    --use_gpu false \
    --nn_model $repo/exp/cpu_jit.pt \
    --hlg $repo/data/lang_bpe_500/HLG.pt \
    --g $repo/data/lm/G_4_gram.pt \
    --ngram_lm_scale 1.0 \
    --word_table $repo/data/lang_bpe_500/words.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav

  log "Test n-gram LM rescore + attention rescore (librispeech)"
  ./build/bin/attention_rescore \
    --use_gpu false \
    --nn_model $repo/exp/cpu_jit.pt \
    --hlg $repo/data/lang_bpe_500/HLG.pt \
    --g $repo/data/lm/G_4_gram.pt \
    --ngram_lm_scale 1.0 \
    --attention_scale 1.0 \
    --num_paths 100 \
    --nbest_scale 0.5 \
    --word_table $repo/data/lang_bpe_500/words.txt \
    --sos_id 1 \
    --eos_id 1 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
fi

log "Streaming CTC decoding"

./build/bin/online_decode \
  --use_ctc_decoding true \
  --jit_pt $repo/exp/cpu_jit.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Streaming HLG decoding"

./build/bin/online_decode \
  --use_ctc_decoding false \
  --jit_pt $repo/exp/cpu_jit.pt \
  --hlg $repo/data/lang_bpe_500/HLG.pt \
  --word_table $repo/data/lang_bpe_500/words.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf repo

# Now for RNN-T

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"
popd

log "Test RNN-T decoding"

./build/bin/pruned_stateless_transducer \
  --use-gpu=false \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

./build/bin/rnnt_demo \
  --use_lg false \
  --jit_pt $repo/exp/cpu_jit.pt \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

./build/bin/rnnt_demo \
  --use_lg true \
  --jit_pt $repo/exp/cpu_jit.pt \
  --lg $repo/data/lang_bpe_500/LG.pt \
  --word_table $repo/data/lang_bpe_500/words.txt \
  --beam 8 \
  --max_contexts 8 \
  --max_states 64 \
  $repo/test_wavs/1089-134686-0001.wav  \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav
