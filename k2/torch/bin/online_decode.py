import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from k2 import (
    DecodeStateInfo,
    OnlineDenseIntersecter,
    one_best_decoding,
    get_aux_labels,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model", type=str, required=True, help="Path to the jit script model. "
    )

    parser.add_argument(
        "--words-file",
        type=str,
        help="""Path to words.txt.
        Used only when method is not ctc-decoding.
        """,
    )

    parser.add_argument(
        "--HLG",
        type=str,
        help="""Path to HLG.pt.
        Used only when method is not ctc-decoding.
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.
        Used only when method is ctc-decoding.
        """,
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method.
        Possible values are:
        (0) ctc-decoding - Use CTC decoding. It uses a sentence
            piece model, i.e., lang_dir/bpe.model, to convert
            word pieces to words. It needs neither a lexicon
            nor an n-gram LM.
        (1) 1best - Use the best path as decoding output. Only
            the transformer encoder output is used for decoding.
            We call it HLG decoding.
        """,
    )

    parser.add_argument(
        "--num-streams",
        type=int,
        default=2,
        help="""The number of streams that can be run in parallel.""",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.sample_rate = 16000
    args.subsampling_factor = 4
    args.feature_dim = 80
    args.num_classes = 500
    args.chunk_size = 10

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = torch.jit.load(args.nn_model)
    model = model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = args.sample_rate
    opts.mel_opts.num_bins = args.feature_dim

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {args.sound_files}")
    waves = read_sound_files(
        filenames=args.sound_files, expected_sample_rate=args.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)

    feature_len = []
    for f in features:
        feature_len.append(f.shape[0])

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))

    # Note: We don't use key padding mask for attention during decoding
    nnet_output, _, _ = model(features)
    num_frames = [x // args.subsampling_factor for x in feature_len]
    T = nnet_output.shape[1]

    if args.method == "ctc-decoding":
        logging.info("Use CTC decoding")
        max_token_id = args.num_classes - 1
        decoding_graph = k2.ctc_topo(
            max_token=max_token_id,
            device=device,
        )
        token_sym_table = k2.SymbolTable.from_file(args.tokens)
    else:
        assert args.method == "1best", args.method
        logging.info(f"Loading HLG from {args.HLG}")
        decoding_graph = k2.Fsa.from_dict(torch.load(args.HLG, map_location="cpu"))
        decoding_graph = decoding_graph.to(device)
        word_sym_table = k2.SymbolTable.from_file(args.words_file)
    decoding_graph = k2.Fsa.from_fsas([decoding_graph])

    intersector = k2.OnlineDenseIntersecter(
        decoding_graph=decoding_graph,
        num_streams=args.num_streams,
        search_beam=20,
        output_beam=8,
        min_active_states=30,
        max_active_states=10000,
    )

    state_infos = [DecodeStateInfo()] * len(waves)
    positions = [0] * len(waves)
    results = [""] * len(waves)

    while True:
        current_state_infos = []
        current_nnet_outputs = []
        current_wave_ids = []
        current_num_frames = []
        for i in range(len(waves)):
            if positions[i] == num_frames[i]:
                continue
            start = positions[i]
            if (num_frames[i] - positions[i]) <= args.chunk_size:
                current_num_frames.append(num_frames[i] - positions[i])
                end = num_frames[i]
                positions[i] = num_frames[i]
                state_infos[i].is_final = True
            else:
                current_num_frames.append(args.chunk_size)
                end = positions[i] + args.chunk_size
                positions[i] += args.chunk_size

            current_state_infos.append(state_infos[i])
            current_wave_ids.append(i)
            current_nnet_outputs.append(nnet_output[i, start:end, :])

            if len(current_wave_ids) == args.num_streams:
                break
        if len(current_wave_ids) == 0:
            break
        while len(current_num_frames) < args.num_streams:
            current_num_frames.append(1)
            current_nnet_outputs.append(
                torch.zeros(
                    (args.chunk_size, nnet_output.shape[2]),
                    device=nnet_output.device,
                )
            )
            current_state_infos.append(DecodeStateInfo())

        current_nnet_outputs = pad_sequence(current_nnet_outputs, batch_first=True)
        supervision_segments = torch.tensor(
            # seq_index, start_time, duration
            [[i, 0, current_num_frames[i]] for i in range(args.num_streams)],
            dtype=torch.int32,
        )
        logging.info(f"supervision_segments : {supervision_segments}")
        dense_fsa_vec = k2.DenseFsaVec(current_nnet_outputs, supervision_segments)
        lattice, current_state_infos = intersector.decode(
            dense_fsa_vec, current_state_infos
        )

        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        symbol_ids = get_aux_labels(best_path)

        if args.method == "ctc-decoding":
            hyps = ["".join([token_sym_table[i] for i in ids]) for ids in symbol_ids]
        else:
            assert args.method == "1best", args.method
            hyps = [" ".join([word_sym_table[i] for i in ids]) for ids in symbol_ids]
        logging.info(f"hyps : {hyps}")

        s = "\n"
        for i in range(len(current_wave_ids)):
            state_infos[current_wave_ids[i]] = current_state_infos[i]
            results[current_wave_ids[i]] = hyps[i].replace("▁", " ").strip()
            s += f"{args.sound_files[current_wave_ids[i]]}:\n"
            s += f"{results[current_wave_ids[i]]}\n\n"
        logging.info(s)

    s = "\n"
    for filename, hyp in zip(args.sound_files, results):
        s += f"{filename}:\n{hyp}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
