import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import k2
import kaldifeat
import torch
import torchaudio
from k2 import (
    DecodeStateInfo,
    OnlineDenseIntersecter,
    one_best_decoding,
    get_aux_labels,
)
from torch.nn.utils.rnn import pad_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model",
        type=str,
        required=True,
        help="Path to the jit script model.",
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
        "--wav-scp",
        type=str,
        help="""The audio lists to transcribe in wav.scp format""",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="The file to write out results to, only used when giving --wav-scp",
    )

    parser.add_argument(
        "--print-partial",
        dest="print_partial",
        action="store_true",
        help="Whether print partial results.",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="*",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


@dataclass
class DecodeStream:
    # The identifier of wavs.
    utt_id: str
    # The total number of frames for current nnet_output.
    num_frames: int
    # The output of encoder.
    nnet_output: torch.Tensor
    # Current position, index in to feature.
    position: int
    # Decode state for intersect_dense_pruned.
    state_info: DecodeStateInfo
    # Current decoding result.
    result: str


def decode_one_chunk(
    params: object,
    intersector: k2.OnlineDenseIntersecter,
    streams: List[DecodeStream],
    token_sym_table: Optional[k2.SymbolTable] = None,
    word_sym_table: Optional[k2.SymbolTable] = None,
) -> List[int]:
    assert params.num_streams == intersector.num_streams, (
        params.num_streams,
        intersector.num_streams,
    )
    current_state_infos = []
    current_nnet_outputs = []
    current_num_frames = []
    finised_streams = []
    for i, stream in enumerate(streams):
        start = stream.position
        if (stream.num_frames - stream.position) <= params.chunk_size:
            current_num_frames.append(stream.num_frames - stream.position)
            end = stream.num_frames
            stream.position = stream.num_frames
            finised_streams.append(i)
        else:
            current_num_frames.append(params.chunk_size)
            end = stream.position + params.chunk_size
            stream.position += params.chunk_size
        current_state_infos.append(stream.state_info)
        current_nnet_outputs.append(stream.nnet_output[start:end, :])

    while len(current_num_frames) < params.num_streams:
        current_num_frames.append(0)
        current_nnet_outputs.append(
            torch.zeros(
                (params.chunk_size, params.num_classes), device=params.device,
            )
        )
        current_state_infos.append(DecodeStateInfo())

    current_nnet_outputs = pad_sequence(current_nnet_outputs, batch_first=True)
    supervision_segments = torch.tensor(
        # seq_index, start_time, duration
        [[i, 0, current_num_frames[i]] for i in range(params.num_streams)],
        dtype=torch.int32,
    )
    dense_fsa_vec = k2.DenseFsaVec(current_nnet_outputs, supervision_segments)
    lattice, current_state_infos = intersector.decode(
        dense_fsa_vec, current_state_infos
    )

    best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
    symbol_ids = get_aux_labels(best_path)

    if params.method == "ctc-decoding":
        assert token_sym_table is not None
        hyps = [
            "".join([token_sym_table[i] for i in ids]) for ids in symbol_ids
        ]
    else:
        assert word_sym_table is not None
        assert params.method == "1best", params.method
        hyps = [
            " ".join([word_sym_table[i] for i in ids]) for ids in symbol_ids
        ]
    for i, stream in enumerate(streams):
        stream.state_info = current_state_infos[i]
        stream.result = hyps[i].replace("▁", " ").strip()
    return finised_streams


def decode_dataset(
    params: object,
    waves: List[Tuple[str, str]],
    model: torch.nn.Module,
    feature_extractor: kaldifeat.Fbank,
    intersector: k2.OnlineDenseIntersecter,
    token_sym_table: Optional[k2.SymbolTable] = None,
    word_sym_table: Optional[k2.SymbolTable] = None,
) -> Dict[str, str]:
    results = {}
    decode_streams = []
    wave_index = 0
    while True:
        if wave_index < len(waves) and len(decode_streams) < params.num_streams:
            data, sample_rate = torchaudio.load(waves[wave_index][1])
            assert (
                sample_rate == params.sample_rate
            ), f"expected sample rate: {params.sample_rate}. Given: {sample_rate}"
            data = data[0].to(params.device)
            feature = feature_extractor(data)
            nnet_output, _, _ = model(feature.unsqueeze(0))
            decode_streams.append(
                DecodeStream(
                    utt_id=waves[wave_index][0],
                    num_frames=nnet_output.shape[1],
                    nnet_output=nnet_output[0],
                    position=0,
                    state_info=DecodeStateInfo(),
                    result="",
                )
            )
            wave_index += 1
            if wave_index % 100 == 0:
                logging.info(f"Decoding progress: {wave_index}/{len(waves)}.")
            continue

        if len(decode_streams) == 0:
            break

        finised_streams = decode_one_chunk(
            params=params,
            intersector=intersector,
            streams=decode_streams,
            token_sym_table=token_sym_table,
            word_sym_table=word_sym_table,
        )

        if params.print_partial:
            s = "\n"
            for stream in decode_streams:
                s += f"{stream.utt_id}:\t{stream.result}\n\n"
            logging.info(s)

        if finised_streams:
            finised_streams = sorted(finised_streams, reverse=True)
            for j in finised_streams:
                results[decode_streams[j].utt_id] = decode_streams[j].result
                del decode_streams[j]
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.sample_rate = 16000
    args.subsampling_factor = 4
    args.feature_dim = 80
    args.num_classes = 500
    args.chunk_size = 16

    wave_list: List[Tuple[str, str]] = []
    if args.wav_scp is not None:
        assert os.path.isfile(
            args.wav_scp
        ), f"wav_scp not exists : {args.wav_scp}"
        assert (
            args.output_file is not None
        ), "You should provide output_file when using wav_scp"
        with open(args.wav_scp, "r") as f:
            for line in f:
                toks = line.strip().split()
                assert len(toks) == 2, toks
                if not os.path.isfile(toks[1]):
                    logging.warning(f"File {toks[1]} not exists, skipping.")
                    continue
                wave_list.append(toks)
    else:
        assert len(args.sound_files) > 0, "No wav_scp or waves provided."
        for i, f in enumerate(args.sound_files):
            if not os.path.isfile(f):
                logging.warning(f"File {f} not exists, skipping.")
                continue
            wave_list.append((i, f))

    # logging.info(f"wave_list : {wave_list}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    args.device = device

    logging.info(f"params : {args}")

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

    token_sym_table = None
    word_sym_table = None
    if args.method == "ctc-decoding":
        logging.info("Use CTC decoding")
        max_token_id = args.num_classes - 1
        decoding_graph = k2.ctc_topo(max_token=max_token_id, device=device,)
        token_sym_table = k2.SymbolTable.from_file(args.tokens)
    else:
        assert args.method == "1best", args.method
        logging.info(f"Loading HLG from {args.HLG}")
        decoding_graph = k2.Fsa.from_dict(
            torch.load(args.HLG, map_location="cpu")
        )
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

    results = decode_dataset(
        params=args,
        waves=wave_list,
        model=model,
        feature_extractor=fbank,
        intersector=intersector,
        token_sym_table=token_sym_table,
        word_sym_table=word_sym_table,
    )

    if args.wav_scp is not None:
        output_dir = os.path.dirname(args.output_file)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for x in wave_list:
                f.write(x[0] + "\t" + results[x[0]] + "\n")
        logging.info(f"Decoding results are written to {args.output_file}")
    else:
        s = "\n"
        logging.info(f"results : {results}")
        for x in wave_list:
            s += f"{x[1]}:\n{results[x[0]]}\n\n"
        logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
