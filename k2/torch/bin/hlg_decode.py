import argparse
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import k2
import kaldifeat
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from k2 import (
    get_lattice,
    one_best_decoding,
    get_aux_labels,
)


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
        "--wav-scp",
        type=str,
        help="""The audio lists to transcribe in wav.scp format""",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="""
        The file to write out results to, only used when giving --wav-scp
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="The number of wavs in a batch.",
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


def decode_one_batch(
    params: object,
    batch: List[Tuple[str, str]],
    model: torch.nn.Module,
    feature_extractor: kaldifeat.Fbank,
    decoding_graph: k2.Fsa,
    token_sym_table: Optional[k2.SymbolTable] = None,
    word_sym_table: Optional[k2.SymbolTable] = None,
) -> Dict[str, str]:
    device = params.device
    filenames = [x[1] for x in batch]
    waves = read_sound_files(
        filenames=filenames, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    features = feature_extractor(waves)

    feature_len = []
    for f in features:
        feature_len.append(f.shape[0])

    features = pad_sequence(
        features, batch_first=True, padding_value=math.log(1e-10)
    )

    # Note: We don't use key padding mask for attention during decoding
    nnet_output, _, _ = model(features)

    log_prob = torch.nn.functional.log_softmax(nnet_output, dim=-1)
    log_prob_len = torch.tensor(feature_len) // params.subsampling_factor
    log_prob_len = log_prob_len.to(device)

    lattice = get_lattice(
        log_prob=log_prob,
        log_prob_len=log_prob_len,
        decoding_graph=decoding_graph,
        subsampling_factor=params.subsampling_factor,
    )
    best_path = one_best_decoding(lattice=lattice, use_double_scores=True)

    hyps = get_aux_labels(best_path)

    if params.method == "ctc-decoding":
        hyps = ["".join([token_sym_table[i] for i in ids]) for ids in hyps]
    else:
        assert params.method == "1best", params.method
        hyps = [" ".join([word_sym_table[i] for i in ids]) for ids in hyps]

    results = {}
    for i, hyp in enumerate(hyps):
        results[batch[i][0]] = hyp.replace("▁", " ").strip()
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.sample_rate = 16000
    args.subsampling_factor = 4
    args.feature_dim = 80
    args.num_classes = 500

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
        decoding_graph = k2.ctc_topo(
            max_token=max_token_id,
            device=device,
        )
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

    results = {}
    start = 0
    while start + args.batch_size <= len(wave_list):

        if start % 100 == 0:
            logging.info(f"Decoding progress: {start}/{len(wave_list)}.")

        res = decode_one_batch(
            params=args,
            batch=wave_list[start: start + args.batch_size],
            model=model,
            feature_extractor=fbank,
            decoding_graph=decoding_graph,
            token_sym_table=token_sym_table,
            word_sym_table=word_sym_table,
        )
        start += args.batch_size

        results.update(res)

    logging.info(f"results : {results}")

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
