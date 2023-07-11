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
    get_lattice,
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

    log_prob = torch.nn.functional.log_softmax(nnet_output, dim=-1)
    log_prob_len = torch.tensor(feature_len) // args.subsampling_factor
    log_prob_len = log_prob_len.to(device)

    if args.method == "ctc-decoding":
        logging.info("Use CTC decoding")
        max_token_id = args.num_classes - 1

        H = k2.ctc_topo(max_token=max_token_id, device=device,)

        lattice = get_lattice(
            log_prob=log_prob,
            log_prob_len=log_prob_len,
            decoding_graph=H,
            subsampling_factor=args.subsampling_factor,
        )

        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        token_ids = get_aux_labels(best_path)
        token_sym_table = k2.SymbolTable.from_file(args.tokens)

        hyps = ["".join([token_sym_table[i] for i in ids]) for ids in token_ids]

    else:
        assert args.method == "1best", args.method
        logging.info(f"Loading HLG from {args.HLG}")
        HLG = k2.Fsa.from_dict(torch.load(args.HLG, map_location="cpu"))
        HLG = HLG.to(device)

        lattice = get_lattice(
            log_prob=log_prob,
            log_prob_len=log_prob_len,
            decoding_graph=HLG,
            subsampling_factor=args.subsampling_factor,
        )

        if args.method == "1best":
            logging.info("Use HLG decoding")
            best_path = one_best_decoding(lattice=lattice, use_double_scores=True)

        hyps = get_aux_labels(best_path)
        word_sym_table = k2.SymbolTable.from_file(args.words_file)
        hyps = [" ".join([word_sym_table[i] for i in ids]) for ids in hyps]

    s = "\n"
    for filename, hyp in zip(args.sound_files, hyps):
        words = hyp.replace("▁", " ").strip()
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    torch.save(lattice.as_dict(), "offline.pt")

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
