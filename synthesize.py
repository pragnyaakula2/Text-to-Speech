"""Text -> wav.   python synthesize.py "hello world" --model seq2seq   (or --model all)"""
import argparse
import json
import random
import re

import numpy as np
import torch

import config as C
import models
from postprocess import save_wav, spec_to_wav
from textproc import encode_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--model", default="seq2seq", choices=list(models.NAMES) + ["all"])
    ap.add_argument("--p_phoneme", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_iter", type=int, default=60, help="Griffin-Lim iterations")
    ap.add_argument("--train_idx", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = json.load(open(C.DATA_DIR / "meta.json"))["norm"]
    if args.train_idx is not None:
        d = np.load(C.DATA_DIR / "dataset.npz")
        ids = d["X"][args.train_idx:args.train_idx + 1]
        args.text = str(d["texts"][args.train_idx])
    else:
        ids = encode_text(
        args.text,
        C.TEXT_LEN,
        args.p_phoneme,
        random.Random(args.seed)
    )[None]
    slug = re.sub(r"[^a-z0-9]+", "_", args.text.lower()).strip("_")[:30]

    names = models.NAMES if args.model == "all" else [args.model]
    for name in names:
        model = models.load(name, device)
        S = models.predict(name, model, ids, device)[0]
        y = spec_to_wav(S, norm, args.n_iter)
        out = C.WAV_DIR / f"{slug}_{name}.wav"
        save_wav(out, y)
        print("wrote", out)


if __name__ == "__main__":
    main()
