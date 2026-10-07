"""Build data/dataset.npz from LJ Speech.

X : (N, TEXT_LEN)            symbol ids (random phoneme / letter mapping, zero padded)
Y : (N, T_FRAMES, 1025)      log-magnitude spectrogram scaled to [0, 1] (zero padded)
lens : (N,)                  real number of frames per clip
"""
import argparse
import json
import random

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import lfilter

import config as C
from textproc import encode_text


def wav_to_spec(y):
    """FIR pre-emphasis filter -> STFT magnitude, returned as (frames, bins)."""
    y = lfilter([1.0, -C.PREEMPH], [1.0], y)
    S = np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP, win_length=C.N_FFT))
    return S.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lj_dir", default=str(C.LJ_DIR))
    ap.add_argument("--n", type=int, default=C.N_SENT)
    args = ap.parse_args()

    from pathlib import Path
    lj = Path(args.lj_dir)
    meta_file = lj / "metadata.csv"
    assert meta_file.exists(), f"{meta_file} not found - set --lj_dir or LJ_DIR"
    lines = [l.rstrip("\n").split("|") for l in open(meta_file, encoding="utf-8")]
    rng = random.Random(C.SEED)
    rng.shuffle(lines)

    texts, specs = [], []
    for parts in lines:
        if len(texts) >= args.n:
            break
        wav_id = parts[0]
        text = parts[2] if len(parts) > 2 and parts[2] else parts[1]
        if len(text) > C.MAX_CHARS:
            continue
        path = lj / "wavs" / f"{wav_id}.wav"
        if sf.info(path).duration > C.MAX_SEC + 1.0:   # cheap pre-check before decoding
            continue
        y, _ = librosa.load(path, sr=C.SR)
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) / C.SR > C.MAX_SEC:
            continue
        S = wav_to_spec(y)
        if S.shape[0] > C.T_FRAMES:
            continue
        texts.append(text)
        specs.append(S)

    n = len(texts)
    print(f"selected {n} clips (asked for {args.n})")
    if n < args.n:
        print("WARNING: fewer clips than requested; relax MAX_CHARS / MAX_SEC in config.py")

    logs = [np.log1p(S) for S in specs]
    norm = float(max(l.max() for l in logs))
    Y = np.zeros((n, C.T_FRAMES, C.N_BINS), dtype=np.float32)
    lens = np.zeros(n, dtype=np.int64)
    for i, l in enumerate(logs):
        Y[i, : l.shape[0]] = l / norm
        lens[i] = l.shape[0]

    trng = random.Random(C.SEED)
    X = np.stack([encode_text(t, C.TEXT_LEN, 0.5, trng) for t in texts])

    np.savez_compressed(C.DATA_DIR / "dataset.npz", X=X, Y=Y, lens=lens, texts=np.array(texts))
    json.dump({"norm": norm}, open(C.DATA_DIR / "meta.json", "w"))
    print("X", X.shape, "Y", Y.shape, "norm", round(norm, 3))
    print("saved ->", C.DATA_DIR / "dataset.npz")


if __name__ == "__main__":
    main()
