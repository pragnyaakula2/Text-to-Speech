"""Spectrogram -> wav (Griffin-Lim + inverse pre-emphasis).

Run `python postprocess.py --roundtrip` to check the pipeline on a real training clip:
audio -> spectrogram -> wav. That is the quality ceiling for every model.
"""
import argparse
import json

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

import config as C


def spec_to_wav(S01, norm, n_iter=60):
    """S01: (frames, bins) in [0,1] -> float waveform."""
    mag = np.expm1(np.clip(S01, 0, 1) * norm).T ** 1.3      # sharpen
    y = librosa.griffinlim(mag, n_iter=100)             # was n_iter=60
    y = lfilter([1.0], [1.0, -C.PREEMPH], y)             # undo pre-emphasis
    return y / (np.max(np.abs(y)) + 1e-9) * 0.9


def save_wav(path, y):
    wavfile.write(str(path), C.SR, (y * 32767).astype(np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    args = ap.parse_args()
    if not args.roundtrip:
        ap.print_help()
        return
    d = np.load(C.DATA_DIR / "dataset.npz")
    norm = json.load(open(C.DATA_DIR / "meta.json"))["norm"]
    S = d["Y"][args.idx][: d["lens"][args.idx]]
    out = C.WAV_DIR / f"roundtrip_{args.idx}.wav"
    save_wav(out, spec_to_wav(S, norm))
    print("text:", d["texts"][args.idx])
    print("wrote", out)


if __name__ == "__main__":
    main()
