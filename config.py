"""Shared settings. Change LJ_DIR (or set the LJ_DIR env var) to where LJSpeech-1.1 lives."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
LJ_DIR = Path(os.environ.get("LJ_DIR", ROOT / "LJSpeech-1.1"))
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
WAV_DIR = ROOT / "outputs"

# audio
SR = 22050
N_FFT = 2048                 # -> 1025 frequency bins, same as the paper
HOP = 512
PREEMPH = 0.97               # FIR pre-emphasis filter y[t] = x[t] - 0.97 x[t-1]
N_BINS = N_FFT // 2 + 1

# dataset subset (paper: 200 sentences)
N_SENT = 200
MAX_SEC = 3.5                # keep clips short so SVR / NN stay trainable
MAX_CHARS = 50
T_FRAMES = int(MAX_SEC * SR / HOP) + 2   # fixed number of spectrogram frames (zero padded)
TEXT_LEN = 96                # fixed number of input symbols (zero padded)

SEED = 0

for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, WAV_DIR):
    d.mkdir(exist_ok=True, parents=True)
