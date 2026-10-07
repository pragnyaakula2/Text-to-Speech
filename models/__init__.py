"""Uniform build / save / load / predict helpers for the three models."""
import numpy as np
import torch

import config as C
from textproc import VOCAB_SIZE

from .mlp import MLP
from .seq2seq_attn import Seq2SeqAttn
from .svr import SVRModel

NAMES = ("svr", "nn", "seq2seq")


def build(name, kernel="poly"):
    if name == "svr":
        return SVRModel(kernel=kernel)
    if name == "nn":
        return MLP(VOCAB_SIZE, C.TEXT_LEN, C.T_FRAMES, C.N_BINS)
    if name == "seq2seq":
        return Seq2SeqAttn(VOCAB_SIZE, C.N_BINS, C.T_FRAMES)
    raise ValueError(name)


def ckpt_path(name):
    return C.CKPT_DIR / (f"{name}.joblib" if name == "svr" else f"{name}.pt")


def save(name, model):
    p = ckpt_path(name)
    if name == "svr":
        model.save(p)
    else:
        torch.save({"kwargs": model.kwargs, "state": model.state_dict()}, p)
    return p


def load(name, device="cpu"):
    p = ckpt_path(name)
    if name == "svr":
        return SVRModel.load(p)
    blob = torch.load(p, map_location=device)
    cls = MLP if name == "nn" else Seq2SeqAttn
    m = cls(**blob["kwargs"])
    m.load_state_dict(blob["state"])
    return m.to(device).eval()


def predict(name, model, ids, device="cpu"):
    """ids: (N, TEXT_LEN) int numpy -> (N, T, n_bins) float numpy in [0, 1]."""
    if name == "svr":
        return model.predict(ids.astype(np.float64))
    out = model.predict(torch.as_tensor(ids, dtype=torch.long, device=device))
    return out.cpu().numpy()
