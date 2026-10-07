"""Text -> integer symbol ids. Each word is randomly mapped to CMU phonemes or plain letters
(as in the paper), then the sequence is zero padded to a fixed length."""
import random
import re

import cmudict
import numpy as np

CHARS = list("abcdefghijklmnopqrstuvwxyz '.,?!-")
PHONES = cmudict.symbols()
SYMBOLS = ["<pad>"] + CHARS + ["@" + p for p in PHONES]
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = len(SYMBOLS)

_TOKEN_RE = re.compile(r"[a-z']+|[.,?!\-]")
_CMU = None


def _cmu():
    global _CMU
    if _CMU is None:
        _CMU = cmudict.dict()
    return _CMU


def encode_text(text, text_len, p_phoneme=0.5, rng=None):
    """Return int64 array of shape (text_len,)."""
    rng = rng or random.Random(0)
    d = _cmu()
    space = SYM2ID[" "]
    ids = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in ".,?!-":
            ids.append(SYM2ID[tok])
            continue
        if ids and ids[-1] != space:
            ids.append(space)
        prons = d.get(tok)
        if prons and rng.random() < p_phoneme:
            ids += [SYM2ID["@" + p] for p in prons[0]]
        else:
            ids += [SYM2ID[c] for c in tok if c in SYM2ID]
    ids = ids[:text_len]
    return np.array(ids + [0] * (text_len - len(ids)), dtype=np.int64)
