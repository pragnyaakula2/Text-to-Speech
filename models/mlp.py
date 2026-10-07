"""Baseline 2: simple fully connected neural network (embedding -> sigmoid hidden -> flat output)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, vocab, text_len, T, n_bins, emb=32, hidden=256):
        super().__init__()
        self.kwargs = dict(vocab=vocab, text_len=text_len, T=T, n_bins=n_bins, emb=emb, hidden=hidden)
        self.T, self.n_bins = T, n_bins
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.hidden = nn.Linear(text_len * emb, hidden)
        self.out = nn.Linear(hidden, T * n_bins)      # stretched 1D spectrogram

    def forward(self, ids, target=None):
        h = torch.sigmoid(self.hidden(self.emb(ids).flatten(1)))
        return torch.sigmoid(self.out(h)).view(-1, self.T, self.n_bins)

    @torch.no_grad()
    def predict(self, ids):
        self.eval()
        return self.forward(ids)
