"""Advanced model: seq2seq with attention.

Encoder : embedding + LSTM(512)
Decoder : LSTMCell(512) with additive attention over the encoder outputs.
Training: teacher forcing (decoder input at step t is the true frame t-1).
Inference: free running (decoder input is its own previous prediction).
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class Seq2SeqAttn(nn.Module):
    def __init__(self, vocab, n_bins, T, emb=128, hidden=512, prenet=256):
        super().__init__()
        self.kwargs = dict(vocab=vocab, n_bins=n_bins, T=T, emb=emb, hidden=hidden, prenet=prenet)
        self.T, self.n_bins, self.hidden = T, n_bins, hidden
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.enc = nn.LSTM(emb, hidden, batch_first=True)
        self.prenet = nn.Linear(n_bins, prenet)
        self.W_enc = nn.Linear(hidden, hidden, bias=False)
        self.W_dec = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.cell = nn.LSTMCell(prenet + hidden, hidden)
        self.out = nn.Linear(hidden * 2, n_bins)

    def _encode(self, ids):
        enc_out, (h, c) = self.enc(self.emb(ids))
        return enc_out, self.W_enc(enc_out), (h[0], c[0]), ids == 0

    def _step(self, prev, state, enc_out, keys, mask):
        h, c = state
        e = self.v(torch.tanh(keys + self.W_dec(h).unsqueeze(1))).squeeze(-1)
        a = torch.softmax(e.masked_fill(mask, -1e9), dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        p = nnf.dropout(torch.relu(self.prenet(prev)), 0.5, training=self.training)
        h, c = self.cell(torch.cat([p, ctx], -1), (h, c))
        y = torch.sigmoid(self.out(torch.cat([h, ctx], -1)))
        return y, (h, c)

    def forward(self, ids, target):
        """Teacher forcing. target: (B, T, n_bins)."""
        enc_out, keys, state, mask = self._encode(ids)
        prev = torch.zeros(ids.size(0), self.n_bins, device=ids.device)
        outs = []
        for t in range(self.T):
            y, state = self._step(prev, state, enc_out, keys, mask)
            outs.append(y)
            prev = target[:, t]
        return torch.stack(outs, 1)

    @torch.no_grad()
    def predict(self, ids):
        self.eval()
        enc_out, keys, state, mask = self._encode(ids)
        prev = torch.zeros(ids.size(0), self.n_bins, device=ids.device)
        outs = []
        for _ in range(self.T):
            y, state = self._step(prev, state, enc_out, keys, mask)
            outs.append(y)
            prev = y
        return torch.stack(outs, 1)
