"""
Encoder-decoder (seq2seq) models implemented from scratch in PyTorch.

    Input description -> Embedding -> RNN encoder -> [bridge] -> RNN decoder -> Linear -> softmax

Two decoders share the same encoder:
  * baseline: the decoder only receives the encoder's FINAL state as its initial state
              (a single fixed-size vector summarises the whole sentence);
  * attention: at every step the decoder also looks at ALL encoder states through
               additive (Bahdanau) attention and builds a context vector.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import EOS_ID, PAD_ID, SOS_ID, UNK_ID


def make_rnn(kind, input_size, hidden, layers, bidirectional=False, dropout=0.0):
    cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
    return cls(input_size, hidden, num_layers=layers, batch_first=True,
               bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, layers, rnn="lstm", bidirectional=False, dropout=0.2):
        super().__init__()
        self.rnn_kind, self.layers, self.bidir = rnn, layers, bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.rnn = make_rnn(rnn, emb_dim, hid_dim, layers, bidirectional, dropout)
        self.out_dim = hid_dim * (2 if bidirectional else 1)

    def forward(self, src, src_len):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, state = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        return out, state  # out: [B, S, H*dirs]; state: (h, c) or h with shape [L*dirs, B, H]


class AdditiveAttention(nn.Module):
    """score(s_t, h_i) = v^T tanh(W_s s_t + W_h h_i);  alpha = softmax(score);  ctx = sum alpha_i h_i"""

    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, att_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, att_dim, bias=True)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_out, mask, enc_proj=None):
        # dec_hidden: [B, H]; enc_out: [B, S, E]; mask: [B, S] (True = real token)
        if enc_proj is None:
            enc_proj = self.W_h(enc_out)
        scores = self.v(torch.tanh(enc_proj + self.W_s(dec_hidden).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))   # padding gets zero weight
        alpha = F.softmax(scores, dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)
        return ctx, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, layers, enc_dim, rnn="lstm",
                 attention=False, att_dim=None, dropout=0.2):
        super().__init__()
        self.rnn_kind, self.layers, self.use_attention = rnn, layers, attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        rnn_in = emb_dim + (enc_dim if attention else 0)
        self.rnn = make_rnn(rnn, rnn_in, hid_dim, layers, False, dropout)
        self.attention = AdditiveAttention(enc_dim, hid_dim, att_dim or hid_dim) if attention else None
        out_in = hid_dim + (enc_dim + emb_dim if attention else 0)
        self.out = nn.Linear(out_in, vocab_size)

    def step(self, prev_tok, state, enc_out=None, mask=None, enc_proj=None):
        """One decoding step. prev_tok: [B]. Returns logits [B, V], new state, attention [B, S] or None."""
        emb = self.dropout(self.embedding(prev_tok)).unsqueeze(1)           # [B, 1, E]
        alpha = None
        if self.use_attention:
            top = state[0][-1] if self.rnn_kind == "lstm" else state[-1]      # last layer hidden [B, H]
            ctx, alpha = self.attention(top, enc_out, mask, enc_proj)
            rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)
        else:
            rnn_in = emb
        out, state = self.rnn(rnn_in, state)
        h = out.squeeze(1)
        feats = torch.cat([h, ctx, emb.squeeze(1)], dim=-1) if self.use_attention else h
        return self.out(self.dropout(feats)), state, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=128, hid_dim=256, layers=1, rnn="lstm",
                 bidirectional=False, attention=False, dropout=0.2, word_dropout=0.0):
        super().__init__()
        self.word_dropout = word_dropout
        self.cfg = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, emb_dim=emb_dim, hid_dim=hid_dim,
                        layers=layers, rnn=rnn, bidirectional=bidirectional, attention=attention,
                        dropout=dropout, word_dropout=word_dropout)
        self.rnn_kind, self.layers, self.hid = rnn, layers, hid_dim
        self.encoder = Encoder(src_vocab, emb_dim, hid_dim, layers, rnn, bidirectional, dropout)
        self.decoder = Decoder(tgt_vocab, emb_dim, hid_dim, layers, self.encoder.out_dim, rnn,
                               attention, hid_dim, dropout)
        # bridge: maps the encoder final state (possibly 2 directions concatenated) to the decoder size
        self.bridge_h = nn.Linear(self.encoder.out_dim, hid_dim)
        self.bridge_c = nn.Linear(self.encoder.out_dim, hid_dim) if rnn == "lstm" else None

    # ------------------------------------------------------------------ encoding
    def _bridge_one(self, h, layer):
        # h: [L*dirs, B, H]  ->  [L, B, H_dec]
        L, d = self.layers, 2 if self.encoder.bidir else 1
        h = h.view(L, d, h.size(1), h.size(2))
        h = torch.cat([h[:, i] for i in range(d)], dim=-1)      # concat directions
        return torch.tanh(layer(h))

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        if self.rnn_kind == "lstm":
            state = (self._bridge_one(state[0], self.bridge_h), self._bridge_one(state[1], self.bridge_c))
        else:
            state = self._bridge_one(state, self.bridge_h)
        mask = src != PAD_ID
        enc_proj = self.decoder.attention.W_h(enc_out) if self.decoder.use_attention else None
        return enc_out, state, mask, enc_proj

    # ------------------------------------------------------------------- training
    def forward(self, src, src_len, tgt, tf_ratio=1.0):
        """tgt: [B, T] = <SOS> w1 ... wn <EOS> <PAD>*. Returns logits for positions 1..T-1: [B, T-1, V]."""
        if self.training and self.word_dropout > 0:   # randomly replace input words by <UNK>
            drop = (torch.rand(src.shape) < self.word_dropout) & (src != PAD_ID)
            src = src.masked_fill(drop, UNK_ID)
        enc_out, state, mask, enc_proj = self.encode(src, src_len)
        B, T = tgt.shape
        prev = tgt[:, 0]
        logits = []
        for t in range(1, T):
            out, state, _ = self.decoder.step(prev, state, enc_out, mask, enc_proj)
            logits.append(out)
            # teacher forcing: with prob tf_ratio feed the gold token, otherwise the model's own prediction
            if tf_ratio >= 1.0 or random.random() < tf_ratio:
                prev = tgt[:, t]
            else:
                prev = out.argmax(-1)
        return torch.stack(logits, dim=1)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len=60, return_attention=False):
        """Batched greedy decoding: pick the arg-max token at every step, stop at <EOS>."""
        self.eval()
        enc_out, state, mask, enc_proj = self.encode(src, src_len)
        B = src.size(0)
        prev = torch.full((B,), SOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        tokens, atts = [], []
        for _ in range(max_len):
            out, state, alpha = self.decoder.step(prev, state, enc_out, mask, enc_proj)
            prev = out.argmax(-1)
            prev = prev.masked_fill(done, PAD_ID)
            tokens.append(prev)
            if alpha is not None:
                atts.append(alpha)
            done = done | (prev == EOS_ID)
            if done.all():
                break
        tokens = torch.stack(tokens, dim=1)
        if return_attention and atts:
            return tokens, torch.stack(atts, dim=1)   # [B, T, S]
        return tokens

    @torch.no_grad()
    def beam_search(self, src, src_len, beam=5, max_len=60, length_norm=True):
        """Beam search for ONE example (batch size 1). Returns the best token-id list (without <SOS>/<EOS>)."""
        self.eval()
        enc_out, state, mask, enc_proj = self.encode(src, src_len)
        dev = src.device
        is_lstm = self.rnn_kind == "lstm"
        hyps = [([SOS_ID], 0.0, state)]          # (tokens, sum of log-probs, decoder state [L,1,H])
        finished = []                            # (tokens, score)
        for _ in range(max_len):
            n = len(hyps)
            prev = torch.tensor([h[0][-1] for h in hyps], device=dev)
            if is_lstm:
                st = (torch.cat([h[2][0] for h in hyps], dim=1), torch.cat([h[2][1] for h in hyps], dim=1))
            else:
                st = torch.cat([h[2] for h in hyps], dim=1)
            ep = enc_proj.expand(n, -1, -1) if enc_proj is not None else None
            out, new_state, _ = self.decoder.step(prev, st, enc_out.expand(n, -1, -1), mask.expand(n, -1), ep)
            logp = F.log_softmax(out, dim=-1)
            cands = []
            for i, (_, score, _) in enumerate(hyps):
                lp, ids = logp[i].topk(beam)
                cands.extend((score + l, i, t) for l, t in zip(lp.tolist(), ids.tolist()))
            cands.sort(key=lambda c: -c[0])
            new_hyps = []
            for score, i, tid in cands:
                toks = hyps[i][0] + [tid]
                if tid == EOS_ID:
                    finished.append((toks[1:-1], score))
                else:
                    if is_lstm:
                        s_i = (new_state[0][:, i:i + 1].contiguous(), new_state[1][:, i:i + 1].contiguous())
                    else:
                        s_i = new_state[:, i:i + 1].contiguous()
                    new_hyps.append((toks, score, s_i))
                if len(new_hyps) == beam:
                    break
            hyps = new_hyps
            # stop when no live hypothesis can still beat the best finished one (log-probs only go down)
            if not hyps or (finished and max(f[1] for f in finished) >= max(h[1] for h in hyps)):
                break
        if not finished:
            finished = [(h[0][1:], h[1]) for h in hyps]
        key = (lambda f: f[1] / max(1, len(f[0]) + 1)) if length_norm else (lambda f: f[1])
        return max(finished, key=key)[0]
