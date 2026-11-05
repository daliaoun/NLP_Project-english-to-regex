"""
Data loading, preprocessing, vocabulary construction and batching for NL -> regex.

Datasets
  * veriregex (default): our generated and verified dataset (src/generate.py), data/veriregex/*.jsonl.
    Quoted literals ('cat', "2024") are replaced by placeholders __w0__, __w1__, ... numbered by order of
    appearance, in the description and in the regex; they are put back after decoding.
  * NL-RX-Turk (Locascio et al., 2016): ~10k crowd-written descriptions; kept for comparison.
"""
import json
import random
import re
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "deep-regex" / "datasets"

PAD, UNK, SOS, EOS = "<PAD>", "<UNK>", "<SOS>", "<EOS>"
SPECIALS = [PAD, UNK, SOS, EOS]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

MAX_SRC_LEN = 40   # truncation limit (tokens); longest description has ~30 tokens
MAX_TGT_LEN = 60   # truncation limit (tokens) for the regex, EOS included

DATA = ROOT / "data" / "veriregex"

# -------------------------------------------------------------------- tokenisation
# a quoted literal: opening quote not preceded by a letter/digit (so the apostrophe of "don't" is not one),
# closing quote not followed by a letter/digit
_QUOTED = re.compile(r"""(?<![A-Za-z0-9])(?:'([^']+?)'|"([^"]+?)")(?![A-Za-z0-9])""")


def _straighten(text):
    return (text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"'))


def literal_spans(text):
    """the quoted literals of a description, in order of appearance"""
    return [a or b for a, b in _QUOTED.findall(_straighten(text))]


def anonymize(text):
    """Replace quoted literals by __w0__, __w1__, ... (same literal, same placeholder).
    Returns (anonymised text, list of literals in placeholder order)."""
    lits = []

    def sub(m):
        w = m.group(1) or m.group(2)
        if w not in lits:
            lits.append(w)
        return f" __w{lits.index(w)}__ "
    return _QUOTED.sub(sub, _straighten(text)), lits


def escape_literal(w):
    """write a literal in the NL-RX dialect: special characters go inside a character class"""
    return "".join(f"[{c}]" if c in ".*+?()[]{}|&~\\" else c for c in w)


def deanonymize(regex, lits):
    return re.sub(r"__w(\d)__", lambda m: escape_literal(lits[int(m.group(1))]) if int(m.group(1)) < len(lits)
                  else m.group(0), regex)


_SRC_TOKEN = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*|[^\sa-z0-9_]")


def normalize_src(text):
    """Lowercase and drop quotation marks. Quoted words ('dog', 'truck', ...) are
    placeholders in NL-RX and already identical to the literal used in the regex."""
    text = text.lower().replace("'", " ").replace('"', " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize_src(text):
    return _SRC_TOKEN.findall(normalize_src(text))


def tokenize_src_veri(text):
    """veriregex preprocessing: placeholders for quoted literals, then lowercase + word tokens"""
    anon, _ = anonymize(text)
    return _SRC_TOKEN.findall(normalize_src(anon))


# regex side: the four placeholder literals and the word boundary '\b' are single tokens,
# every other character is a token on its own.
_TGT_TOKEN = re.compile(r"__w\d__|\\b|dog|truck|ring|lake|.", re.S)


def tokenize_tgt(regex):
    return _TGT_TOKEN.findall(regex)


def detokenize_tgt(tokens):
    return "".join(tokens)


# ------------------------------------------------------------------------ vocabulary
class Vocab:
    def __init__(self, tokens=None):
        self.itos = list(SPECIALS)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        for t in tokens or []:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)

    @classmethod
    def build(cls, sequences, min_freq=1):
        counts = {}
        for seq in sequences:
            for t in seq:
                counts[t] = counts.get(t, 0) + 1
        kept = sorted([t for t, c in counts.items() if c >= min_freq], key=lambda t: (-counts[t], t))
        return cls(kept)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in ids:
            if i == EOS_ID and stop_at_eos:
                break
            if i in (PAD_ID, SOS_ID, EOS_ID):
                continue
            out.append(self.itos[i])
        return out


# ---------------------------------------------------------------------------- loading
def load_pairs(name="NL-RX-Turk"):
    src = (RAW / name / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt = (RAW / name / "targ.txt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(tgt)
    return list(zip(src, tgt))


def build_data(name="NL-RX-Turk", seed=42, val_frac=0.1, test_frac=0.1, min_freq=1, augment_synth=False):
    """Returns dict with vocabularies, the three splits and dataset statistics."""
    pairs = load_pairs(name)
    n_raw = len(pairs)
    # remove exact duplicate (description, regex) pairs so that no pair is in two splits
    # (the Synth and Turk files are line-aligned: same regex, different description)
    synth = load_pairs("NL-RX-Synth") if augment_synth else None
    seen, uniq = set(), []
    for k, (s, t) in enumerate(pairs):
        key = (normalize_src(s), t)
        if key not in seen:
            seen.add(key)
            uniq.append((s, t, k))
    pairs = uniq

    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_test, n_val = int(len(idx) * test_frac), int(len(idx) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def make(ids):
        rows = []
        for i in ids:
            s, t, _ = pairs[i]
            rows.append({"src_raw": s, "tgt_raw": t,
                         "src_tok": tokenize_src(s), "tgt_tok": tokenize_tgt(t)})
        return rows

    splits = {"train": make(train_idx), "val": make(val_idx), "test": make(test_idx)}
    n_train_turk = len(splits["train"])
    if augment_synth:
        # Only the TRAIN indices are augmented, so val/test regexes never appear in training
        # (no leakage): each training regex gets a second, template-generated description.
        have = {(normalize_src(r["src_raw"]), r["tgt_raw"]) for r in splits["train"]}
        for i in train_idx:
            _, t, k = pairs[i]
            s2 = synth[k][0]
            assert synth[k][1] == t
            if (normalize_src(s2), t) not in have:
                have.add((normalize_src(s2), t))
                splits["train"].append({"src_raw": s2, "tgt_raw": t, "src_tok": tokenize_src(s2),
                                        "tgt_tok": tokenize_tgt(t)})
    # vocabularies are built on the TRAINING split only
    src_vocab = Vocab.build([r["src_tok"] for r in splits["train"]], min_freq=min_freq)
    tgt_vocab = Vocab.build([r["tgt_tok"] for r in splits["train"]])

    for split in splits.values():
        for r in split:
            src = r["src_tok"][:MAX_SRC_LEN]
            tgt = r["tgt_tok"][:MAX_TGT_LEN - 1]
            r["src_ids"] = src_vocab.encode(src)
            r["tgt_ids"] = [SOS_ID] + tgt_vocab.encode(tgt) + [EOS_ID]

    train_targets = {r["tgt_raw"] for r in splits["train"]}

    def unk_rate(rows):
        total = sum(len(r["src_ids"]) for r in rows)
        return sum(i == UNK_ID for r in rows for i in r["src_ids"]) / total

    for split in ("val", "test"):
        for r in splits[split]:
            r["unseen_target"] = r["tgt_raw"] not in train_targets

    stats = {
        "dataset": name,
        "raw_pairs": n_raw,
        "after_dedup": len(pairs),
        "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]),
        "src_vocab_size": len(src_vocab), "tgt_vocab_size": len(tgt_vocab),
        "avg_src_tokens": sum(len(r["src_tok"]) for r in splits["train"]) / len(splits["train"]),
        "max_src_tokens": max(len(r["src_tok"]) for s in splits.values() for r in s),
        "avg_tgt_tokens": sum(len(r["tgt_tok"]) for r in splits["train"]) / len(splits["train"]),
        "max_tgt_tokens": max(len(r["tgt_tok"]) for s in splits.values() for r in s),
        "n_truncated_src": sum(len(r["src_tok"]) > MAX_SRC_LEN for s in splits.values() for r in s),
        "n_truncated_tgt": sum(len(r["tgt_tok"]) > MAX_TGT_LEN - 1 for s in splits.values() for r in s),
        "unk_rate_val": unk_rate(splits["val"]), "unk_rate_test": unk_rate(splits["test"]),
        "unique_targets_train": len(train_targets),
        "test_unseen_target_frac": sum(r["unseen_target"] for r in splits["test"]) / len(splits["test"]),
        "val_unseen_target_frac": sum(r["unseen_target"] for r in splits["val"]) / len(splits["val"]),
        "seed": seed, "augment_synth": augment_synth, "train_turk_only": n_train_turk,
    }
    return {"splits": splits, "src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "stats": stats}


def build_veri(seed=42, min_freq=1, **_):
    """veriregex splits are fixed by src/generate.py (split by regex); `seed` is unused."""
    splits = {}
    for name in ("train", "val", "test", "test_phrasing", "test_handwritten"):
        rows = []
        for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            rows.append({"src_raw": ex["text"], "tgt_raw": ex["regex_anon"], "regex": ex["regex"],
                         "src_tok": tokenize_src_veri(ex["text"]), "tgt_tok": tokenize_tgt(ex["regex_anon"]),
                         "kind": ex["kind"], "words": ex["words"]})
        splits[name] = rows
    src_vocab = Vocab.build([r["src_tok"] for r in splits["train"]], min_freq=min_freq)
    tgt_vocab = Vocab.build([r["tgt_tok"] for r in splits["train"]])
    for split in splits.values():
        for r in split:
            r["src_ids"] = src_vocab.encode(r["src_tok"][:MAX_SRC_LEN])
            r["tgt_ids"] = [SOS_ID] + tgt_vocab.encode(r["tgt_tok"][:MAX_TGT_LEN - 1]) + [EOS_ID]
    train_targets = {r["tgt_raw"] for r in splits["train"]}
    for name, split in splits.items():
        for r in split:
            r["unseen_target"] = r["tgt_raw"] not in train_targets

    def unk_rate(rows):
        total = sum(len(r["src_ids"]) for r in rows)
        return sum(i == UNK_ID for r in rows for i in r["src_ids"]) / max(1, total)
    allrows = [r for s in splits.values() for r in s]
    stats = {"dataset": "veriregex", **{k: len(v) for k, v in splits.items()},
             "src_vocab_size": len(src_vocab), "tgt_vocab_size": len(tgt_vocab),
             "avg_src_tokens": sum(len(r["src_tok"]) for r in splits["train"]) / len(splits["train"]),
             "max_src_tokens": max(len(r["src_tok"]) for r in allrows),
             "avg_tgt_tokens": sum(len(r["tgt_tok"]) for r in splits["train"]) / len(splits["train"]),
             "max_tgt_tokens": max(len(r["tgt_tok"]) for r in allrows),
             "n_truncated_src": sum(len(r["src_tok"]) > MAX_SRC_LEN for r in allrows),
             "n_truncated_tgt": sum(len(r["tgt_tok"]) > MAX_TGT_LEN - 1 for r in allrows),
             "unique_targets_train": len(train_targets),
             **{f"unk_rate_{k}": unk_rate(v) for k, v in splits.items() if k != "train"},
             **{f"unseen_target_frac_{k}": sum(r["unseen_target"] for r in v) / max(1, len(v))
                for k, v in splits.items() if k != "train"}}
    return {"splits": splits, "src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "stats": stats}


def get_data(dataset="veriregex", seed=42, augment_synth=False):
    if dataset == "veriregex":
        return build_veri(seed)
    return build_data(dataset, seed=seed, augment_synth=augment_synth)


# ----------------------------------------------------------------------------- batches
def collate(rows):
    src = pad_sequence([torch.tensor(r["src_ids"] or [UNK_ID]) for r in rows],
                       batch_first=True, padding_value=PAD_ID)
    src_len = torch.tensor([max(1, len(r["src_ids"])) for r in rows])
    tgt = pad_sequence([torch.tensor(r["tgt_ids"]) for r in rows],
                       batch_first=True, padding_value=PAD_ID)
    return src, src_len, tgt


def batches(rows, batch_size, shuffle=False, rng=None, bucket=True):
    """Yield (rows, src, src_len, tgt). When shuffling, examples are bucketed by target length inside
    large random chunks so that a batch contains sequences of similar length (less padding, faster)."""
    order = list(range(len(rows)))
    if shuffle:
        (rng or random).shuffle(order)
        if bucket:
            chunk_size = batch_size * 20
            order = [j for i in range(0, len(order), chunk_size)
                     for j in sorted(order[i:i + chunk_size], key=lambda k: len(rows[k]["tgt_ids"]))]
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        (rng or random).shuffle(chunks)
    for ch in chunks:
        chunk = [rows[j] for j in ch]
        yield chunk, *collate(chunk)


if __name__ == "__main__":
    d = get_data(sys.argv[1] if len(sys.argv) > 1 else "veriregex")
    print(json.dumps(d["stats"], indent=2))
    r = d["splits"]["train"][0]
    print(r["src_raw"], "->", r["src_tok"], r["src_ids"])
    print(r["tgt_raw"], "->", r["tgt_tok"], r["tgt_ids"])
    bad = [r for s in d["splits"].values() for r in s if detokenize_tgt(r["tgt_tok"]) != r["tgt_raw"]]
    print("tokenisation round-trip failures:", len(bad))
