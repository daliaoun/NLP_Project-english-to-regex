"""
Prediction + evaluation utilities for the NL -> regex models.

Metrics
  * exact match (EM)          : generated regex string identical to the reference
  * functional correctness    : generated regex is valid AND accepts/rejects the same probe strings
                                as the reference (behavioural equivalence, see regex_engine.py)
  * valid-regex rate          : generated string parses as an NL-RX regex
  * BLEU-4 (token level)      : n-gram overlap between generated and reference token sequences
  * token accuracy            : teacher-forced next-token accuracy (diagnostic)
  * perplexity                : exp of the teacher-forced token cross-entropy
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import PAD_ID, batches, collate, detokenize_tgt, get_data  # noqa: E402
import re  # noqa: E402
from models import Seq2Seq  # noqa: E402
from regex_engine import make_test_strings, signature  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------- generation
@torch.no_grad()
def predict(model, rows, tgt_vocab, mode="greedy", beam=5, batch_size=128, keep_attention=False):
    """Returns a list of dicts {tokens, text, attention (optional)} aligned with `rows`."""
    model.eval()
    out = []
    if mode == "greedy":
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            src, src_len, _ = collate(chunk)
            res = model.greedy_decode(src, src_len, return_attention=keep_attention)
            toks, att = res if isinstance(res, tuple) else (res, None)
            for k in range(len(chunk)):
                t = tgt_vocab.decode(toks[k].tolist())
                item = {"tokens": t, "text": detokenize_tgt(t)}
                if att is not None:
                    item["attention"] = att[k, :len(t) + 1, :len(chunk[k]["src_ids"])].cpu()
                out.append(item)
    elif mode == "beam":
        for r in rows:
            src, src_len, _ = collate([r])
            ids = model.beam_search(src, src_len, beam=beam)
            t = tgt_vocab.decode(ids)
            out.append({"tokens": t, "text": detokenize_tgt(t)})
    else:
        raise ValueError(mode)
    return out


# ---------------------------------------------------------------------------- metrics
def corpus_bleu(hyps, refs, max_n=4):
    """Corpus BLEU on token lists with add-one smoothing for n>1 (so short outputs do not give 0)."""
    clipped, total = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            clipped[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[n - 1] += max(0, len(h) - n + 1)
    if hyp_len == 0:
        return 0.0
    logp = 0.0
    for n in range(max_n):
        num, den = clipped[n], total[n]
        if n > 0:
            num, den = num + 1, den + 1
        if num == 0 or den == 0:
            return 0.0
        logp += math.log(num / den) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * math.exp(logp)


@torch.no_grad()
def teacher_forced_stats(model, rows, batch_size=128):
    """Token-level cross-entropy, perplexity and next-token accuracy with gold prefixes."""
    model.eval()
    loss_sum = correct = n_tok = 0
    for _, src, src_len, tgt in batches(rows, batch_size):
        logits = model(src, src_len, tgt, 1.0)
        gold = tgt[:, 1:]
        mask = gold != PAD_ID
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
                                    ignore_index=PAD_ID, reduction="sum").item()
        correct += ((logits.argmax(-1) == gold) & mask).sum().item()
        n_tok += mask.sum().item()
    loss = loss_sum / n_tok
    return {"loss": loss, "perplexity": math.exp(loss), "token_acc": correct / n_tok}


PROBE_WORDS = ["dog", "truck", "ring"]   # placeholders are replaced by these words for behavioural tests


def probe_form(rx):
    return re.sub(r"__w(\d)__", lambda m: PROBE_WORDS[int(m.group(1))] if int(m.group(1)) < 3 else "lake", rx)


def evaluate_split(model, rows, tgt_vocab, mode="greedy", beam=5, functional=True, probe_strings=None):
    preds = predict(model, rows, tgt_vocab, mode, beam)
    refs = [r["tgt_raw"] for r in rows]
    em = [p["text"] == ref for p, ref in zip(preds, refs)]
    res = {"mode": mode if mode == "greedy" else f"beam{beam}", "n": len(rows),
           "exact_match": sum(em) / len(rows),
           "bleu": corpus_bleu([p["tokens"] for p in preds], [r["tgt_tok"] for r in rows])}
    func = valid = degen = None
    if functional:
        probe_strings = probe_strings or make_test_strings()
        ref_sigs = {}
        func, valid, degen = [], [], []
        for p, ref in zip(preds, refs):
            if ref not in ref_sigs:
                ref_sigs[ref] = signature(probe_form(ref), probe_strings)
            # a reference that accepts none (or all) of the probes is "degenerate": a prediction that
            # also accepts none (or all) is then judged equivalent too, which inflates functional accuracy
            degen.append(sum(ref_sigs[ref]) in (0, len(probe_strings)))
            sig = signature(probe_form(p["text"]), probe_strings)
            valid.append(sig is not None)
            func.append(sig is not None and sig == ref_sigs[ref])
        res["functional_acc"] = sum(func) / len(rows)
        res["valid_rate"] = sum(valid) / len(rows)
        nd = [i for i, d in enumerate(degen) if not d]
        res["n_nondegenerate_ref"] = len(nd)
        res["functional_acc_nondegenerate"] = sum(func[i] for i in nd) / max(1, len(nd))
        res["em_nondegenerate"] = sum(em[i] for i in nd) / max(1, len(nd))
    return res, preds, em, func, valid, degen


def full_report(model, data, split="test", mode="greedy", beam=5, functional=True):
    rows = data["splits"][split]
    tgt_vocab = data["tgt_vocab"]
    tf = teacher_forced_stats(model, rows)
    res, preds, em, func, valid, degen = evaluate_split(model, rows, tgt_vocab, mode, beam, functional)
    res.update({"loss": tf["loss"], "perplexity": tf["perplexity"], "token_acc": tf["token_acc"]})
    # break-down: regexes never seen in training vs regexes that also occur in the training set
    unseen = [i for i, r in enumerate(rows) if r.get("unseen_target")]
    seen = [i for i, r in enumerate(rows) if not r.get("unseen_target")]
    res["n_unseen_target"], res["n_seen_target"] = len(unseen), len(seen)
    if unseen:
        res["em_unseen_target"] = sum(em[i] for i in unseen) / len(unseen)
        if func:
            res["func_unseen_target"] = sum(func[i] for i in unseen) / len(unseen)
    if seen:
        res["em_seen_target"] = sum(em[i] for i in seen) / len(seen)
    # break-down by kind of meaning (veriregex): single condition, negation, and, or
    kinds = sorted({r.get("kind") for r in rows if r.get("kind")})
    for k in kinds:
        ids = [i for i, r in enumerate(rows) if r.get("kind") == k]
        res[f"em_kind_{k}"] = sum(em[i] for i in ids) / len(ids)
        if func:
            res[f"func_kind_{k}"] = sum(func[i] for i in ids) / len(ids)
        res[f"n_kind_{k}"] = len(ids)
    per_example = [{"src": r["src_raw"], "ref": r["tgt_raw"], "pred": p["text"], "em": e, "kind": r.get("kind"),
                    "functional": (func[i] if func else None), "valid": (valid[i] if valid else None),
                    "ref_degenerate": (degen[i] if degen else None),
                    "unseen_target": r.get("unseen_target")}
                   for i, (r, p, e) in enumerate(zip(rows, preds, em))]
    return res, per_example


def load_model(run_dir, device="cpu"):
    ck = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    model = Seq2Seq(**ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="all", help="test, test_phrasing, test_handwritten or all")
    ap.add_argument("--beam", type=int, default=0, help="0 = greedy only; k>0 also evaluates beam search")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, ck = load_model(a.run_dir)
    dataset = ck.get("dataset", "NL-RX-Turk")
    data = get_data(dataset, seed=ck["data_seed"], augment_synth=ck.get("augment_synth", False))
    splits = ([s for s in ("test", "test_phrasing", "test_handwritten") if s in data["splits"]]
              if a.split == "all" else [a.split])
    for split in splits:
        for mode, beam in [("greedy", 0)] + ([("beam", a.beam)] if a.beam else []):
            res, per_ex = full_report(model, data, split, mode, beam or 5)
            name = f"{split}_{res['mode']}"
            (Path(a.run_dir) / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
            (Path(a.run_dir) / f"{name}_predictions.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in per_ex))
            print(name, json.dumps({k: v for k, v in res.items() if not k.startswith("n_")}, indent=1))
