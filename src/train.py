"""
Train an encoder-decoder for NL -> regex.

Example:
    python src/train.py --name lstm_base --rnn lstm
    python src/train.py --name lstm_attn --rnn lstm --attention
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import PAD_ID, batches, get_data  # noqa: E402
from evaluate import evaluate_split, teacher_forced_stats  # noqa: E402
from models import Seq2Seq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rnn", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dataset", default="veriregex", help="veriregex (ours) or NL-RX-Turk")
    ap.add_argument("--augment_synth", action="store_true", help="add NL-RX-Synth descriptions of the training regexes")
    ap.add_argument("--word_dropout", type=float, default=0.0, help="prob. of replacing a source word by <UNK> in training")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--tf", type=float, default=1.0, help="teacher-forcing ratio (1.0 = always gold prefix)")
    ap.add_argument("--patience", type=int, default=8, help="early stopping on validation exact match")
    ap.add_argument("--seed", type=int, default=1, help="model/training seed")
    ap.add_argument("--data_seed", type=int, default=42, help="seed of the train/val/test split")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)

    data = get_data(a.dataset, seed=a.data_seed, augment_synth=a.augment_synth)
    sv, tv = len(data["src_vocab"]), len(data["tgt_vocab"])
    train, val = data["splits"]["train"], data["splits"]["val"]
    model = Seq2Seq(sv, tv, a.emb, a.hid, a.layers, a.rnn, a.bidirectional, a.attention, a.dropout, a.word_dropout)
    n_params = sum(p.numel() for p in model.parameters())

    run_dir = ROOT / "results" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(a) | {"n_params": n_params, "src_vocab": sv, "tgt_vocab": tv}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[{a.name}] params={n_params:,}  train={len(train)} val={len(val)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=a.label_smoothing)
    rng = random.Random(a.seed)

    history, best_em, best_epoch, bad = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for bi, (_, src, src_len, tgt) in enumerate(batches(train, a.batch, shuffle=True, rng=rng)):
            if a.max_batches and bi >= a.max_batches:
                break
            logits = model(src, src_len, tgt, a.tf)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            tot += loss.item()
            n += 1
        train_loss = tot / n
        vtf = teacher_forced_stats(model, val)
        vres, *_ = evaluate_split(model, val, data["tgt_vocab"], "greedy", functional=False)
        sched.step(vtf["loss"])
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": vtf["loss"], "val_ppl": vtf["perplexity"],
               "val_token_acc": vtf["token_acc"], "val_em": vres["exact_match"], "val_bleu": vres["bleu"],
               "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        improved = vres["exact_match"] > best_em
        if improved:
            best_em, best_epoch, bad = vres["exact_match"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "model_cfg": model.cfg, "epoch": epoch,
                        "data_seed": a.data_seed, "augment_synth": a.augment_synth,
                        "dataset": a.dataset}, run_dir / "best.pt")
        else:
            bad += 1
        print(f"[{a.name}] ep {epoch:02d} train {train_loss:.3f} | val loss {vtf['loss']:.3f} "
              f"ppl {vtf['perplexity']:.2f} tokacc {vtf['token_acc']:.3f} EM {vres['exact_match']:.3f} "
              f"BLEU {vres['bleu']:.1f} | {rec['sec']:.0f}s{'  *' if improved else ''}", flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        if bad >= a.patience:
            print(f"[{a.name}] early stop (no val-EM gain for {a.patience} epochs)", flush=True)
            break

    summary = {"best_epoch": best_epoch, "best_val_em": best_em, "epochs_run": len(history),
               "train_minutes": (time.time() - t_start) / 60, "n_params": n_params}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{a.name}] done: {summary}", flush=True)


if __name__ == "__main__":
    main()
