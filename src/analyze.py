"""
Aggregate the results of the runs into tables and figures for the report.

    python src/analyze.py base attn       # run folders in results/ (evaluate.py must have been run on each)
Outputs (in results/): comparison.md, learning_curves.png, attention_examples.png,
                       error_analysis.md, qualitative_examples.md
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SPLITS = [("test", "New regexes (test)"), ("test_phrasing", "New phrasings"), ("test_handwritten", "Handwritten")]
LABEL = {"base": "LSTM baseline, 15 epochs", "base_long": "LSTM baseline, 30 epochs",
         "attn": "LSTM + attention", "attn_s2": "LSTM + attention, seed 2"}


def load(name):
    d = RES / name
    r = {"dir": d, "cfg": json.loads((d / "config.json").read_text()),
         "hist": json.loads((d / "history.json").read_text()),
         "summary": json.loads((d / "summary.json").read_text())}
    for sp, _ in SPLITS:
        f = d / f"{sp}_greedy_metrics.json"
        r[sp] = json.loads(f.read_text()) if f.exists() else None
        p = d / f"{sp}_greedy_predictions.jsonl"
        r[sp + "_pred"] = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        b = next(iter(d.glob(f"{sp}_beam*_metrics.json")), None)
        r[sp + "_beam"] = json.loads(b.read_text()) if b else None
    return r


def pct(x):
    return f"{100 * x:.1f}"


def comparison(runs):
    L = ["## Main results (greedy decoding)", "",
         "| Model | Params | Test set | n | Exact match (%) | Same behaviour (%) | BLEU-4 | Valid regex (%) | Perplexity |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        for sp, lab in SPLITS:
            m = r[sp]
            if not m:
                continue
            L.append(f"| {LABEL.get(n, n)} | {r['cfg']['n_params']:,} | {lab} | {m['n']} | {pct(m['exact_match'])} | "
                     f"{pct(m['functional_acc'])} | {m['bleu']:.1f} | {pct(m['valid_rate'])} | {m['perplexity']:.3f} |")
    L += ["", "## By kind of description (test set, exact match / same behaviour, %)", "",
          "| Model | One condition | Negated condition | Two or three conditions with AND | Two conditions with OR |",
          "|---|---|---|---|---|"]
    for n, r in runs.items():
        m = r["test"]
        cell = lambda k: (f"{pct(m[f'em_kind_{k}'])} / {pct(m[f'func_kind_{k}'])} (n={m[f'n_kind_{k}']})"  # noqa: E731
                          if f"em_kind_{k}" in m else "-")
        L.append(f"| {LABEL.get(n, n)} | {cell('atom')} | {cell('not')} | {cell('and')} | {cell('or')} |")
    beams = [(n, r) for n, r in runs.items() if r["test_beam"]]
    if beams:
        L += ["", "## Greedy versus beam search (same trained model)", "",
              "| Model | Test set | Decoding | Exact match (%) | Same behaviour (%) | Valid regex (%) |", "|---|---|---|---|---|---|"]
        for n, r in beams:
            for sp, lab in SPLITS:
                if r[sp + "_beam"]:
                    for dec, m in (("greedy", r[sp]), (r[sp + "_beam"]["mode"], r[sp + "_beam"])):
                        L.append(f"| {LABEL.get(n, n)} | {lab} | {dec} | {pct(m['exact_match'])} | "
                                 f"{pct(m['functional_acc'])} | {pct(m['valid_rate'])} |")
    L += ["", "## Training", "", "| Model | Best epoch | Epochs run | Minutes | Best validation exact match (%) |",
          "|---|---|---|---|---|"]
    for n, r in runs.items():
        s = r["summary"]
        L.append(f"| {LABEL.get(n, n)} | {s['best_epoch']} | {s['epochs_run']} | {s['train_minutes']:.0f} | {pct(s['best_val_em'])} |")
    return "\n".join(L)


def learning_curves(runs):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = {"base": "#E0B04A", "base_long": "#9A5B00", "attn": "#0B6465", "attn_s2": "#5FA8A3"}
    for n, r in runs.items():
        h = r["hist"]
        ep = [x["epoch"] for x in h]
        c = colors.get(n, "gray")
        ax[0].plot(ep, [x["train_loss"] for x in h], "--", color=c, alpha=0.8)
        ax[0].plot(ep, [x["val_loss"] for x in h], "-", color=c, label=LABEL.get(n, n))
        ax[1].plot(ep, [x["val_em"] * 100 for x in h], color=c, marker="o", ms=3, label=LABEL.get(n, n))
        ax[2].plot(ep, [x["val_bleu"] for x in h], color=c, marker="o", ms=3, label=LABEL.get(n, n))
    ax[0].set_title("Loss (dashed = train, solid = validation)")
    ax[1].set_title("Validation exact match (%)")
    ax[2].set_title("Validation BLEU-4")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("token cross-entropy")
    ax[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(RES / "learning_curves.png", dpi=140)
    plt.close()


def tree_kind_of_ref(ref):
    feats = []
    if "~" in ref:
        feats.append("negation (~)")
    if "&" in ref:
        feats.append("AND (&)")
    if "|" in ref:
        feats.append("OR (|)")
    if re.search(r"\{\d+,?\d*\}", ref):
        feats.append("counter {n,}")
    if "__w1__" in ref:
        feats.append("two or more literals")
    return feats


def error_analysis(runs):
    out = ["# Error analysis", ""]
    for n, r in runs.items():
        out.append(f"## {LABEL.get(n, n)}\n")
        for sp, lab in SPLITS:
            rows = r[sp + "_pred"]
            if not rows:
                continue
            agg = defaultdict(lambda: [0, 0, 0])
            for x in rows:
                for f in tree_kind_of_ref(x["ref"]):
                    agg[f][0] += 1
                    agg[f][1] += x["em"]
                    agg[f][2] += bool(x["functional"])
            wrong = [x for x in rows if not x["functional"]]
            eq = [x for x in rows if x["functional"] and not x["em"]]
            out.append(f"### {lab}: {len(rows)} examples, {sum(x['em'] for x in rows)} exact, {len(eq)} equivalent but "
                       f"written differently, {len(wrong)} wrong ({sum(1 for x in wrong if not x['valid'])} invalid)\n")
            out.append("| Feature of the reference | n | Exact (%) | Same behaviour (%) |\n|---|---|---|---|")
            for f, (c, e, fu) in sorted(agg.items()):
                out.append(f"| {f} | {c} | {100 * e / c:.1f} | {100 * fu / c:.1f} |")
            # categorise wrong outputs
            cats = Counter()
            for x in wrong:
                p, g = x["pred"], x["ref"]
                if not x["valid"]:
                    cats["invalid regex"] += 1
                elif re.sub(r"\d", "N", p) == re.sub(r"\d", "N", g):
                    cats["wrong number only"] += 1
                elif re.sub(r"__w\d__", "W", p) == re.sub(r"__w\d__", "W", g):
                    cats["literals swapped or wrong slot"] += 1
                elif p.count("~") != g.count("~"):
                    cats["negation missing or extra"] += 1
                elif ("&" in p) != ("&" in g) or ("|" in p) != ("|" in g):
                    cats["wrong connective (and/or)"] += 1
                elif re.sub(r"\[[^\]]*\]", "C", p) == re.sub(r"\[[^\]]*\]", "C", g):
                    cats["wrong character class"] += 1
                else:
                    cats["other structure error"] += 1
            if wrong:
                out.append("\nWrong outputs by type: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()) + "\n")
    (RES / "error_analysis.md").write_text("\n".join(out))


def qualitative(r, name):
    out = [f"# Qualitative examples ({LABEL.get(name, name)}, greedy)\n"]
    for sp, lab in SPLITS:
        rows = r[sp + "_pred"]
        ok = [x for x in rows if x["em"]][:5]
        eq = [x for x in rows if x["functional"] and not x["em"]][:3]
        bad = [x for x in rows if not x["functional"]][:6]
        out.append(f"## {lab}\n")
        for title, items in (("Correct", ok), ("Written differently, same behaviour", eq), ("Wrong", bad)):
            if not items:
                continue
            out.append(f"### {title}\n")
            for x in items:
                out.append(f"- {x['src']}\n  - reference: `{x['ref']}`\n  - generated: `{x['pred']}`")
            out.append("")
    (RES / "qualitative_examples.md").write_text("\n".join(out))


def attention_figure(name, texts):
    from data import anonymize, collate, get_data, tokenize_src_veri
    from evaluate import load_model
    model, ck = load_model(RES / name)
    data = get_data(ck.get("dataset", "veriregex"))
    sv, tv = data["src_vocab"], data["tgt_vocab"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for ax, text in zip(axes.ravel(), texts):
        toks = tokenize_src_veri(text)
        lits = anonymize(text)[1]
        row = {"src_ids": sv.encode(toks), "tgt_ids": [2, 3]}
        src, src_len, _ = collate([row])
        ids, att = model.greedy_decode(src, src_len, return_attention=True)
        out = tv.decode(ids[0].tolist())
        a = att[0, :len(out)].numpy()
        show = lambda t: f"'{lits[int(t[3])]}'" if re.fullmatch(r"__w\d__", t) else t  # noqa: E731
        ax.imshow(a, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([show(t) for t in toks], rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(out)))
        ax.set_yticklabels([show(t) for t in out], fontsize=8)
        ax.set_title(text, fontsize=10)
    plt.suptitle("Attention weights of the attention model: rows = generated regex tokens, columns = input words", fontsize=11)
    plt.tight_layout()
    plt.savefig(RES / "attention_examples.png", dpi=140)
    plt.close()


ATT_TEXTS = ["lines containing 'cat' at least 3 times", "lines where a vowel comes before a digit",
             "lines that do not start with 'the' and end with a number", "strings with 'blue' somewhere after 'red'"]

if __name__ == "__main__":
    names = sys.argv[1:] or ["base", "base_long", "attn", "attn_s2"]
    runs = {n: load(n) for n in names}
    (RES / "comparison.md").write_text(comparison(runs))
    learning_curves(runs)
    error_analysis(runs)
    best = "attn" if "attn" in runs else names[-1]
    qualitative(runs[best], best)
    attention_figure(best, ATT_TEXTS)
    print((RES / "comparison.md").read_text())
